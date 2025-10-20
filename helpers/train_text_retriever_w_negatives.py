import json
import logging
import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, Set, List, Optional, Callable, Any, Tuple
import traceback
import pretty_errors
import random
import glob

from sklearn.model_selection import train_test_split

import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import set_seed, AutoTokenizer
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import (
    LoggingHandler,
    util, 
    models, 
    losses, 
    InputExample
)

from sentence_transformers import SentenceTransformer, InformationRetrievalEvaluator
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Set up environment
load_dotenv()
app_path = os.environ.get("APP_PATH", os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(app_path))

# Configure detailed logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Prevent propagation to root logger to avoid duplicate messages
logger.propagate = False

# Add file handler if LOG_FILE env variable is set
log_file = os.environ.get("LOG_FILE", "")
if log_file:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handler)

def __clean_text(text: str) -> str:
    """
    Clean text by removing extra spaces and newlines.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    # Handle case where text is actually a list
    if isinstance(text, list):
        if len(text) > 0:
            text = str(text[0])  # Take first element if it's a list
        else:
            return ""  # Return empty string if list is empty
    
    # Handle None values
    if text is None:
        return ""
        
    # Now proceed with normal string cleaning
    return str(text).strip()


class TripletDataset(Dataset):
    """
    A dataset for contrastive training with triplets of query, positive, and negative examples.
    
    This dataset formats samples as input examples for the SentenceTransformer model's
    training process using triplet loss.
    
    Attributes:
        samples (list): List of triplet samples (query, positive, negative)
    """
    
    def __init__(self, 
                 samples: List[Tuple[str, str, str]]):
        """
        Initialize the dataset with triplet samples.
        
        Args:
            samples (List[Tuple[str, str, str]]): List of (query, positive, negative) text triplets
        """
        self.samples = samples
        logger.info(f"Created dataset with {len(samples)} triplet examples")
        
        # Log some statistics about the dataset
        query_lengths = [len(q) for q, _, _ in samples[:1000]]  # Sample first 1000 for statistics
        pos_lengths = [len(p) for _, p, _ in samples[:1000]]
        neg_lengths = [len(n) for _, _, n in samples[:1000]]
        
        logger.info(f"Query text length stats: min={min(query_lengths)}, max={max(query_lengths)}, "
                   f"avg={sum(query_lengths)/len(query_lengths):.1f} chars")
        logger.info(f"Positive text length stats: min={min(pos_lengths)}, max={max(pos_lengths)}, "
                   f"avg={sum(pos_lengths)/len(pos_lengths):.1f} chars")
        logger.info(f"Negative text length stats: min={min(neg_lengths)}, max={max(neg_lengths)}, "
                   f"avg={sum(neg_lengths)/len(neg_lengths):.1f} chars")

    def __getitem__(self, idx):
        """
        Get a triplet at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            InputExample: Formatted input example for the SentenceTransformer model
        """
        query_text, pos_text, neg_text = self.samples[idx]
        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        """
        Get the total number of triplets in the dataset.
        
        Returns:
            int: Number of triplet samples
        """
        return len(self.samples)


def load_tourism_qa_data(dataset_name: str,
                         max_train_samples: int = None,
                         num_negatives: int = 5,
                         use_hard_negatives: bool = True,
                         hard_negative_ratio: float = 0.7,
                         hard_negative_model: SentenceTransformer = None,
                         negative_mining_strategy: str = "hard") -> Tuple[List[Tuple[str, str, str]], Dict[str, str], List[Tuple[str, str, str]]]:
    """
    Load Tourism QA dataset and prepare triplets for retrieval training.
    
    Args:
        dataset_name (str): HuggingFace dataset name for tourism QA data
        max_train_samples (int, optional): Maximum number of training samples to use.
        num_negatives (int, optional): Number of negative examples to use per query-positive pair. Defaults to 5.
        use_hard_negatives (bool): Whether to use hard negative mining. Defaults to True.
        hard_negative_ratio (float): Ratio of hard negatives vs random negatives. Defaults to 0.7.
        hard_negative_model (SentenceTransformer, optional): Model to use for hard negative mining. If None and use_hard_negatives is True, will fall back to random negatives.
        negative_mining_strategy (str): Strategy for negative mining. Options: "hard", "random". Defaults to "hard".
        
    Returns:
        Tuple containing:
            - List of (question, source_chunk, negative_source_chunk) triplets for training
            - Dictionary mapping chunk IDs to source chunks (corpus)
            - List of (question, source_chunk, negative_source_chunk) triplets for testing
    """
    start_time = time.time()
    logger.info(f"Loading tourism QA dataset {dataset_name}")
    logger.info(f"Negative mining strategy: {negative_mining_strategy}")
    
    # Override use_hard_negatives based on strategy
    if negative_mining_strategy == "random":
        use_hard_negatives = False
        logger.info("Using random negative sampling strategy - hard negative mining disabled")
    elif negative_mining_strategy == "hard":
        logger.info("Using hard negative mining strategy")
    else:
        logger.warning(f"Unknown negative mining strategy '{negative_mining_strategy}', defaulting to 'hard'")
        negative_mining_strategy = "hard"

    # Create mappings
    question_to_sources = {}  # {"question": set of source_chunks}
    source_to_questions = {}  # {"source_chunk": set of questions}
    all_source_chunks = set()  # All unique source chunks
    source_chunk_corpus = {}  # {"chunk_id": "source_chunk_text"}
    
    # Track source field mapping for negative sampling
    source_chunk_to_source = {}  # {"source_chunk": "source"}
    source_to_source_chunks = {}  # {"source": set of source_chunks}
    
    # NEW: Track chunk to question mapping for improved negative mining
    chunk_to_questions_map = {}  # {"source_chunk": set of questions}
    
    try:
        # Load dataset with train and test splits
        logger.info("Loading dataset from HuggingFace...")
        dataset = load_dataset(dataset_name)
        logger.info(f"Dataset loaded successfully with splits: {', '.join(dataset.keys())}")
        
        train_data = []
        test_data = []
        
        # Process train split with progress bar
        if 'train' in dataset:
            logger.info(f"Processing train split with {len(dataset['train'])} items")
            train_items = list(dataset['train'])
            for i, item in enumerate(tqdm(train_items, desc="Processing train data", unit="items")):
                train_data.append({
                    'question': item['question'],
                    'source_chunk': item['source_chunk'],
                    'source': item.get('source', ''),  # Add source field
                    'sample_id': item.get('sample_id', f"train_{i}")
                })
        
        # Process test split with progress bar
        if 'test' in dataset:
            logger.info(f"Processing test split with {len(dataset['test'])} items")
            test_items = list(dataset['test'])
            for i, item in enumerate(tqdm(test_items, desc="Processing test data", unit="items")):
                test_data.append({
                    'question': item['question'],
                    'source_chunk': item['source_chunk'],
                    'source': item.get('source', ''),  # Add source field
                    'sample_id': item.get('sample_id', f"test_{i}")
                })
        
        # Build mappings from both train and test data with progress bar
        all_data = train_data + test_data
        chunk_counter = 0
        
        logger.info("Building data mappings...")
        for item in tqdm(all_data, desc="Building mappings", unit="items"):
            question = __clean_text(item["question"])
            source_chunk = __clean_text(item["source_chunk"])
            source = __clean_text(item["source"])
            
            if not question or not source_chunk:
                continue
            
            # Track question -> source_chunks mapping
            if question not in question_to_sources:
                question_to_sources[question] = set()
            question_to_sources[question].add(source_chunk)
            
            # Track source_chunk -> questions mapping
            if source_chunk not in source_to_questions:
                source_to_questions[source_chunk] = set()
            source_to_questions[source_chunk].add(question)
            
            # NEW: Track chunk to questions mapping for negative mining
            if source_chunk not in chunk_to_questions_map:
                chunk_to_questions_map[source_chunk] = set()
            chunk_to_questions_map[source_chunk].add(question)
            
            # Track source_chunk -> source mapping
            if source:
                source_chunk_to_source[source_chunk] = source
                if source not in source_to_source_chunks:
                    source_to_source_chunks[source] = set()
                source_to_source_chunks[source].add(source_chunk)
            
            # Add to corpus and track all chunks
            all_source_chunks.add(source_chunk)
            chunk_id = f"chunk_{chunk_counter}"
            source_chunk_corpus[chunk_id] = source_chunk
            chunk_counter += 1
    
    except Exception as e:
        logger.error(f"Failed to load tourism QA dataset: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Remove duplicate chunks from corpus 
    unique_source_chunks = list(set(source_chunk_corpus.values()))
    # Rebuild corpus with proper chunk_id -> chunk_text mapping
    final_corpus = {}
    for i, chunk in enumerate(unique_source_chunks):
        final_corpus[f"chunk_{i}"] = chunk
    
    logger.info(f"Found {len(question_to_sources)} unique questions")
    logger.info(f"Found {len(final_corpus)} unique source chunks in corpus")
    logger.info(f"Found {len(source_to_source_chunks)} unique sources")
    
    # VALIDATION: Check for potential issues
    logger.info("Running validation checks...")
    
    # Check 1: Questions with multiple source chunks
    multi_source_questions = [(q, len(sources)) for q, sources in question_to_sources.items() if len(sources) > 1]
    if multi_source_questions:
        logger.warning(f"Found {len(multi_source_questions)} questions with multiple source chunks:")
        for q, count in multi_source_questions[:5]:  # Show first 5
            logger.warning(f"  Question: '{q[:100]}...' has {count} source chunks")
    
    # Check 2: Source chunks with multiple questions
    multi_question_chunks = [(chunk, len(questions)) for chunk, questions in source_to_questions.items() if len(questions) > 1]
    if multi_question_chunks:
        logger.warning(f"Found {len(multi_question_chunks)} source chunks with multiple questions:")
        for chunk, count in multi_question_chunks[:5]:  # Show first 5
            logger.warning(f"  Chunk: '{chunk[:100]}...' has {count} questions")
    
    # Check if hard negative mining is enabled and model is provided
    if use_hard_negatives and hard_negative_model is not None:
        logger.info("Using provided model for improved hard negative mining")
    elif use_hard_negatives and hard_negative_model is None:
        logger.warning("Hard negative mining requested but no model provided. Falling back to random negatives.")
        use_hard_negatives = False
    
    # OPTIMIZED: Pre-compute embeddings cache for efficient hard negative mining
    embedding_cache = {}
    
    def get_or_compute_embedding(text: str, model: SentenceTransformer, cache: dict) -> torch.Tensor:
        """Get embedding from cache or compute and cache it."""
        if text not in cache:
            cache[text] = model.encode(text, convert_to_tensor=True)
        return cache[text]
    
    def get_improved_hard_negatives_batch(samples_data: List[dict], 
                                         model: SentenceTransformer, 
                                         chunk_to_questions: Dict[str, set],
                                         embedding_cache: Dict[str, torch.Tensor]) -> Dict[int, List[str]]:
        """
        Efficiently select hard negatives for a batch of samples using cached embeddings.
        
        Args:
            samples_data: List of dicts with keys: 'sample_idx', 'anchor_question', 'positive_chunk', 
                         'available_negatives', 'num_hard'
            model: Sentence transformer model for computing embeddings
            chunk_to_questions: Mapping of chunks to their associated questions
            embedding_cache: Cache for computed embeddings
            
        Returns:
            Dict mapping sample_idx to list of selected hard negative chunks
        """
        if not samples_data:
            return {}
        
        try:
            results = {}
            
            # Collect all unique texts that need embeddings
            all_questions = set()
            all_chunks = set()
            
            for sample in samples_data:
                all_questions.add(sample['anchor_question'])
                all_chunks.add(sample['positive_chunk'])
                all_chunks.update(sample['available_negatives'])
                
                # Add questions associated with negative chunks
                for neg_chunk in sample['available_negatives']:
                    chunk_questions = chunk_to_questions.get(neg_chunk, set())
                    all_questions.update(chunk_questions)
            
            # Pre-compute embeddings for all unique texts (batch processing)
            logger.debug(f"Computing embeddings for {len(all_questions)} questions and {len(all_chunks)} chunks")
            
            # Compute question embeddings in batches
            new_questions = [q for q in all_questions if q not in embedding_cache]
            if new_questions:
                question_embeddings = model.encode(new_questions, convert_to_tensor=True, batch_size=32, show_progress_bar=False)
                for i, question in enumerate(new_questions):
                    embedding_cache[question] = question_embeddings[i]
            
            # Compute chunk embeddings in batches
            new_chunks = [c for c in all_chunks if c not in embedding_cache]
            if new_chunks:
                chunk_embeddings = model.encode(new_chunks, convert_to_tensor=True, batch_size=32, show_progress_bar=False)
                for i, chunk in enumerate(new_chunks):
                    embedding_cache[chunk] = chunk_embeddings[i]
            
            # Process each sample efficiently
            for sample in samples_data:
                sample_idx = sample['sample_idx']
                anchor_question = sample['anchor_question']
                positive_chunk = sample['positive_chunk']
                available_negatives = sample['available_negatives']
                num_hard = sample['num_hard']
                
                if not available_negatives or num_hard <= 0:
                    results[sample_idx] = []
                    continue
                
                # Get cached embeddings
                anchor_question_embedding = embedding_cache[anchor_question]
                positive_chunk_embedding = embedding_cache[positive_chunk]
                
                # Prepare negative chunk data
                negative_chunks_embeddings = []
                negative_questions_embeddings = []
                chunk_to_question_text = {}
                
                for neg_chunk in available_negatives:
                    # Get chunk embedding
                    negative_chunks_embeddings.append(embedding_cache[neg_chunk])
                    
                    # Get representative question for this chunk
                    chunk_questions = chunk_to_questions.get(neg_chunk, set())
                    if chunk_questions:
                        representative_question = list(chunk_questions)[0]
                        chunk_to_question_text[neg_chunk] = representative_question
                        negative_questions_embeddings.append(embedding_cache[representative_question])
                    else:
                        chunk_to_question_text[neg_chunk] = ""
                        # Use zero embedding for empty questions
                        zero_embedding = torch.zeros_like(anchor_question_embedding)
                        negative_questions_embeddings.append(zero_embedding)
                
                # Stack embeddings for batch computation
                if negative_chunks_embeddings:
                    negative_chunks_tensor = torch.stack(negative_chunks_embeddings)
                    negative_questions_tensor = torch.stack(negative_questions_embeddings)
                    
                    # Compute similarities in batch
                    question_similarities = util.cos_sim(anchor_question_embedding.unsqueeze(0), negative_questions_tensor).cpu().numpy()[0]
                    chunk_similarities = util.cos_sim(positive_chunk_embedding.unsqueeze(0), negative_chunks_tensor).cpu().numpy()[0]
                    
                    # Create ranking scores
                    ranking_scores = []
                    for i, neg_chunk in enumerate(available_negatives):
                        question_sim = question_similarities[i]
                        chunk_sim = chunk_similarities[i]
                        
                        # NEW SCORING LOGIC:
                        # We want: LOW question similarity + HIGH chunk similarity
                        # Score = chunk_similarity - question_similarity
                        # Higher score = better hard negative (similar content, different question)
                        combined_score = chunk_sim - question_sim
                        
                        ranking_scores.append({
                            'chunk': neg_chunk,
                            'question_sim': question_sim,
                            'chunk_sim': chunk_sim,
                            'combined_score': combined_score,
                            'negative_question': chunk_to_question_text[neg_chunk]
                        })
                    
                    # Sort by combined score (descending)
                    ranking_scores.sort(key=lambda x: x['combined_score'], reverse=True)
                    
                    # Select top hard negatives
                    selected_hard = [item['chunk'] for item in ranking_scores[:num_hard]]
                    results[sample_idx] = selected_hard
                    
                    # Enhanced logging for first few samples
                    if sample_idx < 3:
                        logger.debug(f"Hard negative mining for sample {sample_idx}:")
                        logger.debug(f"  Question: '{anchor_question[:60]}...'")
                        logger.debug(f"  Selected {len(selected_hard)} hard negatives")
                        for idx, item in enumerate(ranking_scores[:min(2, len(ranking_scores))]):
                            logger.debug(f"    Top negative #{idx+1}: Q_sim={item['question_sim']:.3f}, "
                                       f"C_sim={item['chunk_sim']:.3f}, Score={item['combined_score']:.3f}")
                else:
                    results[sample_idx] = []
            
            return results
            
        except Exception as e:
            logger.warning(f"Error in batch hard negative mining: {e}. Falling back to random selection.")
            # Return random selections as fallback
            fallback_results = {}
            for sample in samples_data:
                sample_idx = sample['sample_idx']
                available_negatives = sample['available_negatives']
                num_hard = sample['num_hard']
                if available_negatives and num_hard > 0:
                    fallback_results[sample_idx] = random.sample(available_negatives, min(num_hard, len(available_negatives)))
                else:
                    fallback_results[sample_idx] = []
            return fallback_results
    

    # Create train triplets 
    train_triplets = []
    train_questions = set(item["question"] for item in train_data)
    
    logger.info("Creating train triplets...")
    
    # Prepare batch data for hard negative mining
    batch_samples_data = []
    sample_to_data_map = {}  # Map sample_idx to original data for triplet creation
    
    for sample_idx, item in enumerate(tqdm(train_data, desc="Preparing negative candidates", unit="samples")):
        question = __clean_text(item["question"])
        source_chunk = __clean_text(item["source_chunk"])
        source = __clean_text(item["source"])
        
        if not question or not source_chunk:
            continue
        
        # Get available negative sources with proper filtering
        available_negative_sources = []
        positive_source = source_chunk_to_source.get(source_chunk, '')
        
        # Get all questions that share the same source chunk as the current positive
        positive_chunk_questions = source_to_questions.get(source_chunk, set())
        
        for other_source_chunk in unique_source_chunks:
            # Skip if it's the same source chunk
            if other_source_chunk == source_chunk:
                continue
            
            # Skip if it belongs to the same source document
            other_source = source_chunk_to_source.get(other_source_chunk, '')
            if positive_source and other_source and positive_source == other_source:
                continue
            
            # Skip if ANY question associated with this chunk is the same as current question
            # OR if this chunk is associated with ANY question that also uses the current positive chunk
            other_chunk_questions = source_to_questions.get(other_source_chunk, set())
            
            # Check for question overlap - if any question is shared, skip this as negative
            if positive_chunk_questions.intersection(other_chunk_questions):
                continue
            
            # If we get here, this chunk is a valid negative
            available_negative_sources.append(other_source_chunk)
        
        if not available_negative_sources:
            logger.warning(f"No valid negative sources available for question: {question[:50]}...")
            logger.warning(f"  Positive chunk has {len(positive_chunk_questions)} associated questions")
            logger.warning(f"  Total unique chunks: {len(unique_source_chunks)}")
            continue
        
        # Store data for batch processing
        sample_to_data_map[sample_idx] = {
            'question': question,
            'source_chunk': source_chunk,
            'source': source,
            'available_negative_sources': available_negative_sources
        }
        
        # Prepare for hard negative mining if enabled
        if use_hard_negatives and hard_negative_model and len(available_negative_sources) > num_negatives:
            num_hard_negatives = int(num_negatives * hard_negative_ratio)
            
            batch_samples_data.append({
                'sample_idx': sample_idx,
                'anchor_question': question,
                'positive_chunk': source_chunk,
                'available_negatives': available_negative_sources,
                'num_hard': num_hard_negatives
            })
    
    # Run batch hard negative mining
    hard_negatives_results = {}
    if batch_samples_data and use_hard_negatives and hard_negative_model:
        logger.info(f"Running batch hard negative mining for {len(batch_samples_data)} samples...")
        
        # Process in chunks to avoid memory issues
        chunk_size = 100
        for i in tqdm(range(0, len(batch_samples_data), chunk_size), desc="Mining hard negatives", unit="batches"):
            chunk_data = batch_samples_data[i:i + chunk_size]
            chunk_results = get_improved_hard_negatives_batch(
                chunk_data, hard_negative_model, chunk_to_questions_map, embedding_cache
            )
            hard_negatives_results.update(chunk_results)
    
    # Create final triplets with validation
    logger.info("Creating final triplets...")
    triplet_validation_errors = 0
    
    for sample_idx, data in tqdm(sample_to_data_map.items(), desc="Creating triplets", unit="samples"):
        question = data['question']
        source_chunk = data['source_chunk']
        available_negative_sources = data['available_negative_sources']
        
        # Select negatives
        selected_negatives = []
        
        if use_hard_negatives and sample_idx in hard_negatives_results:
            # Use hard negatives + random negatives
            num_hard_negatives = int(num_negatives * hard_negative_ratio)
            num_random_negatives = num_negatives - num_hard_negatives
            
            hard_negatives = hard_negatives_results[sample_idx]
            
            # Get random negatives from remaining candidates
            remaining_candidates = [neg for neg in available_negative_sources if neg not in hard_negatives]
            if remaining_candidates and num_random_negatives > 0:
                random_negatives = random.sample(
                    remaining_candidates, 
                    min(num_random_negatives, len(remaining_candidates))
                )
            else:
                random_negatives = []
            
            selected_negatives = hard_negatives + random_negatives
            
            # Fill up to num_negatives if needed
            if len(selected_negatives) < num_negatives:
                additional_needed = num_negatives - len(selected_negatives)
                additional_candidates = [neg for neg in available_negative_sources 
                                       if neg not in selected_negatives]
                if additional_candidates:
                    additional_negatives = random.sample(
                        additional_candidates,
                        min(additional_needed, len(additional_candidates))
                    )
                    selected_negatives.extend(additional_negatives)
        else:
            # Use random sampling
            if len(available_negative_sources) >= num_negatives:
                selected_negatives = random.sample(available_negative_sources, num_negatives)
            else:
                selected_negatives = available_negative_sources
                logger.warning(f"Only {len(available_negative_sources)} negatives available, requested {num_negatives}")
        
        # VALIDATION: Check each triplet before adding
        for negative_source in selected_negatives:
            # Validate that negative doesn't share questions with positive
            positive_questions = source_to_questions.get(source_chunk, set())
            negative_questions = source_to_questions.get(negative_source, set())
            
            if positive_questions.intersection(negative_questions):
                triplet_validation_errors += 1
                logger.error(f"VALIDATION ERROR: Negative shares questions with positive!")
                logger.error(f"  Question: {question[:50]}...")
                logger.error(f"  Positive chunk: {source_chunk[:50]}...")
                logger.error(f"  Negative chunk: {negative_source[:50]}...")
                logger.error(f"  Shared questions: {positive_questions.intersection(negative_questions)}")
                continue
            
            # Add valid triplet
            train_triplets.append((question, source_chunk, negative_source))
    
    if triplet_validation_errors > 0:
        logger.error(f"Found {triplet_validation_errors} validation errors in triplet creation!")
    
    # Create test triplets using same logic
    test_triplets = []
    test_questions = set(item["question"] for item in test_data)
    
    logger.info("Creating test triplets...")
    
    # Same optimization for test data with egative selection
    test_batch_samples_data = []
    test_sample_to_data_map = {}
    
    for sample_idx, item in enumerate(tqdm(test_data, desc="Preparing test negative candidates", unit="samples")):
        question = __clean_text(item["question"])
        source_chunk = __clean_text(item["source_chunk"])
        source = __clean_text(item["source"])
        
        if not question or not source_chunk:
            continue
        
        # Get available negative sources with proper filtering
        available_negative_sources = []
        positive_source = source_chunk_to_source.get(source_chunk, '')
        positive_chunk_questions = source_to_questions.get(source_chunk, set())
        
        for other_source_chunk in unique_source_chunks:
            if other_source_chunk == source_chunk:
                continue
            
            other_source = source_chunk_to_source.get(other_source_chunk, '')
            if positive_source and other_source and positive_source == other_source:
                continue
            
            # Skip if questions overlap
            other_chunk_questions = source_to_questions.get(other_source_chunk, set())
            if positive_chunk_questions.intersection(other_chunk_questions):
                continue
            
            available_negative_sources.append(other_source_chunk)
        
        if not available_negative_sources:
            continue
        
        test_sample_to_data_map[sample_idx] = {
            'question': question,
            'source_chunk': source_chunk,
            'source': source,
            'available_negative_sources': available_negative_sources
        }
        
        # Prepare for hard negative mining
        if use_hard_negatives and hard_negative_model and len(available_negative_sources) > num_negatives:
            num_hard_negatives = int(num_negatives * hard_negative_ratio)
            
            test_batch_samples_data.append({
                'sample_idx': sample_idx,
                'anchor_question': question,
                'positive_chunk': source_chunk,
                'available_negatives': available_negative_sources,
                'num_hard': num_hard_negatives
            })
    
    # Run batch hard negative mining for test data
    test_hard_negatives_results = {}
    if test_batch_samples_data and use_hard_negatives and hard_negative_model:
        logger.info(f"Running batch hard negative mining for {len(test_batch_samples_data)} test samples...")
        
        chunk_size = 100
        for i in tqdm(range(0, len(test_batch_samples_data), chunk_size), desc="Mining test hard negatives", unit="batches"):
            chunk_data = test_batch_samples_data[i:i + chunk_size]
            chunk_results = get_improved_hard_negatives_batch(
                chunk_data, hard_negative_model, chunk_to_questions_map, embedding_cache
            )
            test_hard_negatives_results.update(chunk_results)
    
    # Create final test triplets with same validation
    logger.info("Creating final test triplets...")
    test_triplet_validation_errors = 0
    
    for sample_idx, data in tqdm(test_sample_to_data_map.items(), desc="Creating test triplets", unit="samples"):
        question = data['question']
        source_chunk = data['source_chunk']
        available_negative_sources = data['available_negative_sources']
        
        # Select negatives (same logic as train)
        selected_negatives = []
        
        if use_hard_negatives and sample_idx in test_hard_negatives_results:
            num_hard_negatives = int(num_negatives * hard_negative_ratio)
            num_random_negatives = num_negatives - num_hard_negatives
            
            hard_negatives = test_hard_negatives_results[sample_idx]
            
            remaining_candidates = [neg for neg in available_negative_sources if neg not in hard_negatives]
            if remaining_candidates and num_random_negatives > 0:
                random_negatives = random.sample(
                    remaining_candidates, 
                    min(num_random_negatives, len(remaining_candidates))
                )
            else:
                random_negatives = []
            
            selected_negatives = hard_negatives + random_negatives
            
            if len(selected_negatives) < num_negatives:
                additional_needed = num_negatives - len(selected_negatives)
                additional_candidates = [neg for neg in available_negative_sources 
                                       if neg not in selected_negatives]
                if additional_candidates:
                    additional_negatives = random.sample(
                        additional_candidates,
                        min(additional_needed, len(additional_candidates))
                    )
                    selected_negatives.extend(additional_negatives)
        else:
            if len(available_negative_sources) >= num_negatives:
                selected_negatives = random.sample(available_negative_sources, num_negatives)
            else:
                selected_negatives = available_negative_sources
        
        # VALIDATION: Check each test triplet
        for negative_source in selected_negatives:
            positive_questions = source_to_questions.get(source_chunk, set())
            negative_questions = source_to_questions.get(negative_source, set())
            
            if positive_questions.intersection(negative_questions):
                test_triplet_validation_errors += 1
                logger.error(f"TEST VALIDATION ERROR: Negative shares questions with positive!")
                continue
            
            test_triplets.append((question, source_chunk, negative_source))
    
    if test_triplet_validation_errors > 0:
        logger.error(f"Found {test_triplet_validation_errors} validation errors in test triplet creation!")
    
    logger.info(f"Created {len(train_triplets)} train triplets from {len(train_data)} train samples")
    logger.info(f"Created {len(test_triplets)} test triplets from {len(test_data)} test samples")
    logger.info(f"Final corpus size: {len(final_corpus)} unique chunks")
    logger.info(f"Embedding cache size: {len(embedding_cache)} unique texts")
    
    # Log question-source relationships
    multi_source_questions = [q for q, sources in question_to_sources.items() if len(sources) > 1]
    logger.info(f"Found {len(multi_source_questions)} questions with multiple source chunks")
    
    # Apply max_train_samples limit if specified
    if max_train_samples and max_train_samples < len(train_triplets):
        limited_triplets = random.sample(train_triplets, max_train_samples)
        logger.info(f"Limited train triplets from {len(train_triplets)} to {max_train_samples}")
        train_triplets = limited_triplets
    
    elapsed_time = time.time() - start_time
    logger.info(f"Tourism QA data preparation completed in {elapsed_time:.2f} seconds")
    
    return train_triplets, final_corpus, test_triplets


def create_evaluator(triplets: List[Tuple[str, str, str]], 
                     corpus: Dict[str, str], # Pass the full corpus
                     batch_size: int = 4) -> InformationRetrievalEvaluator:
    """
    Convert a list of triplets (question, positive_source, negative_source) to the format required by
    InformationRetrievalEvaluator, handling multiple positives per question.
    
    Args:
        triplets: List of tuples (question, positive_source, negative_source)
        corpus: The full dictionary of all possible documents (chunk_id -> chunk_text)
        batch_size: Batch size for the evaluator

    Returns:
        InformationRetrievalEvaluator: Evaluator for information retrieval
    """
    queries = {}
    relevant_docs = {}
    
    question_to_positives = {}
    
    # Step 1: Group all positive documents by their unique question.
    # This correctly handles questions with multiple associated positive chunks.
    for question, positive_source, _ in triplets:
        if question not in question_to_positives:
            question_to_positives[question] = set()
        question_to_positives[question].add(positive_source)

    logger.info(f"Evaluator: Found {len(question_to_positives)} unique queries.")
    # This log will now correctly report an average > 1.00 if multi-positive questions exist.
    avg_positives = sum(len(pos) for pos in question_to_positives.values()) / len(question_to_positives)
    logger.info(f"Average positives per query: {avg_positives:.2f}")

    # Create a reverse map from chunk text to its ID from the master corpus
    chunk_to_doc_id = {text: doc_id for doc_id, text in corpus.items()}
    
    # Step 2: Build the final queries and relevant_docs dictionaries for the evaluator.
    for query_id_num, (question, positive_chunks) in enumerate(question_to_positives.items()):
        query_id = f"q{query_id_num}"
        queries[query_id] = question
        
        # For each positive chunk text, find its corresponding ID in the master corpus.
        relevant_doc_ids = {chunk_to_doc_id[p] for p in positive_chunks if p in chunk_to_doc_id}
        
        if len(relevant_doc_ids) != len(positive_chunks):
            logger.warning(f"For query '{question[:50]}...', some positive chunks were not found in the master corpus.")
            
        relevant_docs[query_id] = relevant_doc_ids
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluator = InformationRetrievalEvaluator(
        queries, 
        corpus, 
        relevant_docs, 
        batch_size=batch_size,
        show_progress_bar=(len(queries) > 10)
    )
    
    logger.info(f"Created evaluator with {len(queries)} queries and {len(corpus)} documents.")
    logger.info(f"Total relevant document mappings: {sum(len(docs) for docs in relevant_docs.values())}")
    
    return evaluator

def evaluate_on_test(model, 
                     test_triplets, 
                     corpus, 
                     output_dir,
                     batch_size : int = 4) -> dict:
    """
    Evaluate the model on test data using source chunk retrieval.
    
    Args:
        model: Trained SentenceTransformer model
        test_triplets: List of (question, positive_source, negative_source) triplets for testing
        corpus: Dictionary of source chunks for retrieval
        output_dir: Directory to save test evaluation results
        batch_size: Batch size for evaluation
        
    Returns:
        dict: Test evaluation results
    """
    logger.info(f"Evaluating model on {len(test_triplets)} test triplets")
    logger.info(f"Test corpus contains {len(corpus)} documents")
    
    if not test_triplets:
        logger.warning("No test triplets available for evaluation")
        return {}
    
    # The corpus should contain all chunks that appear in test triplets
    test_evaluator = create_evaluator(test_triplets, batch_size)
    
    # ADDITIONAL: Evaluate on FULL corpus (train + test) vs TEST-only corpus
    logger.info("Creating full corpus evaluator (train + test chunks)...")
    
    # Extract test questions and their positives for full corpus evaluation
    test_queries = {}
    test_relevant_docs = {}
    test_questions_to_positives = {}
    
    for question, positive_source, negative_source in test_triplets:
        if question not in test_questions_to_positives:
            test_questions_to_positives[question] = set()
        test_questions_to_positives[question].add(positive_source)
    
    # Build test queries
    for i, question in enumerate(test_questions_to_positives.keys()):
        query_id = f"test_q{i}"
        test_queries[query_id] = question
        test_relevant_docs[query_id] = set()
    
    # Use the FULL corpus from training data
    full_corpus = corpus.copy()  # This should contain all train+test chunks
    
    # Map positives to document IDs in full corpus
    doc_id_to_chunk = {doc_id: chunk for doc_id, chunk in full_corpus.items()}
    chunk_to_doc_id = {chunk: doc_id for doc_id, chunk in doc_id_to_chunk.items()}
    
    for i, (question, positives) in enumerate(test_questions_to_positives.items()):
        query_id = f"test_q{i}"
        for positive_chunk in positives:
            # Find this chunk in the full corpus
            for doc_id, chunk_text in full_corpus.items():
                if chunk_text == positive_chunk:
                    test_relevant_docs[query_id].add(doc_id)
                    break
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    full_corpus_evaluator = InformationRetrievalEvaluator(
        test_queries,
        full_corpus,
        test_relevant_docs,
        batch_size=batch_size,
        show_progress_bar=(len(test_queries) > 10)
    )
    
    logger.info(f"Full corpus evaluator: {len(test_queries)} queries, {len(full_corpus)} documents")
    
    # Run both evaluations
    logger.info("Running test evaluation on test-only corpus...")
    eval_start = time.time()
    test_stats = test_evaluator(model, output_path=os.path.join(output_dir, "test_results_test_corpus"))
    eval_end = time.time()
    eval_time = eval_end - eval_start
    test_stats['avg_time_sec'] = eval_time / len(test_triplets)
    test_stats['corpus_type'] = 'test_only'
    
    logger.info("Running test evaluation on full corpus (train+test)...")
    eval_start = time.time()
    full_stats = full_corpus_evaluator(model, output_path=os.path.join(output_dir, "test_results_full_corpus"))
    eval_end = time.time()
    eval_time = eval_end - eval_start
    full_stats['avg_time_sec'] = eval_time / len(test_queries)
    full_stats['corpus_type'] = 'train_plus_test'
    
    logger.info(f"Test evaluation (test corpus) completed")
    logger.info(f"Test evaluation (test corpus) results:")
    for metric, value in test_stats.items():
        logger.info(f"  {metric}: {value}")
    
    logger.info(f"Test evaluation (full corpus) completed")
    logger.info(f"Test evaluation (full corpus) results:")
    for metric, value in full_stats.items():
        logger.info(f"  {metric}: {value}")
    
    # Log some sample query-source rankings
    logger.info("\n===== SAMPLE TEST EVALUATIONS =====")
    num_samples = min(3, len(test_triplets))
    sample_triplets = random.sample(test_triplets, num_samples)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i, (question, positive_source, negative_source) in enumerate(sample_triplets):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"Question: {question[:150]}..." if len(question) > 150 else f"Question: {question}")
        logger.info(f"Gold positive source: {positive_source[:200]}..." if len(positive_source) > 200 else f"Gold positive source: {positive_source}")
        logger.info(f"Gold negative source: {negative_source[:200]}..." if len(negative_source) > 200 else f"Gold negative source: {negative_source}")
        
        # Create mini-corpus with positive, negative, and some random sources
        mini_corpus = {
            "positive": positive_source,
            "negative": negative_source
        }
        
        # Add some additional samples from corpus if available
        if corpus and len(corpus) > 2:
            corpus_items = list(corpus.items())
            random_samples = random.sample(corpus_items, min(5, len(corpus)))
            for idx, (doc_id, doc_text) in enumerate(random_samples):
                mini_corpus[f"sample_{idx}"] = doc_text
        
        # Compute embeddings and similarities
        question_embedding = model.encode(question, convert_to_tensor=True, device=device)
        corpus_texts = list(mini_corpus.values())
        corpus_ids = list(mini_corpus.keys())
        corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, device=device)
        
        scores = util.cos_sim(question_embedding, corpus_embeddings).cpu().numpy()[0]
        sorted_indices = np.argsort(-scores)
        
        logger.info("Ranked results:")
        for rank, idx in enumerate(sorted_indices):
            doc_id = corpus_ids[idx]
            score = scores[idx]
            text = corpus_texts[idx]
            special_marker = ""
            if doc_id == "positive":
                special_marker = "✓ [GOLD POSITIVE]"
            elif doc_id == "negative":
                special_marker = "✗ [GOLD NEGATIVE]"
            
            logger.info(f"  {rank+1}. ({score:.4f}) {special_marker} {text[:150]}..." if len(text) > 150 else f"  {rank+1}. ({score:.4f}) {special_marker} {text}")
    
    logger.info("=====================================\n")
    
    # Save both evaluation results
    test_eval_path = os.path.join(output_dir, "test_eval_results_test_corpus.json")
    with open(test_eval_path, 'w', encoding='utf-8') as f_out:
        json.dump(test_stats, f_out, ensure_ascii=False, indent=2)
    logger.info(f"Saved test corpus evaluation results to {test_eval_path}")
    
    full_eval_path = os.path.join(output_dir, "test_eval_results_full_corpus.json")
    with open(full_eval_path, 'w', encoding='utf-8') as f_out:
        json.dump(full_stats, f_out, ensure_ascii=False, indent=2)
    logger.info(f"Saved full corpus evaluation results to {full_eval_path}")
    
    # Return the test-only corpus results as primary
    return test_stats


def evaluate_on_test(model, 
                     test_triplets, 
                     corpus, 
                     output_dir,
                     batch_size : int = 4) -> dict:
    """
    Evaluate the model on test data using source chunk retrieval.
    
    Args:
        model: Trained SentenceTransformer model
        test_triplets: List of (question, positive_source, negative_source) triplets for testing
        corpus: Dictionary of source chunks for retrieval
        output_dir: Directory to save test evaluation results
        batch_size: Batch size for evaluation
        
    Returns:
        dict: Test evaluation results
    """
    logger.info(f"Evaluating model on {len(test_triplets)} test triplets")
    logger.info(f"Test corpus contains {len(corpus)} documents")
    
    if not test_triplets:
        logger.warning("No test triplets available for evaluation")
        return {}
    
    # The corpus should contain all chunks that appear in test triplets
    test_evaluator = create_evaluator(test_triplets, batch_size)
    
    # ADDITIONAL: Evaluate on FULL corpus (train + test) vs TEST-only corpus
    logger.info("Creating full corpus evaluator (train + test chunks)...")
    
    # Extract test questions and their positives for full corpus evaluation
    test_queries = {}
    test_relevant_docs = {}
    test_questions_to_positives = {}
    
    for question, positive_source, negative_source in test_triplets:
        if question not in test_questions_to_positives:
            test_questions_to_positives[question] = set()
        test_questions_to_positives[question].add(positive_source)
    
    # Build test queries
    for i, question in enumerate(test_questions_to_positives.keys()):
        query_id = f"test_q{i}"
        test_queries[query_id] = question
        test_relevant_docs[query_id] = set()
    
    # Use the FULL corpus from training data
    full_corpus = corpus.copy()  # This should contain all train+test chunks
    
    # Map positives to document IDs in full corpus
    doc_id_to_chunk = {doc_id: chunk for doc_id, chunk in full_corpus.items()}
    chunk_to_doc_id = {chunk: doc_id for doc_id, chunk in doc_id_to_chunk.items()}
    
    for i, (question, positives) in enumerate(test_questions_to_positives.items()):
        query_id = f"test_q{i}"
        for positive_chunk in positives:
            # Find this chunk in the full corpus
            for doc_id, chunk_text in full_corpus.items():
                if chunk_text == positive_chunk:
                    test_relevant_docs[query_id].add(doc_id)
                    break
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    full_corpus_evaluator = InformationRetrievalEvaluator(
        test_queries,
        full_corpus,
        test_relevant_docs,
        batch_size=batch_size,
        show_progress_bar=(len(test_queries) > 10)
    )
    
    logger.info(f"Full corpus evaluator: {len(test_queries)} queries, {len(full_corpus)} documents")
    
    # Run both evaluations
    logger.info("Running test evaluation on test-only corpus...")
    eval_start = time.time()
    test_stats = test_evaluator(model, output_path=os.path.join(output_dir, "test_results_test_corpus"))
    eval_end = time.time()
    eval_time = eval_end - eval_start
    test_stats['avg_time_sec'] = eval_time / len(test_triplets)
    test_stats['corpus_type'] = 'test_only'
    
    logger.info("Running test evaluation on full corpus (train+test)...")
    eval_start = time.time()
    full_stats = full_corpus_evaluator(model, output_path=os.path.join(output_dir, "test_results_full_corpus"))
    eval_end = time.time()
    eval_time = eval_end - eval_start
    full_stats['avg_time_sec'] = eval_time / len(test_queries)
    full_stats['corpus_type'] = 'train_plus_test'
    
    logger.info(f"Test evaluation (test corpus) completed")
    logger.info(f"Test evaluation (test corpus) results:")
    for metric, value in test_stats.items():
        logger.info(f"  {metric}: {value}")
    
    logger.info(f"Test evaluation (full corpus) completed")
    logger.info(f"Test evaluation (full corpus) results:")
    for metric, value in full_stats.items():
        logger.info(f"  {metric}: {value}")
    
    # Log some sample query-source rankings
    logger.info("\n===== SAMPLE TEST EVALUATIONS =====")
    num_samples = min(3, len(test_triplets))
    sample_triplets = random.sample(test_triplets, num_samples)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i, (question, positive_source, negative_source) in enumerate(sample_triplets):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"Question: {question[:150]}..." if len(question) > 150 else f"Question: {question}")
        logger.info(f"Gold positive source: {positive_source[:200]}..." if len(positive_source) > 200 else f"Gold positive source: {positive_source}")
        logger.info(f"Gold negative source: {negative_source[:200]}..." if len(negative_source) > 200 else f"Gold negative source: {negative_source}")
        
        # Create mini-corpus with positive, negative, and some random sources
        mini_corpus = {
            "positive": positive_source,
            "negative": negative_source
        }
        
        # Add some additional samples from corpus if available
        if corpus and len(corpus) > 2:
            corpus_items = list(corpus.items())
            random_samples = random.sample(corpus_items, min(5, len(corpus)))
            for idx, (doc_id, doc_text) in enumerate(random_samples):
                mini_corpus[f"sample_{idx}"] = doc_text
        
        # Compute embeddings and similarities
        question_embedding = model.encode(question, convert_to_tensor=True, device=device)
        corpus_texts = list(mini_corpus.values())
        corpus_ids = list(mini_corpus.keys())
        corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, device=device)
        
        scores = util.cos_sim(question_embedding, corpus_embeddings).cpu().numpy()[0]
        sorted_indices = np.argsort(-scores)
        
        logger.info("Ranked results:")
        for rank, idx in enumerate(sorted_indices):
            doc_id = corpus_ids[idx]
            score = scores[idx]
            text = corpus_texts[idx]
            special_marker = ""
            if doc_id == "positive":
                special_marker = "✓ [GOLD POSITIVE]"
            elif doc_id == "negative":
                special_marker = "✗ [GOLD NEGATIVE]"
            
            logger.info(f"  {rank+1}. ({score:.4f}) {special_marker} {text[:150]}..." if len(text) > 150 else f"  {rank+1}. ({score:.4f}) {special_marker} {text}")
    
    logger.info("=====================================\n")
    
    # Save both evaluation results
    test_eval_path = os.path.join(output_dir, "test_eval_results_test_corpus.json")
    with open(test_eval_path, 'w', encoding='utf-8') as f_out:
        json.dump(test_stats, f_out, ensure_ascii=False, indent=2)
    logger.info(f"Saved test corpus evaluation results to {test_eval_path}")
    
    full_eval_path = os.path.join(output_dir, "test_eval_results_full_corpus.json")
    with open(full_eval_path, 'w', encoding='utf-8') as f_out:
        json.dump(full_stats, f_out, ensure_ascii=False, indent=2)
    logger.info(f"Saved full corpus evaluation results to {full_eval_path}")
    
    # Return the test-only corpus results as primary
    return test_stats



def log_comprehensive_evaluation_to_wandb(results: Dict[str, dict], step: int = 0) -> None:
    """
    Log comprehensive evaluation results to W&B with appropriate prefixes.
    
    Args:
        results: Dictionary containing all evaluation results
        step: Current training step
    """
    if not wandb.run:
        logger.warning("No active W&B run found when trying to log comprehensive evaluation")
        return
    
    logger.info("Logging comprehensive evaluation metrics to W&B")
    
    all_metrics = {}
    
    for eval_name, stats in results.items():
        prefix = f"eval_{eval_name}"
        
        for key, value in stats.items():
            if isinstance(value, (int, float, np.number)):
                metric_name = f"{prefix}_{key}"
                all_metrics[metric_name] = float(value)
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float, np.number)):
                        metric_name = f"{prefix}_{key}_{subkey}"
                        all_metrics[metric_name] = float(subvalue)
                    elif isinstance(subvalue, dict):
                        for k, v in subvalue.items():
                            if isinstance(v, (int, float, np.number)):
                                metric_name = f"{prefix}_{key}_{subkey}_{k}"
                                all_metrics[metric_name] = float(v)
    
    wandb.log(all_metrics, step=step)
    logger.info(f"Logged {len(all_metrics)} comprehensive evaluation metrics to W&B")

def evaluate_on_test(model, 
                     test_triplets, 
                     corpus, 
                     output_dir,
                     batch_size : int = 4) -> dict:
    """
    Evaluate the model on test data using source chunk retrieval.
    
    Args:
        model: Trained SentenceTransformer model
        test_triplets: List of (question, positive_source, negative_source) triplets for testing
        corpus: Dictionary of source chunks for retrieval
        output_dir: Directory to save test evaluation results
        batch_size: Batch size for evaluation
        
    Returns:
        dict: Test evaluation results
    """
    logger.info(f"Evaluating model on {len(test_triplets)} test triplets")
    logger.info(f"Test corpus contains {len(corpus)} documents")
    
    if not test_triplets:
        logger.warning("No test triplets available for evaluation")
        return {}
    
    # The corpus should contain all chunks that appear in test triplets
    test_evaluator = create_evaluator(test_triplets, batch_size)
    
    # ADDITIONAL: Evaluate on FULL corpus (train + test) vs TEST-only corpus
    logger.info("Creating full corpus evaluator (train + test chunks)...")
    
    # Extract test questions and their positives for full corpus evaluation
    test_queries = {}
    test_relevant_docs = {}
    test_questions_to_positives = {}
    
    for question, positive_source, negative_source in test_triplets:
        if question not in test_questions_to_positives:
            test_questions_to_positives[question] = set()
        test_questions_to_positives[question].add(positive_source)
    
    # Build test queries
    for i, question in enumerate(test_questions_to_positives.keys()):
        query_id = f"test_q{i}"
        test_queries[query_id] = question
        test_relevant_docs[query_id] = set()
    
    # Use the FULL corpus from training data
    full_corpus = corpus.copy()  # This should contain all train+test chunks
    
    # Map positives to document IDs in full corpus
    doc_id_to_chunk = {doc_id: chunk for doc_id, chunk in full_corpus.items()}
    chunk_to_doc_id = {chunk: doc_id for doc_id, chunk in doc_id_to_chunk.items()}
    
    for i, (question, positives) in enumerate(test_questions_to_positives.items()):
        query_id = f"test_q{i}"
        for positive_chunk in positives:
            # Find this chunk in the full corpus
            for doc_id, chunk_text in full_corpus.items():
                if chunk_text == positive_chunk:
                    test_relevant_docs[query_id].add(doc_id)
                    break
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    full_corpus_evaluator = InformationRetrievalEvaluator(
        test_queries,
        full_corpus,
        test_relevant_docs,
        batch_size=batch_size,
        show_progress_bar=(len(test_queries) > 10)
    )
    
    logger.info(f"Full corpus evaluator: {len(test_queries)} queries, {len(full_corpus)} documents")
    
    # Run both evaluations
    logger.info("Running test evaluation on test-only corpus...")
    eval_start = time.time()
    test_stats = test_evaluator(model, output_path=os.path.join(output_dir, "test_results_test_corpus"))
    eval_end = time.time()
    eval_time = eval_end - eval_start
    test_stats['avg_time_sec'] = eval_time / len(test_triplets)
    test_stats['corpus_type'] = 'test_only'
    
    logger.info("Running test evaluation on full corpus (train+test)...")
    eval_start = time.time()
    full_stats = full_corpus_evaluator(model, output_path=os.path.join(output_dir, "test_results_full_corpus"))
    eval_end = time.time()
    eval_time = eval_end - eval_start
    full_stats['avg_time_sec'] = eval_time / len(test_queries)
    full_stats['corpus_type'] = 'train_plus_test'
    
    logger.info(f"Test evaluation (test corpus) completed")
    logger.info(f"Test evaluation (test corpus) results:")
    for metric, value in test_stats.items():
        logger.info(f"  {metric}: {value}")
    
    logger.info(f"Test evaluation (full corpus) completed")
    logger.info(f"Test evaluation (full corpus) results:")
    for metric, value in full_stats.items():
        logger.info(f"  {metric}: {value}")
    
    # Log some sample query-source rankings
    logger.info("\n===== SAMPLE TEST EVALUATIONS =====")
    num_samples = min(3, len(test_triplets))
    sample_triplets = random.sample(test_triplets, num_samples)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i, (question, positive_source, negative_source) in enumerate(sample_triplets):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"Question: {question[:150]}..." if len(question) > 150 else f"Question: {question}")
        logger.info(f"Gold positive source: {positive_source[:200]}..." if len(positive_source) > 200 else f"Gold positive source: {positive_source}")
        logger.info(f"Gold negative source: {negative_source[:200]}..." if len(negative_source) > 200 else f"Gold negative source: {negative_source}")
        
        # Create mini-corpus with positive, negative, and some random sources
        mini_corpus = {
            "positive": positive_source,
            "negative": negative_source
        }
        
        # Add some additional samples from corpus if available
        if corpus and len(corpus) > 2:
            corpus_items = list(corpus.items())
            random_samples = random.sample(corpus_items, min(5, len(corpus)))
            for idx, (doc_id, doc_text) in enumerate(random_samples):
                mini_corpus[f"sample_{idx}"] = doc_text
        
        # Compute embeddings and similarities
        question_embedding = model.encode(question, convert_to_tensor=True, device=device)
        corpus_texts = list(mini_corpus.values())
        corpus_ids = list(mini_corpus.keys())
        corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, device=device)
        
        scores = util.cos_sim(question_embedding, corpus_embeddings).cpu().numpy()[0]
        sorted_indices = np.argsort(-scores)
        
        logger.info("Ranked results:")
        for rank, idx in enumerate(sorted_indices):
            doc_id = corpus_ids[idx]
            score = scores[idx]
            text = corpus_texts[idx]
            special_marker = ""
            if doc_id == "positive":
                special_marker = "✓ [GOLD POSITIVE]"
            elif doc_id == "negative":
                special_marker = "✗ [GOLD NEGATIVE]"
            
            logger.info(f"  {rank+1}. ({score:.4f}) {special_marker} {text[:150]}..." if len(text) > 150 else f"  {rank+1}. ({score:.4f}) {special_marker} {text}")
    
    logger.info("=====================================\n")
    
    # Save both evaluation results
    test_eval_path = os.path.join(output_dir, "test_eval_results_test_corpus.json")
    with open(test_eval_path, 'w', encoding='utf-8') as f_out:
        json.dump(test_stats, f_out, ensure_ascii=False, indent=2)
    logger.info(f"Saved test corpus evaluation results to {test_eval_path}")
    
    full_eval_path = os.path.join(output_dir, "test_eval_results_full_corpus.json")
    with open(full_eval_path, 'w', encoding='utf-8') as f_out:
        json.dump(full_stats, f_out, ensure_ascii=False, indent=2)
    logger.info(f"Saved full corpus evaluation results to {full_eval_path}")
    
    # Return the test-only corpus results as primary
    return test_stats



class ComprehensiveEvaluator:
    """
    A custom evaluator that performs multiple evaluations during a single call:
    1. Train instances vs. Train corpus
    2. Train instances vs. Full corpus
    3. Test instances vs. Full corpus (This is the primary evaluation)

    The score from the primary evaluation is returned to the fit() method
    for checkpointing and saving the best model.
    """
    def __init__(self,
                 train_triplets: List[Tuple[str, str, str]],
                 test_triplets: List[Tuple[str, str, str]],
                 full_corpus: Dict[str, str],
                 batch_size: int = 16,
                 show_progress_bar: bool = False):

        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.show_progress_bar = show_progress_bar

        logger.info("Initializing ComprehensiveEvaluator...")

        # --- Create all necessary sub-evaluators by calling the main create_evaluator function ---

        # 1. Evaluator for train instances on its own corpus
        logger.info("...creating train_on_train evaluator")
        # First, build the train-only corpus from the unique documents in the train triplets.
        train_corpus_chunks = {chunk for triplet in train_triplets for chunk in triplet[1:]}
        train_corpus = {f"doc_{i}": chunk for i, chunk in enumerate(train_corpus_chunks)}
        
        self.train_on_train_evaluator = create_evaluator(
            train_triplets, 
            train_corpus,  # Pass the correct train-only corpus
            batch_size=self.batch_size
        )

        # 2. Evaluator for train instances on the full corpus
        logger.info("...creating train_on_full evaluator")
        self.train_on_full_evaluator = create_evaluator(
            train_triplets, 
            full_corpus,  # Pass the full corpus
            batch_size=self.batch_size
        )
        self.train_queries_count = len(self.train_on_full_evaluator.queries)

        # 3. Primary Evaluator for test instances on the full corpus
        logger.info("...creating primary (test_on_full) evaluator")
        self.test_on_full_evaluator = create_evaluator(
            test_triplets, 
            full_corpus,  # Pass the full corpus
            batch_size=self.batch_size
        )
        self.test_queries_count = len(self.test_on_full_evaluator.queries)

        # Store results for summary reporting
        self.last_results = {}
        
        # This key was corrected in the previous step.
        self.main_score_key = "cosine_ndcg@10"
    
    def __call__(self, model, output_path=None, epoch=-1, steps=-1, log_to_wandb=True):
        """Run comprehensive evaluation on all three scenarios"""
        start_time = time.time()
        
        # Initialize results storage
        all_results = {}
        
        logger.info("Running comprehensive evaluation...")
        
        # 1. Train on Train Corpus evaluation
        logger.info("Evaluating: Train queries on Train corpus...")
        train_on_train_results = self.train_on_train_evaluator(model, output_path=output_path, epoch=epoch, steps=steps)
        all_results['train_on_train'] = train_on_train_results
        
        # 2. Train on Full Corpus evaluation  
        logger.info("Evaluating: Train queries on Full corpus...")
        train_on_full_results = self.train_on_full_evaluator(model, output_path=output_path, epoch=epoch, steps=steps)
        all_results['train_on_full'] = train_on_full_results
        
        # 3. Test on Full Corpus evaluation (primary)
        logger.info("Evaluating: Test queries on Full corpus...")
        test_on_full_results = self.test_on_full_evaluator(model, output_path=output_path, epoch=epoch, steps=steps)
        all_results['test_on_full'] = test_on_full_results
        
        # Store results for summary
        self.last_results = all_results
        
        # Log metrics to W&B if available and requested
        if log_to_wandb and wandb.run:
            logger.info("Logging comprehensive evaluation metrics to W&B")
            metrics_logged = 0
            for scenario, results in all_results.items():
                if results:
                    for metric, value in results.items():
                        if isinstance(value, (int, float)):
                            metric_name = f"eval_{scenario}_{metric}"
                            # Use step=None for pre-training evaluation (steps=-1)
                            wandb_step = steps if steps >= 0 else None
                            wandb.log({metric_name: value}, step=wandb_step)
                            metrics_logged += 1
            logger.info(f"Logged {metrics_logged} comprehensive evaluation metrics to W&B")
        
        # Save detailed results to file if output path provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            results_file = os.path.join(output_path, f"comprehensive_eval_epoch_{epoch}_step_{steps}.json")
            
            # Convert any numpy types to Python native types for JSON serialization
            serializable_results = {}
            for scenario, results in all_results.items():
                serializable_results[scenario] = {}
                for metric, value in results.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        serializable_results[scenario][metric] = value.item()
                    else:
                        serializable_results[scenario][metric] = value
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        
        # Define a list of possible keys to check for the nDCG score.
        # This makes the evaluator robust to different output formats.
        possible_keys = ['cosine_ndcg@10', 'ndcg@10', 'ndcg_at_10', 'dot_ndcg@10']

        def get_score_from_results(results_dict, keys):
            """Helper function to find the score using a list of possible keys."""
            if not results_dict:
                return "N/A"
            for key in keys:
                if key in results_dict:
                    # Return the value formatted as a string
                    return f"{results_dict[key]:.4f}"
            return "N/A"

        # Print summary using the robust helper function
        logger.info("Comprehensive evaluation summary (nDCG@10):")
        train_on_train_ndcg = get_score_from_results(all_results.get('train_on_train'), possible_keys)
        train_on_full_ndcg = get_score_from_results(all_results.get('train_on_full'), possible_keys)
        test_on_full_ndcg = get_score_from_results(all_results.get('test_on_full'), possible_keys)
        
        logger.info(f"  - Train on Train Corpus: {train_on_train_ndcg}")
        logger.info(f"  - Train on Full Corpus:  {train_on_full_ndcg}")
        logger.info(f"  - Test on Full Corpus:   {test_on_full_ndcg}")
        
        if output_path:
            logger.info(f"Saved comprehensive evaluation results to {results_file}")
        
        # Find the primary score for SBERT's internal logic (for saving the best model)
        primary_results = all_results.get('test_on_full', {})
        main_score = None
        for key in possible_keys:
            if key in primary_results:
                main_score = primary_results[key]
                break
        
        if main_score is None:
            logger.error(f"Main score key not found in primary evaluation results!")
            logger.error(f"Available keys: {list(primary_results.keys())}")
            main_score = 0.0 # Default to 0.0 if no key is found
        
        
        duration = time.time() - start_time
        logger.info(f"Comprehensive evaluation completed in {duration:.2f} seconds")
        
        return main_score

    # Add these methods to make it compatible with SentenceTransformer's evaluation system
    def __iter__(self):
        """Make the evaluator iterable - return itself as single evaluator"""
        return iter([self])
    
    def __len__(self):
        """Return length as 1 since this is a single evaluator"""
        return 1
    
    def __getitem__(self, index):
        """Support indexing - return self for index 0"""
        if index == 0:
            return self
        else:
            raise IndexError("ComprehensiveEvaluator only has one evaluator")


def main(args):
    """
    Main function to train a retrieval model.
    
    Args:
        args: Command-line arguments
    """
    script_start_time = time.time()
    logger.info(f"Starting Tourism QA retrieval model training with args: {args}")
    
    # Log system info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        # Model parameters
        model_name = args.model_name
        train_batch_size = args.train_batch_size
        max_seq_length = args.max_seq_length
        num_epochs = args.epochs
        model_save_path = args.output_dir
        
        logger.info(f"Model configuration:")
        logger.info(f"  Model name: {model_name}")
        logger.info(f"  Batch size: {train_batch_size}")
        logger.info(f"  Max sequence length: {max_seq_length}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Output directory: {model_save_path}")
        logger.info(f"  Negative mining strategy: {args.negative_mining_strategy}")
        
        # Ensure output directory exists
        os.makedirs(model_save_path, exist_ok=True)
        logger.info(f"Created output directory: {model_save_path}")
        
        # Save configuration
        config_path = os.path.join(model_save_path, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f"Saved configuration to {config_path}")

        # Initialize W&B early
        wandb_run = None
        if not args.disable_wandb:
            project_name = "Tourism-QA-Retriever"
            run_name_prefix = "EvalOnly" if args.do_eval_only else "Train"
            model_name_suffix = model_name.split('/')[-1] if isinstance(model_name, str) else "unknown_model"
            run_name = f"{run_name_prefix}-{model_name_suffix}-{datetime.now().strftime('%Y%m%d-%H%M')}"
            
            logger.info(f"Initializing W&B with project: {project_name}, run name: {run_name}")
            wandb_run = wandb.init(project=project_name, name=run_name, config=vars(args))
        
        # Load model
        logger.info(f"Loading model {model_name}...")
        model_load_start = time.time()
        
        model = None 
        try:
            if args.use_pre_trained_model:
                logger.info("Using pretrained SentenceTransformer model")
                model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda")
            else:
                logger.info("Creating new SentenceTransformer model")
                word_embedding_model = models.Transformer(model_name, 
                                                          max_seq_length=max_seq_length, 
                                                          tokenizer_args={"trust_remote_code":True}, 
                                                          model_args={"trust_remote_code":True}, 
                                                          config_args={"trust_remote_code":True})
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
                model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to('cuda')
            
            logger.info(f"Model loaded/created successfully in {time.time() - model_load_start:.2f} seconds")
            if model:
                num_params = sum(p.numel() for p in model.parameters())
                logger.info(f"Model has {num_params:,} parameters")
            else:
                raise RuntimeError("Model initialization failed.")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            raise

        # Load dataset and prepare triplets
        logger.info("Loading data and preparing triplets...")
        train_triplets, source_chunk_corpus, test_triplets = load_tourism_qa_data(
            args.dataset_name,
            max_train_samples=args.max_train_samples,
            num_negatives=args.num_negatives,
            use_hard_negatives=args.use_hard_negatives,
            hard_negative_ratio=args.hard_negative_ratio,
            hard_negative_model=model,
            negative_mining_strategy=args.negative_mining_strategy
        )
        
        logger.info(f"Train triplets: {len(train_triplets)}")
        logger.info(f"Test triplets: {len(test_triplets)}")
        logger.info(f"Source chunk corpus: {len(source_chunk_corpus)} unique chunks")
        
        # Create dataset and dataloader
        logger.info(f"Creating dataset and dataloader [{len(train_triplets)} triplets]")
        train_dataset = TripletDataset(train_triplets)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        logger.info(f"Created dataloader with {len(train_dataloader)} batches")
        
        # Create loss function
        logger.info("Creating MultipleNegativesRankingLoss")
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
        
        # Create evaluator using the new class
        logger.info("Creating evaluation setup with ComprehensiveEvaluator")
        ir_evaluator = ComprehensiveEvaluator(
            train_triplets=train_triplets,
            test_triplets=test_triplets,
            full_corpus=source_chunk_corpus,
            batch_size=args.eval_batch_size,
            show_progress_bar=False  # Recommended to keep logs clean during training
        )
        
        # Pre-training comprehensive evaluation (before training starts)
        logger.info("Running pre-training comprehensive evaluation on all 3 scenarios...")
        pre_training_eval_start_time = time.time()
        pre_training_primary_score = ir_evaluator(
            model,
            output_path=os.path.join(model_save_path, "pre_training_eval_reports"),
            epoch=-1,  # Use -1 to indicate pre-training evaluation
            steps=-1,
            log_to_wandb=(not args.disable_wandb and wandb.run is not None)  # Log to W&B if enabled
        )
        logger.info(f"Pre-training comprehensive evaluation completed in {time.time() - pre_training_eval_start_time:.2f} seconds.")
        logger.info(f"Pre-training primary score (test_on_full nDCG@10): {pre_training_primary_score:.4f}")

        total_training_steps = 0
        if not args.do_eval_only:
            logger.info("Starting training...")
            train_start = time.time()
            
            total_training_steps = len(train_dataloader) * num_epochs
            warmup_steps_absolute = int(args.warmup_steps_ratio * total_training_steps)
            logger.info(f"Total training steps: {total_training_steps}")
            logger.info(f"Warmup steps ratio: {args.warmup_steps_ratio}, Absolute warmup steps: {warmup_steps_absolute}")

            evaluation_steps_absolute = 0
            if args.evaluation_interval_ratio > 0 and total_training_steps > 0:
                evaluation_steps_absolute = int(args.evaluation_interval_ratio * total_training_steps)
                if evaluation_steps_absolute == 0: 
                    evaluation_steps_absolute = 1 
            elif len(train_dataloader) > 0:
                evaluation_steps_absolute = len(train_dataloader)

            logger.info(f"Evaluation interval ratio: {args.evaluation_interval_ratio}, Absolute evaluation steps: {evaluation_steps_absolute}")
            
            def sbert_fit_callback(score, epoch, steps):
                if wandb.run: 
                    # The comprehensive evaluator logs detailed metrics.
                    # This logs the primary score that SBERT uses for its internal checkpointing logic.
                    wandb.log({"sbert_primary_eval_metric": score, "epoch": epoch, "steps": steps}, step=steps)
            
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                warmup_steps=warmup_steps_absolute,
                evaluator=ir_evaluator,
                evaluation_steps=evaluation_steps_absolute if evaluation_steps_absolute > 0 else len(train_dataloader),
                callback=sbert_fit_callback if wandb.run else None,
                use_amp=True,
                checkpoint_path=model_save_path, 
                checkpoint_save_total_limit=args.checkpoint_save_total_limit,
                checkpoint_save_steps=evaluation_steps_absolute if evaluation_steps_absolute > 0 else len(train_dataloader),
                optimizer_params={"lr": args.lr},
                scheduler=args.scheduler,
                save_best_model=True,
                max_grad_norm=args.max_grad_norm,
                weight_decay=args.weight_decay
            )
            
            train_end = time.time()
            train_duration = train_end - train_start
            logger.info(f"Training completed in {train_duration:.2f} seconds ({str(timedelta(seconds=int(train_duration)))})")

            
            final_model_path = os.path.join(model_save_path, "final_epoch_model")
            logger.info(f"Saving final epoch model to {final_model_path}")
            model.save(final_model_path)
            
            if args.push_to_hub and os.environ.get('HF_TOKEN'):
                repo_name = args.hub_repo_name or f"tourism-qa-retriever-{model_name.split('/')[-1]}"
                logger.info(f"Pushing model to Hugging Face Hub: {repo_name}")
                model.push_to_hub(repo_name, private=not args.public_model)
        
        # Final evaluation
        final_eval_model_to_use = model
        if not args.do_eval_only:
            best_model_path = model_save_path
            logger.info(f"Loading best model from {best_model_path} for final evaluation")
            try:
                final_eval_model_to_use = SentenceTransformer(model_save_path, device="cuda" if torch.cuda.is_available() else "cpu")
            except Exception as e:
                logger.warning(f"Could not load best model from {best_model_path}. Error: {e}. Using current model.")
        elif args.do_eval_only:
             logger.info(f"Using model from {args.model_name} for final evaluation.")
             final_eval_model_to_use = SentenceTransformer(args.model_name, device="cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Running final comprehensive evaluation...")
        final_eval_start_time = time.time()
        final_epoch_step = num_epochs if not args.do_eval_only else 0
        final_global_step = total_training_steps if not args.do_eval_only else 0
        
        final_primary_stats = ir_evaluator(
            final_eval_model_to_use, 
            epoch=final_epoch_step, 
            steps=final_global_step, 
            output_path=os.path.join(model_save_path, "final_eval_reports"),
            log_to_wandb=(not args.disable_wandb and wandb.run is not None)  # Log to W&B if enabled
        )
        logger.info(f"Final comprehensive evaluation completed in {time.time() - final_eval_start_time:.2f} seconds.")
        logger.info(f"Final primary stats (test_on_full nDCG@10): {final_primary_stats:.4f}")

        if wandb.run:
            logger.info("Finishing W&B run")
            wandb.finish()

        script_end_time = time.time()
        total_time = script_end_time - script_start_time
        logger.info(f"Script execution completed in {total_time:.2f} seconds ({str(timedelta(seconds=int(total_time)))})")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        logger.error(traceback.format_exc())
        raise

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a tourism QA retrieval model")
    
    # Data arguments
    parser.add_argument("--dataset_name", default="LGenie/turismo_mcQa_formatted", type=str,
                       help="HuggingFace dataset name for tourism QA data")
    parser.add_argument("--max_train_samples", default=None, type=int,
                       help="Maximum number of training samples to use (if specified, randomly selects a subset of training data)")
    parser.add_argument("--num_negatives", default=5, type=int,
                       help="Number of negative examples to use per query-positive pair")
    parser.add_argument("--use_hard_negatives", action="store_true", default=True,
                       help="Use hard negative mining based on semantic similarity")
    parser.add_argument("--hard_negative_ratio", default=0.7, type=float,
                       help="Ratio of hard negatives vs random negatives (0.0-1.0)")
    parser.add_argument("--negative_mining_strategy", default="hard", type=str, 
                       choices=["hard", "random"],
                       help="Strategy for negative mining. 'hard' uses semantic similarity for hard negatives, "
                            "'random' uses random sampling. Both strategies avoid chunks from same source and "
                            "handle chunks with multiple questions properly.")
    
    # Model arguments
    parser.add_argument("--model_name", required=True, type=str,
                       help="Transformer model name or path")
    parser.add_argument("--train_batch_size", default=32, type=int,
                       help="Training batch size")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                       help="Evaluation batch size")
    parser.add_argument("--max_seq_length", default=512, type=int,
                       help="Maximum sequence length")
    parser.add_argument("--epochs", default=5, type=int,
                       help="Number of training epochs")
    parser.add_argument("--pooling", default="mean", type=str, choices=["mean", "cls", "max"],
                       help="Pooling strategy (mean, cls, max)")
    parser.add_argument("--use_pre_trained_model", action="store_true",
                       help="Use pretrained SentenceTransformer instead of building from base model")
    
    # Training arguments
    parser.add_argument("--warmup_steps_ratio", default=0.1, type=float,
                       help="Ratio of total training steps to use for warmup (e.g., 0.1 for 10%)")
    parser.add_argument("--lr", default=2e-5, type=float,
                       help="Learning rate")
    parser.add_argument("--evaluation_interval_ratio", default=0.05, type=float,
                       help="Fraction of total training steps defining the evaluation interval. E.g., 0.1 means evaluate every 10% of total steps. Set to 0 for no evaluation during training.")
    parser.add_argument("--do_eval_only", action="store_true",
                       help="Only run evaluation, no training")
    parser.add_argument("--output_dir", default="output/tourism_retriever",
                       help="Directory to save model checkpoints")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                       help="Weight decay for AdamW optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                       help="Max gradient norm for gradient clipping")
    parser.add_argument("--scheduler", default="warmupcosine", type=str, choices=["warmupcosine", "warmuplinear", "constantlr"],
                       help="Learning rate scheduler (warmupcosine, warmuplinear, constantlr)")
    parser.add_argument("--checkpoint_save_total_limit", default=2, type=int,
                       help="Maximum number of checkpoints to keep (use -1 for no limit)")
    
    # Misc arguments
    parser.add_argument("--disable_wandb", action="store_true", default=False,
                       help="Disable Weights & Biases logging")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push trained model to Hugging Face Hub")
    parser.add_argument("--hub_repo_name", type=str,
                       help="Repository name for Hugging Face Hub")
    parser.add_argument("--public_model", action="store_true", default=False,
                       help="Make the pushed model publicly available")
    parser.add_argument("--log_file", type=str,
                       help="Path to log file (if not set, only console logging is used)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Set log file from argument if provided
    if args.log_file:
        os.environ["LOG_FILE"] = args.log_file

    main(args)