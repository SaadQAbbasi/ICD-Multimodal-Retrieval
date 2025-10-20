from dotenv import load_dotenv
load_dotenv()

import random
import wandb
import io
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
import glob
import argparse
from huggingface_hub import login
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import (
    AutoTokenizer,
    AutoProcessor
)
import numpy as np
from collections import defaultdict, deque
from torch.utils.data import Dataset, DataLoader

# Import Unsloth for efficient finetuning (required)
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
image_path = "/datasets/MIMIC-CXR/files"
input_tokenizer = None
output_tokenizer = None
processor = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# TREE BUILDER FOR CONSTRAINED DECODING
# ============================================================================
class TreeBuilder:
    """Build a tree structure for hierarchical document IDs."""
    def __init__(self) -> None:
        self.tree = {}
        self.count = 0

    def add(self, path: List[int]) -> None:
        """Add a path to the tree."""
        current = self.tree
        for node in path:
            if node not in current:
                current[node] = {}
            current = current[node]
        self.count += 1

    def get_valid_next_tokens(self, prefix: List[int]) -> List[int]:
        """Get valid next tokens given a prefix path."""
        current = self.tree
        for node in prefix:
            if node not in current:
                return []
            current = current[node]
        return list(current.keys())

    def __str__(self) -> str:
        return json.dumps(self.tree, indent=6)


# ============================================================================
# MODEL LOADING UTILITIES
# ============================================================================
def detect_unsloth_model_name(model_name: str) -> str:
    """
    Detect or convert to Unsloth model name.

    Args:
        model_name: HuggingFace model name or Unsloth model name

    Returns:
        Unsloth model name

    Raises:
        ValueError: If model is not supported by Unsloth
    """
    # If already an Unsloth model name, return it
    if model_name.startswith("unsloth/"):
        return model_name

    # Map common models to their Unsloth versions
    if "Qwen2-VL" in model_name or "qwen2-vl" in model_name.lower():
        if "2B" in model_name or "2b" in model_name.lower():
            return "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"
        elif "7B" in model_name or "7b" in model_name.lower():
            return "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit"

    # If no mapping found, raise error
    raise ValueError(
        f"Model '{model_name}' is not supported by Unsloth. "
        f"Supported models: Qwen2-VL-2B-Instruct, Qwen2-VL-7B-Instruct, "
        f"or provide an Unsloth model name directly (e.g., 'unsloth/...')"
    )


def load_model_and_processor(model_name: str, use_4bit: bool = True):
    """
    Load vision-language model and processor using Unsloth.

    Args:
        model_name: HuggingFace or Unsloth model name
        use_4bit: Whether to use 4-bit quantization

    Returns:
        model, tokenizer, processor
    """
    # Convert to Unsloth model name
    unsloth_model_name = detect_unsloth_model_name(model_name)

    print(f"Loading model with Unsloth: {unsloth_model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        unsloth_model_name,
        load_in_4bit=use_4bit,
        use_gradient_checkpointing="unsloth",
    )

    # Load processor from the original model name if not already unsloth path
    if model_name.startswith("unsloth/"):
        # Extract base model name from unsloth path
        processor_name = model_name.replace("unsloth/", "").replace("-bnb-4bit", "")
        # Try to map to official HF model
        if "Qwen2-VL-2B" in processor_name:
            processor_name = "Qwen/Qwen2-VL-2B-Instruct"
        elif "Qwen2-VL-7B" in processor_name:
            processor_name = "Qwen/Qwen2-VL-7B-Instruct"
    else:
        processor_name = model_name

    processor = AutoProcessor.from_pretrained(processor_name)

    return model, tokenizer, processor


def add_lora_to_model(model, lora_r=16, lora_alpha=16, lora_dropout=0.0):
    """
    Add LoRA adapters to model using Unsloth.

    Args:
        model: Base model from FastVisionModel
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate

    Returns:
        Model with LoRA adapters
    """
    print("Adding LoRA adapters with Unsloth")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        random_state=3407,
        use_rslora=False,
    )
    return model


def prepare_model_inputs(inputs):
    """
    Extract relevant inputs for the model.

    Args:
        inputs: Dict of inputs from processor

    Returns:
        Dict with model inputs (input_ids, pixel_values, attention_mask, etc.)
    """
    model_inputs = {}

    # Add all tensor inputs from the processor
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value

    return model_inputs


# ============================================================================
# CONSTRAINED BEAM SEARCH GENERATION
# ============================================================================
@torch.no_grad()
def constrained_beam_search_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    min_new_tokens: int = 2,
    max_new_tokens: int = 6,
    decoding_tree: Optional[TreeBuilder] = None,
    num_beams: int = 10,
    batch_size: int = 4,
    num_return_sequences: int = 10,
    vocab_size: int = 124,
    input_tokenizer: Any = None,
    output_tokenizer: Any = None,
    temperature: float = 1.0,
    model_inputs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Constrained beam search generation to ensure valid document IDs without duplicates.

    Args:
        model: Vision-language model
        input_ids: Input token IDs
        min_new_tokens: Minimum tokens to generate
        max_new_tokens: Maximum tokens to generate
        decoding_tree: Tree structure for constrained decoding
        num_beams: Number of beams
        batch_size: Batch size
        num_return_sequences: Number of sequences to return
        vocab_size: Vocabulary size
        input_tokenizer: Input tokenizer
        output_tokenizer: Output tokenizer
        temperature: Sampling temperature
        model_inputs: Dict containing all model inputs (input_ids, pixel_values, etc.)

    Returns:
        Generated sequences
    """
    device = input_ids.device
    eos_token_id = output_tokenizer.eos_token_id

    batch_size = input_ids.shape[0]

    # Expand input_ids for beam search
    input_ids_expanded = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1)
    input_ids_expanded = input_ids_expanded.contiguous().view(batch_size * num_beams, -1)

    # Expand all model inputs for beam search
    expanded_inputs = {"input_ids": input_ids_expanded}

    if model_inputs is not None:
        for key, value in model_inputs.items():
            if key == "input_ids":
                continue
            if value is not None and isinstance(value, torch.Tensor):
                if value.dim() >= 2:
                    # Expand along beam dimension
                    expanded = value.unsqueeze(1).expand(batch_size, num_beams, *value.shape[1:])
                    expanded_inputs[key] = expanded.contiguous().view(batch_size * num_beams, *value.shape[1:])
                else:
                    expanded_inputs[key] = value

    # Initialize generated sequences and scores
    generated_ids = torch.zeros((batch_size * num_beams, 0), dtype=torch.long, device=device)
    beam_scores = torch.zeros((batch_size, num_beams), device=device)
    done = torch.zeros((batch_size, num_beams), dtype=torch.bool, device=device)

    vocab_size_actual = vocab_size

    for step in range(max_new_tokens):
        # Forward pass through the model
        # Update input_ids in expanded_inputs
        expanded_inputs["input_ids"] = input_ids_expanded
        outputs = model(**expanded_inputs)

        next_token_logits = outputs.logits[:, -1, :]
        vocab_size_actual = next_token_logits.shape[-1]

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply constraints from the decoding tree if provided
        if decoding_tree is not None:
            invalid_tokens_mask = torch.ones_like(next_token_logits, dtype=torch.bool)

            for beam_idx in range(batch_size * num_beams):
                current_path = generated_ids[beam_idx].tolist()

                try:
                    valid_tokens = decoding_tree.get_valid_next_tokens(current_path)
                    if valid_tokens:
                        valid_tensor = torch.tensor(valid_tokens, device=device)
                        valid_mask = (valid_tensor >= 0) & (valid_tensor < vocab_size_actual)
                        valid_tokens_filtered = valid_tensor[valid_mask]
                        if len(valid_tokens_filtered) > 0:
                            invalid_tokens_mask[beam_idx, valid_tokens_filtered] = False
                except Exception:
                    pass

                # Handle EOS token
                if step + 1 >= min_new_tokens and 0 <= eos_token_id < vocab_size_actual:
                    invalid_tokens_mask[beam_idx, eos_token_id] = False

                # Safety fallback
                if invalid_tokens_mask[beam_idx].all():
                    if 0 <= eos_token_id < vocab_size_actual:
                        invalid_tokens_mask[beam_idx, eos_token_id] = False
                    else:
                        invalid_tokens_mask[beam_idx, 0] = False

            next_token_logits = next_token_logits.masked_fill(invalid_tokens_mask, -float('inf'))

        # Convert logits to probabilities
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)
        next_token_scores = next_token_scores.view(batch_size, num_beams, -1)
        next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)
        next_scores = next_token_scores.view(batch_size, -1)

        # Optimized duplicate removal per batch
        new_beam_scores = torch.zeros((batch_size, num_beams), device=device)
        new_beam_indices = torch.zeros((batch_size, num_beams), dtype=torch.long, device=device)
        new_token_indices = torch.zeros((batch_size, num_beams), dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            batch_scores = next_scores[batch_idx]
            num_candidates = min(num_beams * 3, len(batch_scores))
            top_scores, top_indices = torch.topk(batch_scores, num_candidates)

            top_beam_indices = top_indices // vocab_size_actual
            top_token_indices = top_indices % vocab_size_actual

            seen_sequences = set()
            selected_count = 0

            for i in range(num_candidates):
                if selected_count >= num_beams:
                    break

                beam_idx = top_beam_indices[i].item()
                token_idx = top_token_indices[i].item()
                score = top_scores[i].item()

                if torch.isinf(top_scores[i]) and top_scores[i] < 0:
                    continue

                current_seq = generated_ids[batch_idx * num_beams + beam_idx]
                if len(current_seq) == 0:
                    seq_hash = token_idx
                else:
                    seq_hash = hash(tuple(current_seq.tolist() + [token_idx]))

                if seq_hash not in seen_sequences:
                    seen_sequences.add(seq_hash)
                    new_beam_scores[batch_idx, selected_count] = score
                    new_beam_indices[batch_idx, selected_count] = beam_idx
                    new_token_indices[batch_idx, selected_count] = token_idx
                    selected_count += 1

            # Fill remaining slots if needed
            while selected_count < num_beams:
                if selected_count > 0:
                    new_beam_scores[batch_idx, selected_count] = new_beam_scores[batch_idx, 0] - 1e-6
                    new_beam_indices[batch_idx, selected_count] = new_beam_indices[batch_idx, 0]
                    new_token_indices[batch_idx, selected_count] = new_token_indices[batch_idx, 0]
                else:
                    new_beam_scores[batch_idx, selected_count] = -1e6
                    new_beam_indices[batch_idx, selected_count] = 0
                    new_token_indices[batch_idx, selected_count] = eos_token_id if eos_token_id < vocab_size_actual else 0
                selected_count += 1

        beam_scores = new_beam_scores
        beam_indices = new_beam_indices
        token_indices = new_token_indices

        # Update sequences
        beam_indices_expanded = beam_indices.unsqueeze(-1).expand(-1, -1, input_ids_expanded.shape[-1])
        input_ids_gathered = input_ids_expanded.view(batch_size, num_beams, -1)
        input_ids_new = torch.gather(input_ids_gathered, 1, beam_indices_expanded)
        input_ids_expanded = input_ids_new.view(batch_size * num_beams, -1)

        # Append new tokens
        token_indices_expanded = token_indices.view(batch_size * num_beams, 1)
        input_ids_expanded = torch.cat([input_ids_expanded, token_indices_expanded], dim=1)

        if generated_ids.shape[-1] > 0:
            gen_beam_indices = beam_indices.unsqueeze(-1).expand(-1, -1, generated_ids.shape[-1])
            generated_gathered = generated_ids.view(batch_size, num_beams, -1)
            generated_ids_new = torch.gather(generated_gathered, 1, gen_beam_indices)
        else:
            generated_ids_new = torch.zeros((batch_size, num_beams, 0), dtype=torch.long, device=device)

        token_indices_for_gen = token_indices.unsqueeze(-1)
        generated_ids_new = torch.cat([generated_ids_new, token_indices_for_gen], dim=-1)
        generated_ids = generated_ids_new.view(batch_size * num_beams, -1)

        # Check for completion
        is_eos = (token_indices == eos_token_id)
        if step + 1 >= min_new_tokens:
            done = done | is_eos

        if done.all():
            break

    output = generated_ids.view(batch_size, num_beams, -1)
    output = output[:, :num_return_sequences, :]

    return output


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================
def load_first_image_from_path(path_str, base_dir=image_path):
    """Load the first available image from the folder"""
    try:
        modelFolder, folder, subfolder = path_str.split(".")
        image_dir = os.path.join(base_dir, modelFolder, folder, subfolder)
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in img_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))

        if not image_files:
            return None
        return Image.open(image_files[0])
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def load_three_images_from_path(path_str, base_dir=image_path):
    """Load the first three images from folder. If fewer than 3, repeat some."""
    try:
        modelFolder, folder, subfolder = path_str.split(".")
        image_dir = os.path.join(base_dir, modelFolder, folder, subfolder)

        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in img_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))

        if not image_files:
            return [None, None, None]

        image_files.sort()
        images = [Image.open(image_files[i % len(image_files)]) for i in range(3)]
        return images

    except Exception as e:
        print(f"Error loading images: {e}")
        return [None, None, None]


def filter_valid_entries(df):
    """Remove rows where the image is not accessible"""
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking valid files"):
        path_str = row['report_id']
        try:
            image = load_first_image_from_path(path_str)
        except Exception:
            continue

        if image is not None and isinstance(image, Image.Image):
            valid_indices.append(idx)

    df_valid = df.loc[valid_indices].reset_index(drop=True)
    return df_valid


def into_positional_docid(df: pd.DataFrame) -> None:
    """Convert document IDs in dataframe to positional format."""
    offsets = [0, 19, 207, 1403, 7902]

    docids = df["docid"].copy()
    for i, docid in enumerate(docids):
        digits = docid.split('-')
        seq = []
        for j, dig in enumerate(digits):
            dig = int(dig) + offsets[j]
            seq.append(str(dig))
        seq = '-'.join(seq)
        docids[i] = seq

    df["docid"] = docids


# ============================================================================
# TOKENIZER CREATION WITH AUGMENTATION
# ============================================================================
def augment_tokenizer(tokenizer, is_output_tokenizer=False):
    """
    Augment tokenizer with special tokens for ICD codes.

    Args:
        tokenizer: Base tokenizer to augment
        is_output_tokenizer: If True, add tokens for output (1-11881), else for input (10-11881)
    """
    # Set pad and sep tokens
    if not is_output_tokenizer:
        tokenizer.pad_token = '|'
        if '|' not in tokenizer.get_vocab():
            tokenizer.add_tokens(['|'], special_tokens=True)
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('|')

        tokenizer.sep_token = '~'
        if '~' not in tokenizer.get_vocab():
            tokenizer.add_tokens(['~'], special_tokens=True)
        tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids('~')

        # Add numeric tokens from 10 to 11881 for document IDs
        start_num = 10
    else:
        tokenizer.pad_token = '<|>'
        if '<|>' not in tokenizer.get_vocab():
            tokenizer.add_tokens(['<|>'], special_tokens=True)
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|>')

        # Add numeric tokens from 1 to 11881 for positional document IDs
        start_num = 1

    # Add numeric tokens
    tokens_to_add = []
    for num in range(start_num, 11882):
        token = str(num)
        if token not in tokenizer.get_vocab():
            tokens_to_add.append(token)

    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)

    # Add hyphen token for document ID formatting
    if '-' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['-'], special_tokens=True)

    print(f"Tokenizer vocabulary size after augmentation: {len(tokenizer)}")
    return tokenizer


# ============================================================================
# DATASET CLASS FOR VISION LANGUAGE MODEL
# ============================================================================
class VisionLanguageDataset(Dataset):
    """Dataset for vision-language model training"""
    def __init__(self, df, processor, input_tokenizer, instruction="Generate the ICD code hierarchy for this medical image."):
        self.df = df
        self.processor = processor
        self.input_tokenizer = input_tokenizer
        self.instruction = instruction

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path_str = row['report_id']
        icd_code = row['mapped_hierarchy']
        keywords = row.get('report_keywords', '')

        # Load image
        img = load_first_image_from_path(path_str)
        if img is None:
            # Return None, will be filtered out
            return None

        # Reverse the ICD code order as done in original script
        icd_reversed = self.inverti_numeri(icd_code)

        # Create prompt with keywords and separator
        if keywords:
            prompt = f"{keywords}{self.input_tokenizer.sep_token}"
        else:
            prompt = self.instruction

        # Standard conversation format (works for most vision-language models)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": icd_reversed}
                ],
            }
        ]

        return {"messages": messages}

    @staticmethod
    def inverti_numeri(s):
        """Reverse the order of numbers in a string"""
        numeri = s.strip().split()
        numeri_invertiti = numeri[::-1]
        return '-'.join(numeri_invertiti)


def collate_fn(batch):
    """Custom collate function to filter out None samples"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return batch


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
def _decode_output_ids(ids_list, output_tokenizer):
    """Decode a list of IDs to string with fallback"""
    try:
        return output_tokenizer.decode(ids_list, skip_special_tokens=True).strip()
    except Exception:
        return " ".join(map(str, ids_list))


def evaluate_model_with_beam_search(
    model,
    df,
    input_tokenizer,
    output_tokenizer,
    processor,
    num_batches=10,
    batch_size=8,
    num_return_sequences=10,
    max_new_tokens=6,
    min_new_tokens=2,
    num_beams=100,
    k_values=[1, 3, 5, 10, 20],
    use_tree_constraints=True
):
    """Evaluate model with constrained beam search."""
    model.eval()
    model.to(device)

    # Build constraint tree
    tree_builder = None
    if use_tree_constraints:
        tree_builder = TreeBuilder()
        print("Building constraint tree from dataset...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building tree"):
            icd_code = row['mapped_hierarchy']
            temp_df = pd.DataFrame({"docid": [icd_code]})
            into_positional_docid(temp_df)
            transformed_code = temp_df["docid"].iloc[0]
            code_tokens = [int(x) for x in transformed_code.split('-')]
            tree_builder.add(code_tokens)
        print(f"Built constraint tree with {tree_builder.count} codes")

    per_example_gt = []
    per_example_preds = []

    # Split data
    n = int(0.9 * len(df))
    df_val = df.iloc[n:]

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            # Sample batch
            batch_data = df_val.sample(n=batch_size, replace=True)

            images = []
            texts = []
            answers = []

            for _, row in batch_data.iterrows():
                path_str = row['report_id']
                icd_code = row['mapped_hierarchy']

                img = load_first_image_from_path(path_str)
                if img is None:
                    continue
                images.append(img)

                raw_text = row.get("report_keywords", "")
                if raw_text is None:
                    continue

                full_text = raw_text.strip() + input_tokenizer.sep_token
                answer = VisionLanguageDataset.inverti_numeri(icd_code)

                texts.append(full_text)
                encoded_answer = torch.tensor(input_tokenizer.encode(answer), dtype=torch.long)
                answers.append(encoded_answer)

            if len(images) == 0:
                continue

            B = len(images)

            # Extract ground truth
            for i in range(B):
                row = answers[i].tolist()
                per_example_gt.append(row)

            # Process with processor
            messages_list = []
            for img, text in zip(images, texts):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": text},
                        ],
                    }
                ]
                messages_list.append(messages)

            inputs = processor.apply_chat_template(
                messages_list[0] if len(messages_list) == 1 else messages_list,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            # Prepare model inputs
            model_inputs_dict = prepare_model_inputs(inputs)

            # Move all inputs to device
            for key in model_inputs_dict:
                if isinstance(model_inputs_dict[key], torch.Tensor):
                    model_inputs_dict[key] = model_inputs_dict[key].to(device)

            input_ids = model_inputs_dict['input_ids']

            # Generate predictions
            block_preds = [[] for _ in range(B)]
            for r in range(num_return_sequences):
                preds_ids = constrained_beam_search_generate(
                    model=model,
                    input_ids=input_ids,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    decoding_tree=tree_builder,
                    num_beams=num_beams,
                    batch_size=B,
                    num_return_sequences=1,
                    vocab_size=len(output_tokenizer),
                    input_tokenizer=input_tokenizer,
                    output_tokenizer=output_tokenizer,
                    temperature=1.0,
                    model_inputs=model_inputs_dict,
                )

                if isinstance(preds_ids, list):
                    preds_ids = torch.tensor(preds_ids, device='cpu')
                preds_ids = preds_ids.cpu().tolist()

                for i in range(B):
                    if i < len(preds_ids):
                        seq = preds_ids[i] if isinstance(preds_ids[i], list) else [preds_ids[i]]
                        # Remove padding
                        pad_out_id = getattr(output_tokenizer, 'pad_token_id', None)
                        if pad_out_id is not None:
                            seq = [t for t in seq if t != pad_out_id]
                        seq = seq[:max_new_tokens]
                        block_preds[i].append(seq)

            per_example_preds.extend(block_preds)

    # Compute metrics
    total_examples = len(per_example_gt)
    metrics = {}

    for k in k_values:
        hits = 0
        for i in range(total_examples):
            gt_str = _decode_output_ids(per_example_gt[i], output_tokenizer)
            preds_k = per_example_preds[i][:k] if i < len(per_example_preds) else []

            for pred in preds_k:
                pred_str = _decode_output_ids(pred, output_tokenizer)
                if pred_str == gt_str:
                    hits += 1
                    break

        metrics[f"accuracy@{k}"] = hits / total_examples if total_examples > 0 else 0.0

    return metrics


# ============================================================================
# TRAINING AND MAIN EXECUTION
# ============================================================================
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train vision-language model for ICD code generation using Unsloth")

    # Model configuration
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                        help="Model name (Qwen/Qwen2-VL-7B-Instruct, Qwen/Qwen2-VL-2B-Instruct, or unsloth/...)")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization")

    # Data paths
    parser.add_argument("--data_path", type=str, default='/workspace/report_image_test.jsonl',
                        help="Path to training data JSONL file")
    parser.add_argument("--image_base_path", type=str, default="/datasets/MIMIC-CXR/files",
                        help="Base path for images")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum training steps")

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout")

    # Evaluation parameters
    parser.add_argument("--eval_batches", type=int, default=10,
                        help="Number of batches for evaluation")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--num_beams", type=int, default=100,
                        help="Number of beams for beam search")
    parser.add_argument("--num_return_sequences", type=int, default=10,
                        help="Number of sequences to return per example")

    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="vision-llm-icd-finetuning",
                        help="WandB project name")
    parser.add_argument("--wandb_key", type=str, default="96e66356645d31887c817fb31f43d7f55f844c0b",
                        help="WandB API key")
    parser.add_argument("--hf_token", type=str, default="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",
                        help="HuggingFace token")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for checkpoints")

    return parser.parse_args()


def main():
    global input_tokenizer, output_tokenizer, processor, device, image_path

    # Parse arguments
    args = parse_args()
    image_path = args.image_base_path

    # Login to services
    wandb.login(key=args.wandb_key)
    login(token=args.hf_token)

    print(f"Device: {device}")
    print(f"Model: {args.model}")

    # Load data
    with open(args.data_path, 'r') as f:
        records = [json.loads(line) for line in f]
    df0 = pd.DataFrame(records)

    df = filter_valid_entries(df0)
    print(f"Valid rows with images: {len(df)}")
    df = df.sample(frac=1).reset_index(drop=True)

    # ============================================================================
    # LOAD MODEL
    # ============================================================================
    print("\n=== Loading model with Unsloth ===")

    model, tokenizer, processor = load_model_and_processor(
        model_name=args.model,
        use_4bit=args.use_4bit
    )

    # Augment tokenizer for ICD codes
    print("\n=== Augmenting tokenizers ===")
    input_tokenizer = augment_tokenizer(tokenizer, is_output_tokenizer=False)

    # Create a separate output tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(args.model)
    output_tokenizer = augment_tokenizer(base_tokenizer, is_output_tokenizer=True)

    # Add LoRA adapters
    print("\n=== Adding LoRA adapters ===")

    model = add_lora_to_model(
        model=model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    # ============================================================================
    # PREPARE DATASET
    # ============================================================================
    print("\n=== Preparing dataset ===")
    dataset = VisionLanguageDataset(df, processor, input_tokenizer)

    # Convert to format expected by trainer
    converted_dataset = []
    for i in range(len(dataset)):
        item = dataset[i]
        if item is not None:
            converted_dataset.append(item)

    print(f"Dataset size: {len(converted_dataset)}")

    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    # Create a short name from model path
    model_name_short = args.model.split('/')[-1].replace("-", "_")
    name = f"{model_name_short}_icd_r{args.lora_r}_lr{args.learning_rate}_bs{args.batch_size}_epochs{args.epochs}"

    print(f"\n=== Training configuration ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Max steps: {args.max_steps}")

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "model": args.model,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
        name=name
    )

    # ============================================================================
    # INITIALIZE TRAINER
    # ============================================================================
    # Prepare model for training with Unsloth
    FastVisionModel.for_training(model)
    data_collator = UnslothVisionDataCollator(model, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=5,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
            report_to="wandb",

            # Required for vision finetuning
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
        ),
    )

    # ============================================================================
    # TRAIN MODEL
    # ============================================================================
    print("\n=== Starting training ===")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # Show memory stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # ============================================================================
    # SAVE MODEL
    # ============================================================================
    print("\n=== Saving model ===")
    model.save_pretrained(f"{name}_lora")
    tokenizer.save_pretrained(f"{name}_lora")

    # ============================================================================
    # EVALUATION
    # ============================================================================
    print("\n=== Starting evaluation ===")

    # Prepare model for inference with Unsloth
    FastVisionModel.for_inference(model)

    # Evaluation with constrained beam search
    metrics = evaluate_model_with_beam_search(
        model=model,
        df=df,
        input_tokenizer=input_tokenizer,
        output_tokenizer=output_tokenizer,
        processor=processor,
        num_batches=args.eval_batches,
        batch_size=args.eval_batch_size,
        num_return_sequences=args.num_return_sequences,
        max_new_tokens=6,
        min_new_tokens=2,
        num_beams=args.num_beams,
        k_values=[1, 3, 5, 10, 20],
        use_tree_constraints=True
    )

    print("\n=== Evaluation Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        wandb.log({f"eval/{metric}": value})

    wandb.finish()
    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()
