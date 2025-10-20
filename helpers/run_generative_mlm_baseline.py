"""
This code performs ICD classification by using a pretrained LLM to generate the list of ICD codes starting from the medical report of a patient.
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
from tqdm import tqdm
import re
from datetime import datetime
import os
import glob

from vllm import LLM, SamplingParams
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Base path for MIMIC-CXR images
MIMIC_CXR_BASE_PATH = os.getenv("MIMIC_CXR_BASE_PATH", "/datasets/MIMIC-CXR/files/")

PROMPT = """
You are an expert ICD-9 medical coder. Given a medical report and chest X-ray images, identify and list all ICD-9 codes that apply to the conditions, diagnoses, symptoms, and procedures described in the report and visible in the images.

**Medical Report:**
{medical_report}

**Task:** Analyze both the medical report and the provided chest X-ray images to identify ALL applicable ICD-9 codes for every condition, symptom, diagnosis, and procedure mentioned or visible. Consider both primary and secondary diagnoses, as well as any comorbidities or related conditions.

**IMPORTANT FORMATTING REQUIREMENTS:**
- Return ONLY ICD-9 codes in comma-separated format
- Use standard ICD-9-CM 4-5 digit formatting (e.g., 250.00, 401.9, 486)
- NO text descriptions, explanations, or category names
- NO additional text before or after the code list
- Follow official ICD-9-CM coding guidelines
"""

PROMPT_COT = """
You are an expert ICD-9 medical coder. Given a medical report and chest X-ray images, identify and list all ICD-9 codes that apply to the conditions, diagnoses, symptoms, and procedures described in the report and visible in the images.

**Medical Report:**
{medical_report}

**Task:** Analyze both the medical report and the provided chest X-ray images step-by-step to identify ALL applicable ICD-9 codes for every condition, symptom, diagnosis, and procedure mentioned or visible. Consider both primary and secondary diagnoses, as well as any comorbidities or related conditions.

**CRITICAL: You MUST use ICD-9-CM codes ONLY. Do NOT use ICD-10 codes. ICD-9 codes are typically 3-5 digits, sometimes starting with V or E. ICD-10 codes often start with other letters (e.g., J, I, R) followed by digits. Ensure NO ICD-10 codes are included.**

**Instructions:**
1. First, carefully examine the chest X-ray images and identify any visible abnormalities, medical devices, or anatomical findings.
2. Then, carefully read through the medical report and identify all medical conditions, symptoms, diagnoses, or procedures mentioned.
3. For each identified condition/symptom/diagnosis/procedure from both images and text, determine the appropriate ICD-9-CM code (3-5 digits, may have decimal point).
4. Consider any comorbidities, complications, or related conditions.
5. Provide your reasoning for each code selection using ONLY ICD-9 codes in your explanation (e.g., "Condition X: ICD-9 code Y for Z reason.").
6. Finally, list all the ICD-9 codes identified in a comma-separated format in the designated section.

**Format your response exactly as follows:**

**STEP 1 - CHEST X-RAY FINDINGS:**
[Describe findings visible in the chest X-ray images]

**STEP 2 - MEDICAL REPORT CONDITIONS:**
[List and describe each condition, symptom, diagnosis, or procedure found in the report]

**STEP 3 - ICD-9 CODE REASONING:**
[For each condition from images and report, explain which ICD-9-CM code applies and why. Use only ICD-9 format codes in your explanations.]

**STEP 4 - FINAL ICD-9 CODES:**
[List ONLY the ICD-9 codes in comma-separated format. No explanations, no descriptions, no additional text, just the codes.]
"""

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the merged dataset from TSV file."""
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    print(f"Loaded {len(df)} records")
    return df

def get_image_paths_from_id(sample_id: str) -> List[str]:
    """
    Extract image paths from the sample ID.
    
    Args:
        sample_id: String like "p10.p10075925.s51010496"
        
    Returns:
        List of full paths to JPG files
    """
    # Split the ID by dots to get path components
    parts = sample_id.split('.')
    if len(parts) != 3:
        print(f"Warning: Unexpected ID format: {sample_id}")
        return []
    
    # Construct the folder path
    folder_path = Path(MIMIC_CXR_BASE_PATH) / parts[0] / parts[1] / parts[2]
    
    if not folder_path.exists():
        print(f"Warning: Folder does not exist: {folder_path}")
        return []
    
    # Find all JPG files in the folder
    jpg_files = list(folder_path.glob("*.jpg"))
    
    if not jpg_files:
        print(f"Warning: No JPG files found in: {folder_path}")
        return []
    
    # Convert to strings and sort for consistency
    image_paths = sorted([str(path) for path in jpg_files])
    print(f"Found {len(image_paths)} images for ID {sample_id}")
    
    return image_paths

def format_report_text(report_text: str, section_filter: str = None) -> str:
    """Format report text for better readability by adding ### before section titles.
    
    Args:
        report_text: The raw report text
        section_filter: If provided, only include text from this section onwards (e.g., "FINDINGS:")
    """
    if pd.isna(report_text) or report_text.strip() == '':
        return "No report text available."
    
    formatted_text = report_text.strip()
    
    # Apply section filtering if specified
    if section_filter:
        # Remove trailing colon if present for matching
        filter_section = section_filter.rstrip(':').strip().upper()
        
        # Find the position of the specified section
        section_pattern = r'\b' + re.escape(filter_section) + r'\s*:'
        match = re.search(section_pattern, formatted_text, re.IGNORECASE)
        
        if match:
            # Keep text from the matched section onwards
            formatted_text = formatted_text[match.start():]
            print(f"Filtered report from section '{filter_section}': kept {len(formatted_text)} characters")
        else:
            print(f"Warning: Section '{filter_section}' not found in report, keeping full text")
    
    # Common section titles in medical reports
    section_titles = [
        'INDICATION', 'INDICATIONS', 'TECHNIQUE', 'COMPARISON', 'COMPARISONS',
        'FINDINGS', 'IMPRESSION', 'IMPRESSIONS', 'CONCLUSION', 'CONCLUSIONS',
        'HISTORY', 'CLINICAL HISTORY', 'EXAMINATION', 'STUDY', 'PROCEDURE',
        'REPORT', 'FINAL REPORT', 'ADDENDUM', 'RECOMMENDATION', 'RECOMMENDATIONS',
        'COMMENT', 'COMMENTS', 'NOTE', 'NOTES', 'DISCUSSION'
    ]
    
    # Sort section titles by length (longest first) to avoid partial matches
    section_titles_sorted = sorted(section_titles, key=len, reverse=True)
    
    # Add ### before section titles with more careful pattern matching
    for title in section_titles_sorted:
        # Pattern 1: Section title followed by colon, ensuring we don't break existing words
        # Match only at word boundaries and when followed by colon
        pattern1 = r'\b(' + re.escape(title) + r')\s*:'
        replacement1 = r'### \1:'
        formatted_text = re.sub(pattern1, replacement1, formatted_text, flags=re.IGNORECASE)
    
    # Clean up formatting issues
    # Remove ### from the middle of words (if it got inserted incorrectly)
    formatted_text = re.sub(r'(\w)### (\w)', r'\1\2', formatted_text)
    
    # Ensure proper spacing around ### markers
    formatted_text = re.sub(r'### ([^:\n]+:)\s*', r'### \1 ', formatted_text)
    
    # Add line breaks after ### sections if they don't already have them
    formatted_text = re.sub(r'(### [^:]+:)([^\n])', r'\1\n\2', formatted_text)
    
    # Add line breaks before ### sections for better separation
    formatted_text = re.sub(r'([^\n])(### )', r'\1\n\n\2', formatted_text)
    
    # Clean up multiple consecutive line breaks (max 2)
    formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
    
    # Clean up any ### at the start of the text
    formatted_text = re.sub(r'^### ', '', formatted_text)
    
    return formatted_text.strip()

def prepare_prompts(df: pd.DataFrame) -> List[str]:
    """Prepare prompts for batch inference."""
    prompts = []
    for _, row in df.iterrows():
        report_text = row['report_text']
        formatted_report = format_report_text(report_text)
        
        prompt = PROMPT.format(medical_report=formatted_report)
        prompts.append(prompt)
    
    return prompts

def parse_icd_codes(response: str, use_cot: bool = False) -> Set[str]:
    """Parse ICD codes from model response, robustly handling CoT and extracting all code-like strings."""
    response = response.strip()
    codes_text = ""

    if use_cot:
        # For CoT responses, strictly look for the "STEP 4 - FINAL ICD-9 CODES:" section
        final_codes_section_match = re.search(
            r'(?i)\*\*STEP 4 - FINAL ICD-9 CODES:\*\*\s*\n?\s*(.*?)(?=\n\s*\*\*STEP \d|\n\s*\*\*|$)', 
            response,
            re.DOTALL
        )
        if final_codes_section_match:
            codes_text = final_codes_section_match.group(1).strip()
        else:
            # Fallback: look for any line that contains comma-separated codes after the final step
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if 'STEP 4' in line and 'FINAL' in line and 'CODES' in line:
                    # Take the next few lines after this header
                    remaining_lines = lines[i+1:]
                    codes_text = '\n'.join(remaining_lines[:5])  # Take up to 5 lines
                    break
            
            if not codes_text:
                # Last resort: take everything after "STEP 4"
                if 'STEP 4' in response:
                    codes_text = response.split('STEP 4')[-1]
                else:
                    codes_text = response
    else:
        # For non-CoT, the entire response should be codes
        codes_text = response
    
    # Clean up the codes text from any common prefixes/suffixes, and unwanted explanations
    codes_text = re.sub(r'^(ICD-9 codes?:?\s*)', '', codes_text, flags=re.IGNORECASE)
    codes_text = re.sub(r'^(codes?:?\s*)', '', codes_text, flags=re.IGNORECASE)
    codes_text = re.sub(r'^(final\s+codes?:?\s*)', '', codes_text, flags=re.IGNORECASE)
    codes_text = re.sub(r'Example format.*?:', '', codes_text, flags=re.IGNORECASE)
    
    # Remove any explanatory text after codes within the same line (e.g., "401.9 (Hypertension)")
    codes_text = re.sub(r'(\b[A-Z]?\d{2,3}(?:\.\d{1,2})?[A-Z]?)\s*\(.*?\)', r'\1', codes_text)

    # Updated pattern to capture complete ICD codes including those with letters at the end:
    # - Optional single uppercase letter at start (V, E codes)
    # - 2-5 digits 
    # - Optional decimal point followed by 1-2 digits
    # - Optional single uppercase letter at end (like 9920A)
    icd_pattern = r'\b([A-Z]?\d{2,5}(?:\.\d{1,2})?[A-Z]?)\b'
    potential_codes = re.findall(icd_pattern, codes_text)
    
    valid_codes = set()
    for code in potential_codes:
        # Additional validation: 
        # - Code should have at least 2 characters
        # - Should not be just a decimal number (like ".0", ".44")
        # - Should not be pure numbers that are too short (like single digits)
        # - Should not be just a single letter
        if (len(code) >= 2 and 
            not code.startswith('.') and 
            not (code.isdigit() and len(code) == 1) and
            not (code.isalpha() and len(code) == 1)):
            valid_codes.add(code)
    
    # If no valid codes found with regex, try manual parsing of comma-separated values
    if not valid_codes and (',' in codes_text):
        # Split by comma and clean each potential code
        potential_codes = [code.strip() for code in codes_text.split(',')]
        
        for code in potential_codes:
            # Remove any trailing punctuation or whitespace, but preserve letters and dots
            code = re.sub(r'[^\w\.]', '', code).strip()
            
            # Validate that this looks like a proper ICD code (allow letters at start or end)
            if (re.match(r'^[A-Z]?\d{2,5}(\.\d{1,2})?[A-Z]?$', code) and 
                len(code) >= 2 and 
                not code.startswith('.') and
                not (code.isalpha() and len(code) == 1)):
                valid_codes.add(code)
    
    return valid_codes


def parse_gold_codes(gold_codes_str: str) -> Set[str]:
    """Parse gold standard ICD codes from the dataset."""
    if pd.isna(gold_codes_str) or gold_codes_str.strip() == '':
        return set()
    
    codes = gold_codes_str.split(';')
    return {code.strip() for code in codes if code.strip()}

def run_inference(model: LLM, prompts: List[str], sampling_params: SamplingParams, batch_size: int = 32) -> List[str]:
    """Run batch inference using vLLM."""
    print(f"Running inference on {len(prompts)} prompts with batch size {batch_size}")
    
    all_responses = []
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Running inference"):
        batch_prompts = prompts[i:i+batch_size]
        
        # Generate responses for the batch
        outputs = model.generate(batch_prompts, sampling_params)
        
        # Extract generated text
        batch_responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            batch_responses.append(generated_text)
        
        all_responses.extend(batch_responses)
    
    return all_responses

def calculate_metrics(y_true_sets: List[Set[str]], y_pred_sets: List[Set[str]]) -> Dict:
    """Calculate comprehensive evaluation metrics."""
    # Calculate per-sample metrics
    per_sample_precision = []
    per_sample_recall = []
    per_sample_f1 = []
    exact_matches = 0
    
    for true_set, pred_set in zip(y_true_sets, y_pred_sets):
        if len(pred_set) == 0 and len(true_set) == 0:
            # Both empty - perfect match
            precision = 1.0
            recall = 1.0
            f1 = 1.0
            exact_matches += 1
        elif len(pred_set) == 0:
            # Predicted nothing but should have predicted something
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        elif len(true_set) == 0:
            # Predicted something but should have predicted nothing
            precision = 0.0
            recall = 0.0  # or could be undefined
            f1 = 0.0
        else:
            # Both non-empty
            intersection = true_set.intersection(pred_set)
            precision = len(intersection) / len(pred_set)
            recall = len(intersection) / len(true_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if true_set == pred_set:
                exact_matches += 1
        
        per_sample_precision.append(precision)
        per_sample_recall.append(recall)
        per_sample_f1.append(f1)
    
    # Macro averages (average of per-sample metrics)
    macro_precision = np.mean(per_sample_precision)
    macro_recall = np.mean(per_sample_recall)
    macro_f1 = np.mean(per_sample_f1)
    
    # Micro averages (global counts)
    total_true_positives = sum(len(true_set.intersection(pred_set)) 
                              for true_set, pred_set in zip(y_true_sets, y_pred_sets))
    total_predicted = sum(len(pred_set) for pred_set in y_pred_sets)
    total_actual = sum(len(true_set) for true_set in y_true_sets)
    
    micro_precision = total_true_positives / total_predicted if total_predicted > 0 else 0.0
    micro_recall = total_true_positives / total_actual if total_actual > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # Exact match accuracy
    exact_match_accuracy = exact_matches / len(y_true_sets)
    
    # Additional statistics
    avg_predicted_codes = np.mean([len(pred_set) for pred_set in y_pred_sets])
    avg_actual_codes = np.mean([len(true_set) for true_set in y_true_sets])
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'exact_match_accuracy': exact_match_accuracy,
        'avg_predicted_codes': avg_predicted_codes,
        'avg_actual_codes': avg_actual_codes,
        'total_samples': len(y_true_sets),
        'total_actual_codes': total_actual,
        'total_predicted_codes': total_predicted,
        'total_true_positives': total_true_positives
    }

def calculate_metrics_sklearn(y_true_sets: List[Set[str]], y_pred_sets: List[Set[str]]) -> Dict:
    """Calculate comprehensive evaluation metrics using scikit-learn."""
    # Get all unique labels from both true and predicted sets
    all_labels = set()
    for true_set in y_true_sets:
        all_labels.update(true_set)
    for pred_set in y_pred_sets:
        all_labels.update(pred_set)
    
    all_labels = sorted(list(all_labels))  # Sort for consistency
    
    # Convert sets to binary matrix format for scikit-learn
    mlb = MultiLabelBinarizer(classes=all_labels)
    
    # Convert to list of lists for MLB
    y_true_lists = [list(true_set) for true_set in y_true_sets]
    y_pred_lists = [list(pred_set) for pred_set in y_pred_sets]
    
    # Transform to binary matrices
    y_true_binary = mlb.fit_transform(y_true_lists)
    y_pred_binary = mlb.transform(y_pred_lists)
    
    # Calculate metrics using scikit-learn
    # Micro averages
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='micro', zero_division=0
    )
    
    # Macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='macro', zero_division=0
    )
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='weighted', zero_division=0
    )
    
    # Sample-based averages (per-sample metrics)
    samples_precision, samples_recall, samples_f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='samples', zero_division=0
    )
    
    # Exact match accuracy (subset accuracy)
    exact_match_accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    # Calculate additional statistics
    total_true_positives = sum(len(true_set.intersection(pred_set)) 
                              for true_set, pred_set in zip(y_true_sets, y_pred_sets))
    total_predicted = sum(len(pred_set) for pred_set in y_pred_sets)
    total_actual = sum(len(true_set) for true_set in y_true_sets)
    
    avg_predicted_codes = np.mean([len(pred_set) for pred_set in y_pred_sets])
    avg_actual_codes = np.mean([len(true_set) for true_set in y_true_sets])
    
    return {
        'micro_precision': float(micro_precision),
        'micro_recall': float(micro_recall),
        'micro_f1': float(micro_f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'samples_precision': float(samples_precision),
        'samples_recall': float(samples_recall),
        'samples_f1': float(samples_f1),
        'exact_match_accuracy': float(exact_match_accuracy),
        'avg_predicted_codes': float(avg_predicted_codes),
        'avg_actual_codes': float(avg_actual_codes),
        'total_samples': len(y_true_sets),
        'total_actual_codes': total_actual,
        'total_predicted_codes': total_predicted,
        'total_true_positives': total_true_positives,
        'num_unique_labels': len(all_labels)
    }

def save_results(results: Dict, predictions: List, output_dir: str, split_name: str, model_name: str, num_samples: int):
    """Save evaluation results and predictions with model name and sample count in filename."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean model name for filename (replace slashes and special characters)
    clean_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # Create filename with model name and sample count
    base_filename = f"{split_name}_{clean_model_name}_{num_samples}samples"
    
    # Save metrics
    metrics_file = output_path / f"{base_filename}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # # Save predictions
    # predictions_file = output_path / f"{base_filename}_predictions.json"
    # with open(predictions_file, 'w') as f:
    #     json.dump(predictions, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print(f"  Metrics: {metrics_file.name}")
    # print(f"  Predictions: {predictions_file.name}")

def save_model_stats_to_csv(results: Dict, args: argparse.Namespace, split_name: str, num_samples: int, output_dir: str):
    """Save model statistics to a CSV file, creating it if it doesn't exist or appending if it does."""
    csv_path = Path(output_dir) / "model_evaluation_stats.csv"
    
    # Prepare the row data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create model name suffix for identification
    model_name_suffix = f"{args.model_name}"
    if args.use_cot:
        model_name_suffix += "_cot"
    if args.format_reports:
        model_name_suffix += "_fmt"
    if args.section_filter:
        clean_section = args.section_filter.replace(':', '').replace(' ', '').lower()
        model_name_suffix += f"_{clean_section}"
    if args.use_images:
        model_name_suffix += "_multimodal"
    
    row_data = {
        'timestamp': timestamp,
        'model_name': args.model_name,
        'model_config': model_name_suffix,
        'split': split_name,
        'num_samples': num_samples,
        'use_cot': args.use_cot,
        'enable_thinking': args.enable_thinking,
        'format_reports': args.format_reports,
        'section_filter': args.section_filter if args.section_filter else '',
        'use_images': args.use_images,
        'batch_size': args.batch_size,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'micro_precision': results['micro_precision'],
        'micro_recall': results['micro_recall'],
        'micro_f1': results['micro_f1'],
        'macro_precision': results['macro_precision'],
        'macro_recall': results['macro_recall'],
        'macro_f1': results['macro_f1'],
        'weighted_precision': results['weighted_precision'],
        'weighted_recall': results['weighted_recall'],
        'weighted_f1': results['weighted_f1'],
        'samples_precision': results['samples_precision'],
        'samples_recall': results['samples_recall'],
        'samples_f1': results['samples_f1'],
        'exact_match_accuracy': results['exact_match_accuracy'],
        'avg_predicted_codes': results['avg_predicted_codes'],
        'avg_actual_codes': results['avg_actual_codes'],
        'total_actual_codes': results['total_actual_codes'],
        'total_predicted_codes': results['total_predicted_codes'],
        'total_true_positives': results['total_true_positives'],
        'num_unique_labels': results['num_unique_labels']
    }
    
    # Convert to DataFrame
    new_row_df = pd.DataFrame([row_data])
    
    # Check if CSV exists
    if csv_path.exists():
        # Append to existing CSV
        new_row_df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"Appended model stats to existing CSV: {csv_path}")
    else:
        # Create new CSV
        new_row_df.to_csv(csv_path, mode='w', header=True, index=False)
        print(f"Created new CSV with model stats: {csv_path}")

def print_sample_prompts(prompts: List[str], num_samples: int = 3):
    """Print sample prompts to inspect their quality."""
    print(f"\n{'='*80}")
    print("SAMPLE PROMPTS FOR QUALITY INSPECTION")
    print(f"{'='*80}")
    
    for i in range(min(num_samples, len(prompts))):
        print(f"\n--- PROMPT {i+1} ---")
        print(prompts[i])
        print(f"--- END PROMPT {i+1} ---")
        if i < min(num_samples, len(prompts)) - 1:
            print("\n" + "-"*40)
    
    print(f"\n{'='*80}")

def print_sample_conversations(conversations: List[List[Dict]], num_samples: int = 3):
    """Print sample conversations to inspect their quality."""
    print(f"\n{'='*80}")
    print("SAMPLE CONVERSATIONS FOR QUALITY INSPECTION")
    print(f"{'='*80}")
    
    for i in range(min(num_samples, len(conversations))):
        print(f"\n--- CONVERSATION {i+1} ---")
        conversation = conversations[i]
        
        for j, message in enumerate(conversation):
            role = message["role"].upper()
            content = message["content"]
            
            print(f"\n{role} MESSAGE:")
            # Handle both string content and list content (for multimodal)
            if isinstance(content, str):
                # Truncate very long content for readability
                if len(content) > 500:
                    print(content[:500] + "... [TRUNCATED]")
                else:
                    print(content)
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if item["type"] == "text":
                        text_content = item["text"]
                        if len(text_content) > 500:
                            print(f"TEXT: {text_content[:500]}... [TRUNCATED]")
                        else:
                            print(f"TEXT: {text_content}")
                    elif item["type"] == "image":
                        print(f"IMAGE: {item['url']}")
        
        print(f"--- END CONVERSATION {i+1} ---")
        if i < min(num_samples, len(conversations)) - 1:
            print("\n" + "-"*40)
    
    print(f"\n{'='*80}")

def print_metrics(results: Dict, split_name: str):
    """Print evaluation metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS - {split_name.upper()}")
    print(f"{'='*60}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Number of unique ICD codes: {results['num_unique_labels']}")
    print(f"Average actual codes per sample: {results['avg_actual_codes']:.2f}")
    print(f"Average predicted codes per sample: {results['avg_predicted_codes']:.2f}")
    print(f"\nMICRO METRICS (global aggregation):")
    print(f"  Precision: {results['micro_precision']:.4f}")
    print(f"  Recall:    {results['micro_recall']:.4f}")
    print(f"  F1-Score:  {results['micro_f1']:.4f}")
    print(f"\nMACRO METRICS (unweighted average):")
    print(f"  Precision: {results['macro_precision']:.4f}")
    print(f"  Recall:    {results['macro_recall']:.4f}")
    print(f"  F1-Score:  {results['macro_f1']:.4f}")
    print(f"\nWEIGHTED METRICS (support-weighted average):")
    print(f"  Precision: {results['weighted_precision']:.4f}")
    print(f"  Recall:    {results['weighted_recall']:.4f}")
    print(f"  F1-Score:  {results['weighted_f1']:.4f}")
    print(f"\nSAMPLES METRICS (per-sample average):")
    print(f"  Precision: {results['samples_precision']:.4f}")
    print(f"  Recall:    {results['samples_recall']:.4f}")
    print(f"  F1-Score:  {results['samples_f1']:.4f}")
    print(f"\nEXACT MATCH (subset accuracy):")
    print(f"  Accuracy:  {results['exact_match_accuracy']:.4f}")
    print(f"{'='*60}")

def prepare_chat_conversations(df: pd.DataFrame, use_cot: bool = False, format_reports: bool = True, 
                             section_filter: str = None, use_images: bool = False) -> List[List[Dict]]:
    """Prepare chat conversations for batch inference using chat templates."""
    conversations = []
    
    # Choose the appropriate prompt template
    prompt_template = PROMPT_COT if use_cot else PROMPT
    
    for _, row in df.iterrows():
        report_text = row['report_text']
        sample_id = row['id']
        
        # Apply formatting conditionally
        if format_reports:
            formatted_report = format_report_text(report_text, section_filter=section_filter)
        else:
            # Even without formatting, apply section filtering if specified
            if section_filter and not pd.isna(report_text):
                filter_section = section_filter.rstrip(':').strip().upper()
                section_pattern = r'\b' + re.escape(filter_section) + r'\s*:'
                match = re.search(section_pattern, report_text, re.IGNORECASE)
                
                if match:
                    formatted_report = report_text[match.start():]
                else:
                    formatted_report = report_text
            else:
                formatted_report = report_text if not pd.isna(report_text) else "No report text available."
        
        # Use the selected prompt template and format it with the report
        user_message_text = prompt_template.format(medical_report=formatted_report)
        
        if use_images:
            # Get image paths for this sample
            image_paths = get_image_paths_from_id(sample_id)
            
            # Create multimodal content
            content = []
            
            # Add all images first
            for image_path in image_paths:
                content.append({
                    "type": "image",
                    "url": f"file://{image_path}"
                })
            
            # Add the text prompt
            content.append({
                "type": "text",
                "text": user_message_text
            })
            
            conversation = [
                {
                    "role": "user",
                    "content": content
                }
            ]
        else:
            # Text-only conversation
            conversation = [
                {
                    "role": "user",
                    "content": user_message_text
                }
            ]
        
        conversations.append(conversation)
    
    return conversations

def run_chat_inference(model: LLM, conversations: List[List[Dict]], sampling_params: SamplingParams, batch_size: int = 32, enable_thinking : bool = False) -> List[str]:
    """Run batch inference using chat templates with vLLM."""
    print(f"Running chat inference on {len(conversations)} conversations with batch size {batch_size}")
    
    all_responses = []
    
    # Process in batches
    for i in tqdm(range(0, len(conversations), batch_size), desc="Running chat inference"):
        batch_conversations = conversations[i:i+batch_size]
        
        # Generate responses for the batch using chat method
        outputs = model.chat(
            batch_conversations,
            sampling_params=sampling_params,
            use_tqdm=False,
            #chat_template_kwargs={"enable_thinking":enable_thinking}
        )
        
        # Extract generated text
        batch_responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            batch_responses.append(generated_text)
        
        all_responses.extend(batch_responses)
    
    return all_responses

def main():
    parser = argparse.ArgumentParser(description='ICD Code Generation Baseline using LLM')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name for vLLM (e.g., microsoft/DialoGPT-medium)')
    parser.add_argument('--data_dir', type=str, default='data/mimic-cxr-icd-merged',
                       help='Directory containing the merged dataset files')
    parser.add_argument('--output_dir', type=str, default='results/baselines_genCoding',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--max_tokens', type=int, default=200,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for sampling')
    parser.add_argument('--splits', nargs='+', default=['test'],
                       help='Dataset splits to evaluate (train, dev, test)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    parser.add_argument('--use_cot', action='store_true',
                       help='Enable Chain-of-Thought prompting for more detailed reasoning')
    parser.add_argument('--format_reports', action='store_true',
                       help='Enable report text formatting with section headers (### markers)')
    parser.add_argument('--section_filter', type=str, default=None,
                       help='Filter report text from this section onwards (e.g., "FINDINGS", "IMPRESSION")')
    parser.add_argument('--enable_thinking', action='store_true',
                       help='Enable thinking mode for the model to reason through steps')
    parser.add_argument('--use_images', action='store_true',
                       help='Enable multimodal input with chest X-ray images')
    
    args = parser.parse_args()

    print("********** CONFIGURATION **********")
    print(f"Model Name: {args.model_name}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Dataset Splits: {args.splits}")
    print(f"Max Samples: {args.max_samples if args.max_samples else 'No limit'}")
    print(f"Use Chain-of-Thought: {'Enabled' if args.use_cot else 'Disabled'}")
    print(f"Format Reports: {'Enabled' if args.format_reports else 'Disabled'}")
    print(f"Section Filter: {args.section_filter if args.section_filter else 'None'}")
    print(f"Enable Thinking Mode: {'Enabled' if args.enable_thinking else 'Disabled'}")
    print(f"Use Images (Multimodal): {'Enabled' if args.use_images else 'Disabled'}")
    print("************************************\n")
    
    # Adjust max_tokens if using CoT (needs more tokens for reasoning)
    if args.use_cot and args.max_tokens < 500:
        args.max_tokens = 800
        print(f"Increased max_tokens to {args.max_tokens} for Chain-of-Thought prompting")
    
    # Initialize vLLM model
    print(f"Loading model: {args.model_name}")
    model = LLM(model=args.model_name, max_model_len=16384)#, download_dir=os.getenv('HF_HOME', None))

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["\n\n", "**", "---"] if not args.use_cot else ["**STEP 5:", "---"]  # Different stop tokens for CoT with images
    )
    
    # Process each split
    for split in args.splits:
        print(f"\nProcessing {split} split...")
        print(f"Chain-of-Thought: {'Enabled' if args.use_cot else 'Disabled'}")
        print(f"Report Formatting: {'Enabled' if args.format_reports else 'Disabled'}")
        print(f"Section Filter: {args.section_filter if args.section_filter else 'None'}")
        print(f"Multimodal (Images): {'Enabled' if args.use_images else 'Disabled'}")
        
        # Load dataset
        data_file = Path(args.data_dir) / f"{split}.tsv"
        if not data_file.exists():
            print(f"Warning: {data_file} not found, skipping...")
            continue
        
        df = load_dataset(str(data_file))
        
        # Limit samples if specified
        if args.max_samples and len(df) > args.max_samples:
            df = df.head(args.max_samples)
            print(f"Limited to {args.max_samples} samples for testing")
        
        # Store actual number of samples used
        num_samples_used = len(df)
        
        # Prepare chat conversations with CoT, formatting, section filter, and image flags
        conversations = prepare_chat_conversations(
            df, 
            use_cot=args.use_cot, 
            format_reports=args.format_reports, 
            section_filter=args.section_filter,
            use_images=args.use_images
        )
        
        # Print sample conversations for quality inspection
        print_sample_conversations(conversations, num_samples=2 if args.use_cot else 3)
        
        # Run chat inference
        responses = run_chat_inference(model, conversations, sampling_params, args.batch_size, args.enable_thinking)
        
        # Parse predictions and gold standard with CoT flag
        predicted_codes = [parse_icd_codes(response, use_cot=args.use_cot) for response in responses]
        gold_codes = [parse_gold_codes(row['formatted_icd_codes']) for _, row in df.iterrows()]
        
        # Calculate metrics using scikit-learn
        results = calculate_metrics_sklearn(gold_codes, predicted_codes)
        
        # Print results
        print_metrics(results, split)
        
        # Prepare detailed predictions for saving
        detailed_predictions = []
        for i, (_, row) in enumerate(df.iterrows()):
            detailed_predictions.append({
                'id': row['id'],
                'gold_codes': list(gold_codes[i]),
                'predicted_codes': list(predicted_codes[i]),
                'raw_response': responses[i],
                'intersection': list(gold_codes[i].intersection(predicted_codes[i])),
                'precision': len(gold_codes[i].intersection(predicted_codes[i])) / len(predicted_codes[i]) if len(predicted_codes[i]) > 0 else 0.0,
                'recall': len(gold_codes[i].intersection(predicted_codes[i])) / len(gold_codes[i]) if len(gold_codes[i]) > 0 else 0.0,
                'use_cot': args.use_cot,
                'format_reports': args.format_reports,
                'section_filter': args.section_filter,
                'use_images': args.use_images
            })
        
        # Create model name suffix for CoT, formatting, section filter, and multimodal
        model_name_suffix = f"{args.model_name}"
        if args.use_cot:
            model_name_suffix += "_cot"
        if args.format_reports:
            model_name_suffix += "_fmt"
        if args.section_filter:
            clean_section = args.section_filter.replace(':', '').replace(' ', '').lower()
            model_name_suffix += f"_{clean_section}"
        if args.use_images:
            model_name_suffix += "_multimodal"
        
        # Save results with model name and sample count
        save_results(results, detailed_predictions, args.output_dir, split, model_name_suffix, num_samples_used)
        
        # Save model statistics to CSV
        save_model_stats_to_csv(results, args, split, num_samples_used, args.output_dir)

if __name__ == "__main__":
    main()