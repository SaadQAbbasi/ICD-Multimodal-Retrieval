from dotenv import load_dotenv
load_dotenv()

import random
import wandb
import argparse
import io
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os
from tqdm import tqdm
import glob
from huggingface_hub import login
from typing import List, Dict, Any, Optional, Union
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers.tokenization_utils import AddedToken as TransformersAddedToken
import numpy as np
from collections import defaultdict

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
image_path = "/datasets/MIMIC-CXR/files"
input_tokenizer = None
output_tokenizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================
def load_first_image_from_path(path_str, base_dir=image_path):
    """
    Load the first available image from the folder path.

    Args:
        path_str: Path string in format "p10.p10002430.s53254222"
        base_dir: Base directory for images

    Returns:
        PIL Image or None if not found
    """
    try:
        modelFolder, folder, subfolder = path_str.split(".")
        image_dir = os.path.join(base_dir, modelFolder, folder, subfolder)

        # Search for common image file extensions
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in img_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))

        if not image_files:
            return None

        # Load the first image found
        return Image.open(image_files[0])
    except Exception:
        return None


def load_three_images_from_path(path_str, base_dir=image_path):
    """
    Load the first three available images from the folder.
    If fewer than three images exist, repeat some to always return three images.

    Args:
        path_str: Path string in format "p10.p10002430.s53254222"
        base_dir: Base directory for images

    Returns:
        List of three PIL Images or [None, None, None] if not found
    """
    try:
        modelFolder, folder, subfolder = path_str.split(".")
        image_dir = os.path.join(base_dir, modelFolder, folder, subfolder)

        # Search for common image file extensions
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in img_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))

        if not image_files:
            return [None, None, None]

        # Sort files by name for consistency
        image_files.sort()

        # Load up to three images, cycling if necessary
        images = [Image.open(image_files[i % len(image_files)]) for i in range(3)]
        return images

    except Exception:
        return [None, None, None]


def filter_valid_entries(df):
    """
    Remove DataFrame rows where the associated image is not accessible.

    Args:
        df: DataFrame with 'report_id' column

    Returns:
        Filtered DataFrame with only valid entries
    """
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

def create_input_tokenizer(
    pretrained_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    hf_access_token: Optional[str] = None
) -> PreTrainedTokenizerFast:
    """
    Create an input tokenizer based on a pre-trained model and extend it with additional tokens.

    Args:
        pretrained_model_name: Name of the pre-trained model to use as base
        hf_access_token: HuggingFace access token for downloading the model (optional)

    Returns:
        HuggingFace tokenizer for input text
    """
    # Load the pre-trained tokenizer
    input_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name,
        token=hf_access_token,
        padding_side="left"
    )
    
    
    # Set pad token
    input_tokenizer.pad_token = '|'
    input_tokenizer.pad_token_id = input_tokenizer.encode('|')[-1]

    # Set separator token
    input_tokenizer.sep_token = '~'
    input_tokenizer.sep_token_id = input_tokenizer.encode('~')[-1]

    # Add numeric tokens from 10 to 11881 for document IDs
    for num in range(10, 11882):
        input_tokenizer.add_tokens(TransformersAddedToken(str(num), normalized=False), special_tokens=True)

    # Add hyphen token for document ID formatting
    input_tokenizer.add_tokens(TransformersAddedToken('-', normalized=False), special_tokens=True)

    print(f"Input tokenizer vocabulary size: {len(input_tokenizer)}")
    #DEBUG
    #test_nums = [10, 100, 1000, 11881]
    #for num in test_nums:
    #    print(f"Token {num} -> ID: {input_tokenizer.convert_tokens_to_ids(str(num))}")

    return input_tokenizer

def create_output_tokenizer(vocab_size: int = 11886,
                          special_tokens: Optional[List[str]] = None) -> PreTrainedTokenizerFast:
    """
    Create a tokenizer for output text.

    Args:
        vocab_size: Size of the vocabulary
        special_tokens: List of special tokens to add (optional)

    Returns:
        HuggingFace tokenizer for output text
    """
    if special_tokens is None:
        special_tokens = ["<s>", "</s>", "<|>", "<unk>"]

    # Create a vocabulary for numbers that includes <unk> at index 0
    vocabulary = {"<unk>": 0}
    # Add the numbers to the vocabulary starting from index 1
    for i in range(1, vocab_size - len(special_tokens) + 1):
        vocabulary[str(i)] = i

    # Initialize the tokenizer with the vocabulary that already includes <unk>
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    # Use Split pre-tokenizer to handle hyphens in document IDs
    tokenizer.pre_tokenizer = Split(pattern='-', behavior="removed")

    # Add the remaining special tokens (excluding <unk> which is already in the vocabulary)
    for token in [t for t in special_tokens if t != "<unk>"]:
        tokenizer.add_special_tokens([token])

    # Wrap with HuggingFace fast tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<|>",
        clean_up_tokenization_spaces=True
    )

    #DEBUG
    print(f"Final tokenizer created with vocab size: {len(fast_tokenizer)}")
    print(f"Special tokens: {fast_tokenizer.all_special_tokens}")
    print(f"Special token IDs: {fast_tokenizer.all_special_ids}")

    return fast_tokenizer

def text_to_token_ids(text: str, tokenizer: PreTrainedTokenizerFast, max_length: int) -> torch.Tensor:
    """
    Convert text to token IDs.

    Args:
        text: Input text
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Tensor of token IDs
    """
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return encoded["input_ids"]

def token_ids_to_text(token_ids: torch.Tensor, tokenizer: PreTrainedTokenizerFast,
                      skip_special_tokens: bool = True) -> str:
    """
    Convert token IDs back to text.

    Args:
        token_ids: Tensor of token IDs
        tokenizer: Tokenizer to use
        skip_special_tokens: Whether to skip special tokens in the output

    Returns:
        Decoded text
    """
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

class PatchEmbeddings(nn.Module):
    def __init__(self, img_size=96, patch_size=16, hidden_dim=512):
        super().__init__()

        # Store the input image size
        self.img_size = img_size

        # Store the size of each patch
        self.patch_size = patch_size

        # Calculate the total number of patches
        self.num_patches = (img_size // patch_size) ** 2

        # Create a convolutional layer to extract patch embeddings
        # in_channels=3 assumes the input image has 3 color channels (RGB)
        # out_channels=hidden_dim sets the number of output channels to match the hidden dimension
        # kernel_size=patch_size and stride=patch_size ensure each patch is separately embedded
        self.conv = nn.Conv2d(in_channels=3, out_channels=hidden_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        # Extract patch embeddings from the input image
        X = self.conv(X)

        # Flatten the spatial dimensions (height and width) of the patch embeddings
        # This step flattens the patch dimensions into a single dimension
        X = X.flatten(2)

        # Transpose the dimensions to obtain the shape [batch_size, num_patches, hidden_dim]
        # This step brings the num_patches dimension to the second position
        X = X.transpose(1, 2)

        return X
#swapping linear for lazy linear for simplicity. Lazylinear can accept any arbitrary input dimension without having it specified

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1, is_decoder=True):
        super().__init__()

        # Define the layers of the MLP
        layers = [
            # First linear layer that expands the input dimension from n_embd to 4 * n_embd
            nn.Linear(n_embd, 4 * n_embd),

            # Activation function: ReLU if is_decoder is True, else GELU
            nn.ReLU() if is_decoder else nn.GELU(),

            # Second linear layer that projects the intermediate dimension back to n_embd
            nn.Linear(4 * n_embd, n_embd),

            # Dropout layer for regularization
            nn.Dropout(dropout)
        ]

        # Create a sequential container to hold the layers
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the MLP layers
        return self.net(x)
class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout=0.1, is_decoder=False):
        super().__init__()

        # Linear layer for key projection
        self.key = nn.Linear(n_embd, head_size, bias=False)

        # Linear layer for query projection
        self.query = nn.Linear(n_embd, head_size, bias=False)

        # Linear layer for value projection
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Flag indicating whether this head is used in the decoder
        self.is_decoder = is_decoder

    def forward(self, x):
        # Get the batch size (B), sequence length (T), and embedding dimension (C) from the input tensor
        B, T, C = x.shape

        # Compute key, query, and value projections
        k = self.key(x)   # Shape: [B, T, head_size]
        q = self.query(x) # Shape: [B, T, head_size]
        v = self.value(x) # Shape: [B, T, head_size]

        # Compute attention scores by taking the dot product of query and key
        # and scaling by the square root of the embedding dimension
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # Shape: [B, T, T]

        if self.is_decoder:
            # If this head is used in the decoder, apply a causal mask to the attention scores
            # to prevent attending to future positions
            tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            wei = wei.masked_fill(tril == 0, float('-inf'))

        # Apply softmax to the attention scores to obtain attention probabilities
        wei = F.softmax(wei, dim=-1) # Shape: [B, T, T]

        # Apply dropout to the attention probabilities for regularization
        wei = self.dropout(wei)

        # Perform weighted aggregation of values using the attention probabilities
        out = wei @ v # Shape: [B, T, head_size]

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()

        # Ensure that the embedding dimension is divisible by the number of heads
        assert n_embd % num_heads == 0, "n_embd must be divisible by num_heads"

        # Create a ModuleList of attention heads
        self.heads = nn.ModuleList([
            Head(n_embd, n_embd // num_heads, dropout, is_decoder)
            for _ in range(num_heads)
        ])

        # Linear layer for projecting the concatenated head outputs
        self.proj = nn.Linear(n_embd, n_embd)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply each attention head to the input tensor
        head_outputs = [h(x) for h in self.heads]

        # Concatenate the outputs from all heads along the last dimension
        out = torch.cat(head_outputs, dim=-1)

        # Apply the projection layer to the concatenated outputs
        out = self.proj(out)

        # Apply dropout to the projected outputs for regularization
        out = self.dropout(out)

        return out

class Block(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()

        # Layer normalization for the input to the attention layer
        self.ln1 = nn.LayerNorm(n_embd)

        # Multi-head attention module
        self.attn = MultiHeadAttention(n_embd, num_heads, dropout, is_decoder)

        # Layer normalization for the input to the FFN
        self.ln2 = nn.LayerNorm(n_embd)

        # Feed-forward neural network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand the dimension
            nn.GELU(),  # Activation function
            nn.Linear(4 * n_embd, n_embd),  # Project back to the original dimension
        )

    def forward(self, x):
        original_x = x  # Save the input for the residual connection

        # Apply layer normalization to the input
        x = self.ln1(x)

        # Apply multi-head attention
        attn_output = self.attn(x)

        # Add the residual connection (original input) to the attention output
        x = original_x + attn_output

        # Apply layer normalization to the input to the FFN
        x = self.ln2(x)

        # Apply the FFN
        ffn_output = self.ffn(x)

        # Add the residual connection (input to FFN) to the FFN output
        x = x + ffn_output

        return x
class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout):
        super().__init__()

        # Patch embedding layer to convert the input image into patches
        self.patch_embedding = PatchEmbeddings(img_size, patch_size, num_hiddens)

        # Learnable classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))

        # Calculate the number of patches
        num_patches = (img_size // patch_size) ** 2

        # Learnable position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, num_hiddens))

        # Dropout layer for the embeddings
        self.dropout = nn.Dropout(emb_dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block(num_hiddens, num_heads, blk_dropout, is_decoder=False) for _ in range(num_blks)])

        # Layer normalization for the final representation
        self.layer_norm = nn.LayerNorm(num_hiddens)

    def forward(self, X):
        # Convert the input image into patch embeddings
        x = self.patch_embedding(X)

        # Expand the classification token to match the batch size
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        # Concatenate the classification token with the patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)

        # Add the position embedding to the patch embeddings
        x += self.pos_embedding

        # Apply dropout to the embeddings
        x = self.dropout(x)

        # Pass the embeddings through the transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply layer normalization to the final representation
        x = self.layer_norm(x[:, 0])

        return x
class MultiModalProjector(nn.Module):
    def __init__(self, n_embd, image_embed_dim, dropout=0.1):
        super().__init__()

        # Define the projection network
        self.net = nn.Sequential(
            # Linear layer to expand the image embedding dimension
            nn.Linear(image_embed_dim, 4 * image_embed_dim),

            # GELU activation function
            nn.GELU(),

            # Linear layer to project the expanded image embeddings to the text embedding dimension
            nn.Linear(4 * image_embed_dim, n_embd),

            # Dropout layer for regularization
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pass the input through the projection network
        x = self.net(x)
        return x

class DecoderLanguageModel(nn.Module):
    def __init__(self, n_embd, image_embed_dim, vocab_size_input,vocab_size_output, num_heads, n_layer, use_images=False):
        super().__init__()

        self.use_images = use_images

        # Token embedding table
        self.token_embedding_table = nn.Embedding(vocab_size_input, n_embd)

        # Position embedding table
        self.position_embedding_table = nn.Embedding(1000, n_embd)

        if use_images:
            # Image projection layer to align image embeddings with text embeddings
            self.image_projection = MultiModalProjector(n_embd, image_embed_dim)

        # Stack of transformer decoder blocks
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads, is_decoder=True) for _ in range(n_layer)])

        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)

        # Language modeling head
        self.lm_head = nn.Linear(n_embd, vocab_size_output)

    def forward(self, idx, image_embeds=None, targets=None):
        # Get token embeddings from the input indices
        tok_emb = self.token_embedding_table(idx)

        if self.use_images and image_embeds is not None:
            # Project and concatenate image embeddings with token embeddings
            img_emb = self.image_projection(image_embeds).unsqueeze(1)
            tok_emb = torch.cat([img_emb, tok_emb], dim=1)

        # Get position embeddings
        pos_emb = self.position_embedding_table(torch.arange(tok_emb.size(1), device=device)).unsqueeze(0)

        # Add position embeddings to token embeddings
        x = tok_emb + pos_emb

        # Pass through the transformer decoder blocks
        x = self.blocks(x)

        # Apply final layer normalization
        x = self.ln_f(x)

        # Get the logits from the language modeling head
        logits = self.lm_head(x)

        if targets is not None:
            if self.use_images and image_embeds is not None:
                # Prepare targets by concatenating a dummy target for the image embedding
                batch_size = idx.size(0)
                #targets = torch.cat([torch.full((batch_size, 1), input_tokenizer.pad_token, dtype=torch.long, device=device), targets], dim=1)
                targets = torch.cat([torch.full((batch_size, 1), input_tokenizer.pad_token_id, dtype=torch.long, device=device), targets], dim=1)
            # Compute the cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=input_tokenizer.pad_token_id)
            return logits, loss

        return logits

    def generate(self, idx, image_embeds, max_new_tokens):
        # Get the batch size and sequence length
        B, T = idx.shape

        # Initialize the generated sequence with the input indices
        generated = idx

        if self.use_images and image_embeds is not None:
            # Project and concatenate image embeddings with token embeddings
            img_emb = self.image_projection(image_embeds).unsqueeze(1)
            current_output = torch.cat([img_emb, self.token_embedding_table(idx)], dim=1)
        else:
            current_output = self.token_embedding_table(idx)

        # Generate new tokens iteratively
        for i in range(max_new_tokens):
            # Get the current sequence length
            T_current = current_output.size(1)

            # Get position embeddings for the current sequence length
            current_pos_emb = self.position_embedding_table(torch.arange(T_current, device=device)).unsqueeze(0)

            # Add position embeddings to the current output
            current_output += current_pos_emb

            # Pass through the transformer decoder blocks
            for block in self.blocks:
                current_output = block(current_output)

            # Get the logits for the last token
            logits = self.lm_head(current_output[:, -1, :])

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token based on the probabilities
            idx_next = torch.multinomial(probs, num_samples=1)

             #TO DO aggiungere un modo per ricreare un input con inputtokenizer
            #print(idx_next)

            # Concatenate the generated token to the generated sequence
            generated = torch.cat((generated, idx_next), dim=1)

            # Get the embeddings for the generated token
            idx_next_emb = self.token_embedding_table(idx_next)

            # Concatenate the generated token embeddings to the current output
            current_output = torch.cat((current_output, idx_next_emb), dim=1)

        return generated

class DecoderLanguageModel2(DecoderLanguageModel):
    def __init__(self, n_embd, image_embed_dim, num_heads, n_layer, num_classes, use_images=False):
        super().__init__(n_embd, image_embed_dim, vocab_size_input=1,vocab_size_output=1,  # dummy vocab_size, we override lm_head
                         num_heads=num_heads, n_layer=n_layer, use_images=use_images)
        # Sostituiamo il lm_head con uno adatto alla classificazione
        self.lm_head = nn.Linear(n_embd, num_classes)

class VisionLanguageModel(nn.Module):
    def __init__(self, n_embd, image_embed_dim, vocab_size, n_layer, img_size, patch_size, num_heads, num_blks, emb_dropout, blk_dropout):
        super().__init__()

        # Set num_hiddens equal to image_embed_dim
        num_hiddens = image_embed_dim

        # Assert that num_hiddens is divisible by num_heads
        assert num_hiddens % num_heads == 0, "num_hiddens must be divisible by num_heads"

        # Initialize the vision encoder (ViT)
        self.vision_encoder = ViT(img_size, patch_size, num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout)

        # Initialize the language model decoder (DecoderLanguageModel)
        self.decoder = DecoderLanguageModel(n_embd, image_embed_dim, vocab_size,vocab_size, num_heads, n_layer, use_images=True)

    def forward(self, img_array, idx, targets=None):
        # Get the image embeddings from the vision encoder
        image_embeds = self.vision_encoder(img_array)

        # Check if the image embeddings are valid
        if image_embeds.nelement() == 0 or image_embeds.shape[1] == 0:
            raise ValueError("Something is wrong with the ViT model. It's returning an empty tensor or the embedding dimension is empty.")

        if targets is not None:
            # If targets are provided, compute the logits and loss
            logits, loss = self.decoder(idx, image_embeds, targets)
            return logits, loss
        else:
            # If targets are not provided, compute only the logits
            logits = self.decoder(idx, image_embeds)
            return logits

    def generate(self, img_array, idx, max_new_tokens):
        # Get the image embeddings from the vision encoder
        image_embeds = self.vision_encoder(img_array)

        # Check if the image embeddings are valid
        if image_embeds.nelement() == 0 or image_embeds.shape[1] == 0:
            raise ValueError("Something is wrong with the ViT model. It's returning an empty tensor or the embedding dimension is empty.")

        # Generate new tokens using the language model decoder
        generated_tokens = self.decoder.generate(idx, image_embeds, max_new_tokens)
        return generated_tokens
class VisionLanguageModel2(VisionLanguageModel):
    def __init__(self, n_embd, image_embed_dim, vocab_size_input,vocab_size_output, n_layer, img_size, patch_size, num_heads, num_blks, emb_dropout, blk_dropout):
        super().__init__(n_embd, image_embed_dim, vocab_size_input, n_layer, img_size, patch_size, num_heads, num_blks, emb_dropout, blk_dropout)
        # Sostituiamo il lm_head con uno adatto alla classificazione
        self.decoder = DecoderLanguageModel(n_embd, image_embed_dim, vocab_size_input, vocab_size_output, num_heads, n_layer, use_images=True)

def base64_to_tensor(base64_str, img_size=96):
    image = Image.open(io.BytesIO(base64.b64decode(base64_str)))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def inverti_numeri(s):
    numeri = s.strip().split()
    numeri_invertiti = numeri[::-1]
    return '-'.join(numeri_invertiti)


def get_batch(df, batch_size, split='train', img_size=96, val_batch_size=8):
    # Suddivisione del dataset
    n = int(0.9 * len(df))
    df_train = df.iloc[:n]
    df_val = df.iloc[n:]
    data = df_train if split == 'train' else df_val
    batch_size = batch_size if split == 'train' else val_batch_size
    replace = False if split == 'train' else True
    batch = data.sample(n=batch_size, replace=replace)

    # Image transformations


    image_tensors = []
    text_inputs = []
    text_targets = []

    for _, row in batch.iterrows():
        path_str = row['report_id']  # es. "p10.p10002430.s53254222"
        icd_code = row['mapped_hierarchy']

        imgs = load_three_images_from_path(path_str)

        # Trasforma le immagini in grayscale, ridimensiona e normalizza (1 canale)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizzazione per 1 canale
        ])

        # Applica la trasformazione e rimuove la dimensione del canale (da [1, H, W] a [H, W])
        img_tensors = [transform(img).squeeze(0) for img in imgs]  # squeeze(0) rimuove il canale

        # Stacka in un unico tensore [3, H, W]
        img_tensor = torch.stack(img_tensors)

        # Aggiungi al batch
        image_tensors.append(img_tensor)



        raw_text =row["report_keywords"]
        if raw_text is None:
            continue

        # Concatenazione: [text + ICD code] come input
        full_text = raw_text + input_tokenizer.sep_token + inverti_numeri(icd_code)

        encod=input_tokenizer.encode(full_text)
        temp=input_tokenizer.tokenize(full_text)
        encoded = torch.tensor(encod, dtype=torch.long)

        text_inputs.append(encoded)
        #print("Tokenize:",temp)
        #print(encod)
        text_targets.append(torch.cat([encoded[1:], torch.tensor([input_tokenizer.pad_token_id], dtype=torch.long)]))

    # Padding
    max_len = max(len(seq) for seq in text_inputs)
    padded_inputs = torch.full((batch_size, max_len), fill_value=input_tokenizer.pad_token_id, dtype=torch.long)
    padded_targets = torch.full((batch_size, max_len), fill_value=input_tokenizer.pad_token_id, dtype=torch.long)

    for i in range(len(text_inputs)):
        padded_inputs[i, :len(text_inputs[i])] = text_inputs[i]
        padded_targets[i, :len(text_targets[i])] = text_targets[i]

    image_tensor_batch = torch.stack(image_tensors).to(device)
    padded_inputs = padded_inputs.to(device)
    padded_targets = padded_targets.to(device)

    return image_tensor_batch, padded_inputs, padded_targets

def get_batch_test(df, batch_size, split='train', img_size=96, val_batch_size=1):
    # Split data into training and validation sets
    n = int(0.9 * len(df))  # first 90% will be train, rest val
    df_train = df.iloc[:n]
    df_val = df.iloc[n:]
    data = df_train if split == 'train' else df_val
    batch_size = batch_size if split == 'train' else val_batch_size
    replace = False if split == 'train' else True
    batch = data.sample(n=batch_size, replace=replace)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),

    ])

    image_tensors = []
    text_inputs = []
    text_targets = []

    for _, row in batch.iterrows():
        path_str = row['report_id']  # es. "p10.p10002430.s53254222"
        icd_code = row['mapped_hierarchy']

        # Carica immagine e testo
        imgs = load_three_images_from_path(path_str)

        # Trasforma le immagini in grayscale, ridimensiona e normalizza (1 canale)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizzazione per 1 canale
        ])

        # Applica la trasformazione e rimuove la dimensione del canale (da [1, H, W] a [H, W])
        img_tensors = [transform(img).squeeze(0) for img in imgs]  # squeeze(0) rimuove il canale

        # Stacka in un unico tensore [3, H, W]
        img_tensor = torch.stack(img_tensors)

        # Aggiungi al batch
        image_tensors.append(img_tensor)


        raw_text = row["report_keywords"]
        if raw_text is None:
            continue

        # Concatenazione: [text + ICD code] come input
        full_text = raw_text.strip() +input_tokenizer.sep_token 
        answer=inverti_numeri(icd_code)

        encoded = torch.tensor(input_tokenizer.encode(full_text), dtype=torch.long)
        encoded_answer = torch.tensor(input_tokenizer.encode(answer), dtype=torch.long)
        #print("Tokenize:",input_tokenizer.tokenize(answer))
        print(encoded_answer)
        text_inputs.append(encoded)
        text_targets.append(torch.cat([encoded_answer, torch.tensor([input_tokenizer.pad_token_id], dtype=torch.long)]))


    image_tensor_batch = torch.stack(image_tensors).to(device)

    max_length = max(len(t) for t in text_inputs)
    padded_text = torch.full((batch_size, max_length), fill_value=input_tokenizer.pad_token_id, dtype=torch.long).to(device)
    for i, text in enumerate(text_inputs):
        padded_text[i, :len(text)] = text
    max_length = max(len(t) for t in text_targets)
    answare = torch.full((batch_size, max_length), fill_value=input_tokenizer.pad_token_id, dtype=torch.long).to(device)
    for i, text in enumerate(text_targets):
        answare[i, :len(text)] = text

    return image_tensor_batch, padded_text, answare

#Adjusting the training loop from makemore for multimodal data
def train_model(model, df, epochs, vocab_size,learning_rate,max_iters,batch_size,eval_interval,eval_iters, img_size=96,):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #model.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        for _ in range(max_iters):

            images, idx, targets = get_batch(df, batch_size, 'train', img_size)
            optimizer.zero_grad()
            logits, loss = model(images, idx, targets)
            loss.backward()
            optimizer.step()
            if _ % eval_interval == 0:
                print(f"Loss at iteration {_}: {loss.item()}")

        val_loss = estimate_loss(model, df, 'val',eval_iters,batch_size, img_size, val_batch_size=8)
        wandb.log({
            "train/loss": loss,
            
            "validate/loss": val_loss,
            
        }, step=epoch)

        print(f"Validation Loss after epoch {epoch}: {val_loss}")

def estimate_loss(model, df, split,eval_iters,batch_size, img_size=96, val_batch_size=8):
    losses = []
    print(f"Estimating loss for {split} set...")
    model.eval()
    for _ in range(eval_iters):
        images, idx, targets = get_batch(df, batch_size, split, img_size, val_batch_size=val_batch_size)
        _, loss = model(images, idx, targets)
        losses.append(loss.item())
    return sum(losses) / len(losses)

def _safe_generate_old(model, img, input_ids, max_new_tokens):
    try:
        return model.generate(input_ids, img, max_new_tokens)
    except Exception:
        try:
            return model.generate(img, input_ids, max_new_tokens)
        except Exception as e:
            raise RuntimeError(f"model.generate signature unknown: {e}")
def _safe_generate2_notWork(model, img, input_ids, max_new_tokens):
    current_input=input_ids

    try:
        for i in range(max_new_tokens):
            resp = model.generate(img, current_input, 1)
            current_input = current_input+ text_to_token_ids(token_ids_to_text(resp[:,-1],output_tokenizer),input_tokenizer)
        return resp
    except Exception as e:
        raise RuntimeError(f"model.generate signature unknown: {e}")
    
def _safe_generate2(model, img, input_ids, max_new_tokens):
    """
    Iterative generation compatible with two tokenizers:
    - model.decoder.generate (preferred) or model.generate(...) is called to produce tokens IN OUTPUT space.
    - Each generated output-token (output_tokenizer id) is converted to text and then re-tokenized with input_tokenizer
      to produce input-token ids which are appended to current_input (so next step has updated context).
    Returns:
        tensor shape (B, max_new_tokens) with generated token ids in OUTPUT token ids space.
    """
    device = input_ids.device if isinstance(input_ids, torch.Tensor) else ('cuda' if torch.cuda.is_available() else 'cpu')
    current_input = input_ids.clone().to(device)  # (B, L_ctx)
    B = current_input.size(0)

    # output buffer: store generated token ids in OUTPUT tokenizer space
    pad_out_id = getattr(output_tokenizer, 'pad_token_id', None)
    filler_out = pad_out_id if pad_out_id is not None else 0
    generated_out = torch.full((B, max_new_tokens), fill_value=filler_out, dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        resp = None
        # Prefer decoder.generate if available (it should return ids in OUTPUT vocab)
        try:
            if hasattr(model, "decoder") and hasattr(model.decoder, "generate"):
                # if model has vision_encoder, produce image embeddings once (decoder might expect them)
                image_embeds = None
                try:
                    if hasattr(model, "vision_encoder") and img is not None:
                        image_embeds = model.vision_encoder(img.to(device))
                except Exception:
                    image_embeds = None
                # decoder.generate expects (idx_input, image_embeds, max_new_tokens)
                resp = model.decoder.generate(current_input, image_embeds, 1)
        except Exception:
            resp = None

        # Fallback to model.generate with two possible signatures
        if resp is None:
            try:
                resp = model.generate(current_input, img.to(device), 1)
            except Exception:
                try:
                    resp = model.generate(img.to(device), current_input, 1)
                except Exception as e:
                    raise RuntimeError(f"_safe_generate2: model.generate failed for known signatures. Error: {e}")

        # Normalize resp to tensor on device
        if not isinstance(resp, torch.Tensor):
            resp = torch.tensor(resp, dtype=torch.long, device=device)
        else:
            resp = resp.to(device)

        # resp expected shape: (B, L_resp) where last token is the newly generated token in OUTPUT vocab
        if resp.dim() == 1:
            # degenerate shape -> (B,) convert to (B,1)
            resp = resp.unsqueeze(1)

        last_ids = resp[:, -1].long()  # (B,) output-token ids
        generated_out[:, step] = last_ids

        # Convert each output-id to text (OUTPUT tokenizer) and re-tokenize with input_tokenizer
        last_ids_cpu = last_ids.cpu().tolist()  # list length B
        texts = []
        for tok_id in last_ids_cpu:
            # decode a single token id into text
            try:
                txt = output_tokenizer.decode([int(tok_id)], skip_special_tokens=True).strip()
            except Exception:
                # fallback: simple str numeric
                txt = str(int(tok_id))
            texts.append(txt)

        # tokenize these texts with input_tokenizer (could be variable-length per example)
        encoded_lists = []
        for txt in texts:
            if txt is None or txt == "":
                encoded_lists.append([])  # nothing to append
            else:
                # NOTE: use add_special_tokens=False to avoid BOS/EOS insertion
                enc = input_tokenizer.encode(txt, add_special_tokens=False)
                # handle different return types from different tokenizers
                enc_ids = enc.ids if hasattr(enc, "ids") else (enc["input_ids"] if isinstance(enc, dict) and "input_ids" in enc else list(enc))
                encoded_lists.append([int(x) for x in enc_ids])

        # If no new token for any batch element, continue
        max_new_len = 0 if len(encoded_lists)==0 else max((len(x) for x in encoded_lists), default=0)
        if max_new_len == 0:
            # nothing to append as input tokens
            continue

        # Build a tensor (B, max_new_len) padded with input_tokenizer.pad_token_id
        pad_in_id = getattr(input_tokenizer, 'pad_token_id', None)
        if pad_in_id is None:
            # fallback: use 0
            pad_in_id = 0
        new_ids = torch.full((B, max_new_len), fill_value=pad_in_id, dtype=torch.long, device=device)
        for b_idx, lst in enumerate(encoded_lists):
            if len(lst) > 0:
                new_ids[b_idx, :len(lst)] = torch.tensor(lst, dtype=torch.long, device=device)

        # Append to current_input (concatenate along sequence dimension)
        current_input = torch.cat([current_input, new_ids], dim=1)

    # generated_out: (B, max_new_tokens) in OUTPUT id space
    return generated_out



def _decode_output_ids(ids_list, output_tokenizer):
    """Decodifica una lista di id in stringa. Fallback a join numerico."""
    try:
        return output_tokenizer.decode(ids_list, skip_special_tokens=True).strip()
    except Exception:
        return " ".join(map(str, ids_list))

def evaluate_per_N(
    model,
    df,
    get_batch_fn,
    input_tokenizer,
    output_tokenizer,
    split="val",
    img_size=96,
    num_batches=10,
    batch_size=20,
    num_return_sequences=10,
    max_N=5,
    device="cuda",
    k_values=[1,2,3,4,5],
    require_full_gt=True
):
    """
    Evaluate the model separately for N=1..max_N codes.
    - require_full_gt=True: to calculate metrics for N, consider only examples with at least N ground truth tokens.
    Returns: dict with keys for each N containing metrics and q_gt/q_pred details for debugging.
    """
    model.eval()
    model.to(device)

    # Containers: for each N a dict of q_gt and q_pred
    results = {}
    # Prepare global structures for debugging
    debug_q_gt = defaultdict(set)
    debug_q_pred = defaultdict(list)

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc=f"Evaluating on {split}"):
            # corretto: passare batch_size, non batch_idx
            img, input_ids, output_ids = get_batch_fn(df, batch_size, split, img_size, val_batch_size=batch_size)
            img = img.to(device)
            input_ids = input_ids.to(device)
            output_ids = output_ids.to(device)
            B = input_ids.size(0)

            # Extract ground truth sequence for each row (valid ids != -100)
            gt_id_rows = []
            for i in range(B):
                row = output_ids[i].cpu().tolist()
                valid = [int(x) for x in row if int(x) != input_tokenizer.pad_token_id ]
                gt_id_rows.append(valid)

            # genera num_return_sequences volte (sampling)
            for r in range(num_return_sequences):
                preds_ids = _safe_generate_old(model, img, input_ids, max_new_tokens=max_N)
                if isinstance(preds_ids, list):
                    # converti a lista di list se necessario
                    preds_ids = torch.tensor(preds_ids, device='cpu')
                preds_ids = preds_ids.cpu().tolist()  # lista di liste length B

                # proviamo a capire se generate ha incluso il prefisso (confronto col primo es.)
                prefix_len = input_ids.size(1)
                uses_prefix = False
                try:
                    first_pred_prefix = preds_ids[0][:prefix_len]
                    if first_pred_prefix == input_ids[0].cpu().tolist():
                        uses_prefix = True
                    else:
                        # Try with shift of 1 (image token at the beginning)
                        if len(preds_ids[0]) >= prefix_len + 1 and preds_ids[0][1:prefix_len+1] == input_ids[0].cpu().tolist():
                            uses_prefix = True
                            prefix_len += 1
                except Exception:
                    uses_prefix = False

                # For each example, get the generated part (without prefix), take first max_N tokens
                for i in range(B):
                    seq = preds_ids[i]
                    gen_part = seq[prefix_len:] if uses_prefix else seq[:]
                    # Remove any output tokenizer padding
                    pad_out_id = getattr(output_tokenizer, 'pad_token_id', None)
                    if pad_out_id is not None:
                        gen_part = [t for t in gen_part if t != pad_out_id]
                    # Accumulate up to max_N tokens
                    gen_part = gen_part[:max_N]
                    # Save for debugging
                    qid = f"b{batch_idx}_r{r}_i{i}"
                    pred_str_full = _decode_output_ids(gen_part, output_tokenizer)
                    debug_q_pred[qid].append(pred_str_full)
                    # Save ground truth for debugging
                    gt_ids = gt_id_rows[i]
                    gt_str_full = _decode_output_ids(gt_ids, output_tokenizer)
                    debug_q_gt[qid].add(gt_str_full)

            # end sampling loop
        # end batches

    # Now we have debug_q_gt, debug_q_pred populated with multiple keys (for simplicity we use mapping for calculations):
    # It's simpler to reconstruct a dataset of examples (first pass, better to have structure per example)
    # We recreate an iterator on df with the same sampling used above for consistency: for simplicity, we call get_batch_fn again
    # NOTE: for ideal accuracy we should accumulate preds/gt for each batch; here we assume debug_q_gt/debug_q_pred contain correspondences.

    # --- Simpler and more robust approach: regenerate on num_batches and save for each example: gt_ids and all predictions (for r=0..R-1)
    per_example_gt = []
    per_example_preds = []  # list of lists of pred sequences (strings) per example
    # Resample to build comparison dataset
    with torch.no_grad():
        for batch_idx in range(num_batches):
            img, input_ids, output_ids = get_batch_fn(df, batch_size, split, img_size, val_batch_size=batch_size)
            img = img.to(device); input_ids = input_ids.to(device); output_ids = output_ids.to(device)
            B = input_ids.size(0)
            # gt
            for i in range(B):
                row = output_ids[i].cpu().tolist()
                valid = [int(x) for x in row if int(x) != input_tokenizer.pad_token_id]
                per_example_gt.append(valid)
            # generate multiple times & store preds
            # list of lists-of-lists: for each example index append list of predictions (each pred is list of ids)
            block_preds = [[] for _ in range(B)]
            for r in range(num_return_sequences):
                preds_ids = _safe_generate_old(model, img, input_ids, max_new_tokens=max_N)
                if isinstance(preds_ids, list):
                    preds_ids = torch.tensor(preds_ids, device='cpu')
                preds_ids = preds_ids.cpu().tolist()
                # infer prefix
                prefix_len = input_ids.size(1)
                uses_prefix = False
                try:
                    first_pred_prefix = preds_ids[0][:prefix_len]
                    if first_pred_prefix == input_ids[0].cpu().tolist():
                        uses_prefix = True
                    else:
                        if len(preds_ids[0]) >= prefix_len + 1 and preds_ids[0][1:prefix_len+1] == input_ids[0].cpu().tolist():
                            uses_prefix = True
                            prefix_len += 1
                except Exception:
                    uses_prefix = False
                for i in range(B):
                    seq = preds_ids[i]
                    gen_part = seq[prefix_len:] if uses_prefix else seq[:]
                    pad_out_id = getattr(output_tokenizer, 'pad_token_id', None)
                    if pad_out_id is not None:
                        gen_part = [t for t in gen_part if t != pad_out_id]
                    gen_part = gen_part[:max_N]
                    block_preds[i].append(gen_part)
            # extend per_example_preds
            per_example_preds.extend(block_preds)

    # Now we have per_example_gt (list length num_batches*B) and per_example_preds (same length) 
    total_examples = len(per_example_gt)
    # Initialize results dict for each N
    for N in range(1, max_N+1):
        metrics = {}
        # Collect queries that have at least N tokens (if require_full_gt)
        valid_indices = [i for i, g in enumerate(per_example_gt) if (len(g) >= N) or (not require_full_gt)]
        if len(valid_indices) == 0:
            # Nothing to calculate
            results[N] = {"metrics": None, "num_examples": 0}
            continue

        # Build sets for ground truth N
        q_gt = {}
        q_pred = {}
        for idx in valid_indices:
            gt_ids = per_example_gt[idx][:N]  # First N ground truth tokens (can be shorter if < N and require_full_gt False)
            gt_str = _decode_output_ids(gt_ids, output_tokenizer)
            q_key = f"ex_{idx}"
            q_gt[q_key] = {gt_str} if gt_str != "" else set()
            # preds: per example we have num_return_sequences lists of ids; convert each to first N tokens and decode
            preds_lists = per_example_preds[idx]  # List of length num_return_sequences
            pred_strs = []
            for p in preds_lists:
                pred_ids_N = p[:N]
                pred_str = _decode_output_ids(pred_ids_N, output_tokenizer)
                if pred_str not in pred_strs:
                    pred_strs.append(pred_str)
            q_pred[q_key] = pred_strs

        # metrics computation (accuracy@k, recall@k, mrr)
        n_q = len(q_gt)
        for k in k_values:
            hits = 0
            recalls = []
            for qkey in q_gt:
                gtset = q_gt[qkey]
                preds_k = q_pred.get(qkey, [])[:k]
                if len(gtset) == 0:
                    recalls.append(0.0)
                    continue
                if any(p in gtset for p in preds_k):
                    hits += 1
                recalls.append(len(set(preds_k) & gtset) / len(gtset))
            metrics[f"accuracy@{k}"] = hits / n_q
            metrics[f"recall@{k}"] = float(np.mean(recalls))

        # MRR
        mrr = 0.0
        for qkey in q_gt:
            gtset = q_gt[qkey]
            preds_list = q_pred.get(qkey, [])
            found = False
            for rank, p in enumerate(preds_list, 1):
                if p in gtset:
                    mrr += 1.0 / rank
                    found = True
                    break
            if not found:
                mrr += 0.0
        metrics["mrr"] = mrr / n_q

        results[N] = {"metrics": metrics, "num_examples": n_q, "q_gt_sample": dict(list(q_gt.items())[:5]), "q_pred_sample": dict(list(q_pred.items())[:5])}

    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train vision-language model from scratch')
    parser.add_argument('--data_path', type=str, default='/workspace/report_image_test.jsonl',
                        help='Path to training data JSONL file')
    parser.add_argument('--image_base_path', type=str, default='/datasets/MIMIC-CXR/files',
                        help='Base directory for images')
    parser.add_argument('--wandb_project', type=str, default='gen-icpv-from-scratch',
                        help='WandB project name')
    parser.add_argument('--wandb_key', type=str, default=None,
                        help='WandB API key')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='HuggingFace access token')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--max_iters', type=int, default=100,
                        help='Maximum iterations per epoch')
    parser.add_argument('--n_embd', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--n_layer', type=int, default=24,
                        help='Number of transformer layers')
    parser.add_argument('--img_size', type=int, default=96,
                        help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size for ViT')

    args = parser.parse_args()

    global input_tokenizer, output_tokenizer, device

    # Login to WandB if key is provided
    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    print(f"Device: {device}")

    # Load training data from JSONL file
    with open(args.data_path, 'r') as f:
        records = [json.loads(line) for line in f]

    # Create DataFrame
    df0 = pd.DataFrame(records)

    # Filter valid entries
    df = filter_valid_entries(df0)
    print(f"Valid entries with accessible images: {len(df)}")
    df = df.sample(frac=1).reset_index(drop=True)

    # Login to HuggingFace if token is provided
    if args.hf_token:
        login(token=args.hf_token)

    # Create tokenizers
    print("Creating tokenizers...")
    input_tokenizer = create_input_tokenizer(hf_access_token=args.hf_token)
    output_tokenizer = create_output_tokenizer()

    vocab_size = len(input_tokenizer)
    vocab_size2 = len(output_tokenizer)

    # Training hyperparameters
    batch_size = args.batch_size
    block_size = 32  # Maximum context length for predictions
    max_iters = args.max_iters
    eval_interval = 10
    learning_rate = args.learning_rate
    epochs = args.epochs
    eval_iters = 40
    num_blks = 3
    head_size = 16
    n_embd = args.n_embd
    n_head = args.n_head
    n_layer = args.n_layer
    dropout = 0.2
    img_size = args.img_size
    patch_size = args.patch_size
    image_embed_dim = 1024
    emb_dropout = blk_dropout = 0.2

    # Initialize WandB
    print("Initializing WandB...")
    run_name = f"model_embd{n_embd}_head{n_head}_layer{n_layer}_epochs{epochs}"
    wandb.init(
        project=args.wandb_project,
        config={
            "n_embed": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "vocab_size": vocab_size,
            "patch_size": patch_size,
            "lr": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_blks": num_blks,
            "emb_dropout": emb_dropout,
            "dropout": dropout,
            "blk_dropout": blk_dropout
        },
        name=run_name
    )
    print(get_batch(df, batch_size, 'train', img_size))


    # Initialize the model
    model = VisionLanguageModel(n_embd, image_embed_dim, vocab_size, n_layer, img_size, patch_size, n_head, num_blks, emb_dropout, blk_dropout)
    model.to(device)    

    


    # Dummy data to initialize lazy modules
    dummy_img = torch.randn(1, 3, img_size, img_size).to(device)
    dummy_idx = torch.randint(0, vocab_size, (1, block_size)).to(device)
    model(dummy_img, dummy_idx)  # Forward pass to initialize all parameters

    # Train the model
    print("Start training...")
    train_model(model, df, epochs, vocab_size,learning_rate,max_iters,batch_size,eval_interval,eval_iters, img_size)

    # Save model checkpoint
    model_save_path = os.path.join(args.output_dir, f"{run_name}.pth")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")


    # Validation inference test
    print("\n" + "="*50)
    print("Running validation inference test...")
    print("="*50)
    img, input, output = get_batch_test(df, 1, 'val', img_size)
    print(f"Input: {input_tokenizer.decode(input[0].tolist())}")
    print(f"Ground truth: {input_tokenizer.decode(output[0].tolist())}")
    print(f"Ground truth length: {len(output[0].tolist())}")

    try:
        res = model.generate(img, input, len(output[0].tolist()))
        print(f"Generated: {input_tokenizer.decode(res[0].tolist())}")
    except Exception as e:
        print(f"Runtime error during generation: {str(e)}")
        print("Check layer configurations and input shapes.")

    # Full evaluation with metrics
    print("\n" + "="*50)
    print("Running full evaluation...")
    print("="*50)
    results = evaluate_per_N(
        model=model,
        df=df,
        get_batch_fn=get_batch_test,
        input_tokenizer=input_tokenizer,
        output_tokenizer=output_tokenizer,
        split="val",
        img_size=img_size,
        num_batches=5,  # Increase for more thorough evaluation
        batch_size=8,
        num_return_sequences=5,
        max_N=5,  # Maximum number of codes to evaluate
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\nEvaluation Results:")
    for N in results:
        m = results[N]["metrics"]
        if m is not None:
            print(f"\n=== Results for N = {N} ===")
            for k, v in m.items():
                print(f"{k}: {v:.4f}")
        else:
            print(f"\n=== N = {N} === No valid examples found.")


main()