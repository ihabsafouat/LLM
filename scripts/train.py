import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
import math
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time
from tqdm import tqdm
import re
import os
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import json

# Training Configuration
# =====================
# These parameters can be easily modified in Google Colab
BATCH_SIZE = 32
GRADIENT_CLIP_VAL = 1.0
WARMUP_STEPS = 1000
SAVE_EVERY = 1000
EVAL_EVERY = 100
MAX_ITERS = 200
LEARNING_RATE = 3e-4
EVAL_ITERS = 100

# Model Configuration
# ==================
BLOCK_SIZE = 256  # Increased context window
N_EMBD = 768     # Increased embedding dimension
N_HEAD = 12      # Increased number of attention heads
N_LAYER = 6      # Increased number of transformer layers
DROPOUT = 0.1    # Slightly reduced dropout for better training
BIAS = False     # Disable bias in attention layers for better performance

# Data Processing Configuration
# ===========================
MIN_FREQ = 2  # Minimum frequency for vocabulary
MAX_VOCAB_SIZE = 50000  # Maximum vocabulary size
TRAIN_RATIO = 0.9  # Ratio of data to use for training
NUM_WORKERS = 4  # Number of workers for data loading
PIN_MEMORY = True  # Pin memory for faster data transfer to GPU

# Device Configuration
# ===================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class TextPreprocessor:
    def __init__(self):
        self.patterns = [
            (r'[^\w\s]', ' '),  # Replace punctuation with space
            (r'\s+', ' '),      # Replace multiple spaces with single space
            (r'\n+', ' '),      # Replace newlines with space
            (r'\t+', ' '),      # Replace tabs with space
        ]
        self.compiled_patterns = [(re.compile(p), r) for p, r in self.patterns]

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.lower()
        for pattern, replacement in self.compiled_patterns:
            text = pattern.sub(replacement, text)
        return text.strip()

class Vocabulary:
    def __init__(self, min_freq: int = MIN_FREQ, max_size: int = MAX_VOCAB_SIZE):
        self.min_freq = min_freq
        self.max_size = max_size
        self.char2idx = {}
        self.idx2char = {}
        self.char_freq = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3
        }

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        # Count character frequencies
        for text in texts:
            for char in text:
                self.char_freq[char] = self.char_freq.get(char, 0) + 1

        # Filter by minimum frequency
        valid_chars = {char for char, freq in self.char_freq.items() 
                      if freq >= self.min_freq}

        # Add special tokens
        for token, idx in self.special_tokens.items():
            self.char2idx[token] = idx
            self.idx2char[idx] = token

        # Add characters to vocabulary
        for idx, char in enumerate(valid_chars, start=len(self.special_tokens)):
            if len(self.char2idx) >= self.max_size:
                break
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def encode(self, text: str) -> List[int]:
        """Encode text to indices."""
        return [self.char2idx.get(char, self.special_tokens['<unk>']) for char in text]

    def decode(self, indices: List[int]) -> str:
        """Decode indices to text."""
        return ''.join(self.idx2char.get(idx, '<unk>') for idx in indices)

    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char2idx': self.char2idx,
                'idx2char': {int(k): v for k, v in self.idx2char.items()},
                'char_freq': self.char_freq
            }, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char2idx = data['char2idx']
            self.idx2char = {int(k): v for k, v in data['idx2char'].items()}
            self.char_freq = data['char_freq']

class TextDataset(Dataset):
    def __init__(self, data_path: str, vocab: Vocabulary, block_size: int, 
                 preprocessor: TextPreprocessor, split: str = 'train'):
        self.data_path = data_path
        self.vocab = vocab
        self.block_size = block_size
        self.preprocessor = preprocessor
        self.split = split
        
        # Load and preprocess data
        self.data = self._load_data()
        
    def _load_data(self) -> torch.Tensor:
        """Load and preprocess data."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean text
        text = self.preprocessor.clean_text(text)
        
        # Encode text
        encoded = self.vocab.encode(text)
        
        # Convert to tensor
        return torch.tensor(encoded, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a block of text and its target."""
        block = self.data[idx:idx + self.block_size]
        target = self.data[idx + 1:idx + self.block_size + 1]
        return block, target

def create_data_loaders(data_path: str, block_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    # Initialize components
    preprocessor = TextPreprocessor()
    vocab = Vocabulary()
    
    # Load and preprocess all data to build vocabulary
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = preprocessor.clean_text(text)
    vocab.build_vocab([text])
    
    # Save vocabulary
    vocab.save('vocab.json')
    
    # Create datasets
    train_size = int(len(text) * TRAIN_RATIO)
    train_text = text[:train_size]
    val_text = text[train_size:]
    
    # Save splits
    with open('train_split.txt', 'w', encoding='utf-8') as f:
        f.write(train_text)
    with open('val_split.txt', 'w', encoding='utf-8') as f:
        f.write(val_text)
    
    train_dataset = TextDataset('train_split.txt', vocab, block_size, preprocessor, 'train')
    val_dataset = TextDataset('val_split.txt', vocab, block_size, preprocessor, 'val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    return train_loader, val_loader

# Rotary Positional Embeddings
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
            self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=BIAS)
        self.query = nn.Linear(N_EMBD, head_size, bias=BIAS)
        self.value = nn.Linear(N_EMBD, head_size, bias=BIAS)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)
        self.rotary_emb = RotaryEmbedding(head_size // 2)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

# [1, 0, 0]
# [1, 0.6, 0]
# [1, 0.6, 0.4]
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD, bias=BIAS)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=BIAS),
            nn.GELU(),  # Changed from ReLU to GELU
            nn.Linear(4 * n_embd, n_embd, bias=BIAS),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
    
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD) # final layer norm
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights
        self.token_embedding_table.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        print(index.shape)
        B, T = index.shape
        
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        x = self.blocks(tok_emb) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens, temperature=1.0, top_k=None):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop index to block_size
            idx_cond = index[:, -BLOCK_SIZE:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index

# Initialize data loaders
train_loader, val_loader = create_data_loaders('openwebtext/train.txt', BLOCK_SIZE, BATCH_SIZE)

# Initialize model and move to device
model = GPTLanguageModel(vocab_size)
model = model.to(device)

# Initialize optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Initialize learning rate scheduler
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=WARMUP_STEPS,  # First restart epoch
    T_mult=2,          # Multiply T_0 by this factor after each restart
    eta_min=LEARNING_RATE * 0.1  # Minimum learning rate
)

# Initialize gradient scaler for mixed precision training
scaler = GradScaler()

# Training loop
best_val_loss = float('inf')
train_losses = []
val_losses = []
start_time = time.time()

print("Starting training...")
print(f"Configuration:")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Gradient Clip Value: {GRADIENT_CLIP_VAL}")
print(f"Warmup Steps: {WARMUP_STEPS}")
print(f"Save Every: {SAVE_EVERY} iterations")
print(f"Eval Every: {EVAL_EVERY} iterations")
print(f"Max Iterations: {MAX_ITERS}")
print(f"Learning Rate: {LEARNING_RATE}")

for iter in tqdm(range(MAX_ITERS)):
    # Get batch of data
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        
        # Forward pass with mixed precision
        with autocast():
            logits, loss = model(xb, yb)
        
        # Backward pass with gradient scaling
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
        
        # Optimizer step with gradient scaling
        scaler.step(optimizer)
        scaler.update()
        
        # Learning rate scheduling
        scheduler.step()
        
        # Evaluation
        if iter % EVAL_EVERY == 0:
            model.eval()
            val_losses_batch = []
            with torch.no_grad():
                for val_xb, val_yb in val_loader:
                    val_xb, val_yb = val_xb.to(device), val_yb.to(device)
                    with autocast():
                        _, val_loss = model(val_xb, val_yb)
                    val_losses_batch.append(val_loss.item())
            
            val_loss = sum(val_losses_batch) / len(val_losses_batch)
            model.train()
            
            print(f"\nIteration {iter}:")
            print(f"Train Loss: {loss.item():.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                with open('best_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
                print("Saved best model!")
        
        # Regular model checkpointing
        if iter % SAVE_EVERY == 0:
            with open(f'model_checkpoint_{iter}.pkl', 'wb') as f:
                pickle.dump({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'iter': iter,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss
                }, f)
            print(f"Saved checkpoint at iteration {iter}")

# Save final model
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)

# Print training summary
training_time = time.time() - start_time
print(f"\nTraining completed in {training_time/60:.2f} minutes")
print(f"Best validation loss: {best_val_loss:.4f}")
print("Model saved successfully!")

