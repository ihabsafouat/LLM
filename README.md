# GPT Language Model from Scratch

A PyTorch implementation of a GPT-style language model with modern training techniques and efficient data processing.

## Project Structure

```
gpt-from-scratch/
├── data/                      # Data directory
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed data files
│   └── vocab/                 # Vocabulary files
├── notebooks/                 # Jupyter notebooks
│   ├── bigram.ipynb          # Bigram model implementation
│   ├── bpe-v1.ipynb          # BPE tokenizer implementation
│   ├── gpt-v1.ipynb          # Initial GPT implementation
│   └── gpt-v2.ipynb          # Improved GPT implementation
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessor.py   # Text preprocessing
│   │   ├── vocabulary.py     # Vocabulary management
│   │   └── dataset.py        # Dataset implementation
│   ├── model/                # Model architecture
│   │   ├── __init__.py
│   │   ├── gpt.py           # GPT model implementation
│   │   └── attention.py      # Attention mechanisms
│   └── utils/                # Utility functions
│       ├── __init__.py
│       └── training.py       # Training utilities
├── scripts/                  # Training and utility scripts
│   ├── train.py             # Main training script
│   └── data_extract.py      # Data extraction script
├── tests/                    # Unit tests
├── configs/                  # Configuration files
├── requirements.txt          # Project dependencies
├── setup.py                 # Package setup file
└── README.md                # This file
```

## Features

- **Modern Architecture**:
  - Rotary Positional Embeddings
  - Multi-head attention
  - Layer normalization
  - GELU activation

- **Efficient Training**:
  - Mixed precision training (FP16)
  - Gradient clipping
  - Learning rate scheduling
  - Model checkpointing

- **Data Processing**:
  - Efficient text preprocessing
  - Character-level tokenization
  - Special tokens support
  - Memory-mapped data loading

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/gpt-from-scratch.git
   cd gpt-from-scratch
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model:

```bash
python scripts/train.py --batch_size 32 --gradient_clip_val 1.0 --warmup_steps 1000
```

### Configuration

Key parameters can be adjusted in the training script:

```python
# Training Configuration
BATCH_SIZE = 32
GRADIENT_CLIP_VAL = 1.0
WARMUP_STEPS = 1000
SAVE_EVERY = 1000
EVAL_EVERY = 100
MAX_ITERS = 200
LEARNING_RATE = 3e-4

# Model Configuration
BLOCK_SIZE = 256
N_EMBD = 768
N_HEAD = 12
N_LAYER = 6
DROPOUT = 0.1
```

### Google Colab

For training in Google Colab:
1. Open the notebook in `notebooks/gpt-v2.ipynb`
2. Mount your Google Drive
3. Adjust the configuration parameters
4. Run the training cells

## Model Architecture

The model implements a GPT-style architecture with:
- Character-level tokenization
- Rotary positional embeddings
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Residual connections

## Training Process

1. **Data Processing**:
   - Text cleaning and normalization
   - Vocabulary building
   - Train/validation split

2. **Training Loop**:
   - Mixed precision training
   - Gradient clipping
   - Learning rate scheduling
   - Regular evaluation
   - Model checkpointing

3. **Evaluation**:
   - Loss monitoring
   - Perplexity calculation
   - Validation metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the "Attention is All You Need" paper
- Inspired by Andrej Karpathy's nanoGPT
- Uses PyTorch for implementation

## Google Colab for those who don't have a GPU: https://colab.research.google.com/drive/1_7TNpEEl8xjHlr9JzKbK5AuDKXwAkHqj?usp=sharing

Dependencies (assuming windows): `pip install pylzma numpy ipykernel jupyter torch --index-url https://download.pytorch.org/whl/cu118`

If you don't have an NVIDIA GPU, then the `device` parameter will default to `'cpu'` since `device = 'cuda' if torch.cuda.is_available() else 'cpu'`. If device is defaulting to `'cpu'` that is fine, you will just experience slower runtimes.

## All the links you should need are in this repo. I will add detailed explanations as questions and issues are posted.

## Visual Studio 2022 (for lzma compression algo) - https://visualstudio.microsoft.com/downloads/

## OpenWebText Download
- https://skylion007.github.io/OpenWebTextCorpus/
- if this doesn't work, default to the wizard of oz mini dataset for training / validation

## Socials
Twitter / X - https://twitter.com/elliotarledge

My YouTube Channel - https://www.youtube.com/channel/UCjlt_l6MIdxi4KoxuMjhYxg

How to SSH from Mac to Windows - https://www.youtube.com/watch?v=7hBeAb6WyIg&t=

How to Setup Jupyter Notebooks in 5 minutes or less - https://www.youtube.com/watch?v=eLmweqU5VBA&t=

Linkedin - https://www.linkedin.com/in/elliot-arledge-a392b7243/

Join My Discord Server - https://discord.gg/pV7ByF9VNm

Schedule a 1-on-1: https://calendly.com/elliot-ayxc/60min

## Research Papers:
Attention is All You Need - https://arxiv.org/pdf/1706.03762.pdf

A Survey of LLMs - https://arxiv.org/pdf/2303.18223.pdf

QLoRA: Efficient Finetuning of Quantized LLMs - https://arxiv.org/pdf/2305.14314.pdf
