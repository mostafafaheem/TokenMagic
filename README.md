# Custom GPT Language Model Implementation

## Project Overview

This project implements a GPT-style transformer language model from scratch using PyTorch. The implementation includes a custom Byte Pair Encoding (BPE) tokenizer and a complete transformer architecture with causal self-attention mechanisms. This demonstrates deep understanding of modern NLP architectures and provides a foundation for building and experimenting with language models.

## Features

- **Custom BPE Tokenizer**: Implemented from scratch with regex-based text segmentation
- **Transformer Architecture**: Complete GPT-style model with causal self-attention
- **Multi-Head Attention**: Configurable attention heads and embedding dimensions
- **Special Tokens**: Support for PAD, UNK, BOS, and EOS tokens
- **Text Generation**: Top-k sampling with configurable parameters
- **GPU/CPU Support**: Automatic device detection and optimization
- **Configurable Architecture**: Flexible hyperparameters for different model sizes

## Architecture

The model consists of several key components:

- **Causal Self-Attention**: Implements masked attention to prevent future token access
- **Multi-Head Attention**: Parallel attention mechanisms for capturing different relationships
- **MLP Blocks**: Feed-forward networks with GELU activation
- **Layer Normalization**: Pre-normalization architecture for training stability
- **Position Embeddings**: Learnable positional encodings for sequence understanding
- **Token Embeddings**: Word-level embeddings with configurable vocabulary size

## Installation

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- `torch >= 2.2.2` - PyTorch framework
- `tiktoken >= 0.5.1` - Tokenization utilities
- `matplotlib >= 3.7.1` - Visualization
- `tqdm >= 4.66.1` - Progress bars
- `numpy >= 1.26, < 2.1` - Numerical computing
- `pandas >= 2.2.1` - Data manipulation
- `psutil >= 5.9.5` - System monitoring

## Usage Examples

### Training the Custom Tokenizer

```python
from tokenizer import BPETokenizer

# Initialize tokenizer
tokenizer = BPETokenizer()

# Train on your text data
with open("your_text_file.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

# Train with desired vocabulary size
tokenizer.train(vocabulary_size=10000, text=text_data)

# Save trained tokenizer
tokenizer.save("my_tokenizer")
```

### Text Generation

```python
from train import ModelName, Config
import torch

# Initialize model
config = Config(
    block_size=1024,
    vocab_size=50257,
    n_layer=12,
    n_head=12,
    n_embd=768
)

model = ModelName(config)
model.eval()

# Generate text
prompt = "Hello, I'm a language model"
# ... (see train.py for complete generation example (random weights))
```

## Configuration

The model supports various hyperparameters:

- `block_size`: Maximum sequence length (default: 1024)
- `vocab_size`: Vocabulary size (default: 50257)
- `n_layer`: Number of transformer layers (default: 12)
- `n_head`: Number of attention heads (default: 12)
- `n_embd`: Embedding dimension (default: 768)
- `dropout`: Dropout rate (default: 0.2)

## File Structure

- `train.py` - Main model architecture and text generation implementation
- `tokenizer.py` - Custom BPE tokenizer with training and inference capabilities
- `model.py` - Alternative model implementation (commented out)
- `tokenizer_test.py` - Tokenizer testing and validation
- `requirements.txt` - Python dependencies

## Future Enhancements

The project roadmap includes several advanced features (see `features.txt`):

- Group Query Attention (GQA)
- SwiGLU activation functions
- Rotary Positional Embeddings (RoPE)
- RMSNorm normalization
- YARN scaling during training
- Pre-normalization architecture
- QK normalization
- Dual chunk attention
- 8-bit training support
- MLA-KV cache compression

## Contributing

This is an educational project demonstrating transformer architecture implementation. Feel free to experiment with different configurations and enhancements.

## License

This project is for educational purposes. Please ensure compliance with any data usage policies when training on external datasets.
