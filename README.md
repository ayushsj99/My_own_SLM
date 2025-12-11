# My Own Small Language Model (SLM)

A GPT-style language model built from scratch using PyTorch, trained on the TinyStories dataset. This project demonstrates the complete pipeline of building, training, and deploying a small-scale transformer model.

## 🎯 Project Overview

This project implements a 30-million parameter GPT-style language model trained on children's stories from the TinyStories dataset. The model generates coherent short stories and demonstrates understanding of narrative structure, character development, and basic storytelling patterns.

**Training Time**: ~8 hours on NVIDIA GTX 1650 Ti (4GB VRAM)  
**Model Size**: 30M parameters (~120MB)  
**Dataset**: TinyStories (~943MB tokenized)

## 🛠️ Tech Stack

### Core Framework
- **PyTorch 2.7.1+cu118** - Deep learning framework with CUDA support
- **Python 3.13** - Programming language
- **CUDA 11.8** - GPU acceleration

### Data Processing
- **Hugging Face Datasets** - Dataset loading and processing
- **tiktoken** - GPT-2 tokenization (50,257 vocab size)
- **NumPy** - Numerical operations and memory mapping

### Training Infrastructure
- **Mixed Precision Training** - bfloat16 for memory efficiency
- **Gradient Accumulation** - Effective batch size optimization
- **Learning Rate Scheduling** - Warmup + Cosine Annealing
- **Memory Mapping** - Efficient data loading for large datasets

## 📁 Project Structure

```
My_own_SLM/
├── code.ipynb              # Main training notebook
├── train.bin              # Tokenized training data (~944MB)
├── validation.bin         # Tokenized validation data
├── best_model_params.pt   # Best model checkpoint
├── LICENSE                # Project license
└── README.md             # This file
```

## 🚀 Getting Started

### Prerequisites
- NVIDIA GPU with CUDA support (recommended)
- Python 3.8+ (Python 3.13 recommended)
- 8GB+ RAM
- 2GB+ disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ayushsj99/My_own_SLM.git
cd My_own_SLM
```

2. **Create and activate virtual environment**
```bash
# Using Python 3.13
python3.13 -m venv venv313
# Windows
.\venv313\Scripts\Activate.ps1
# Linux/Mac
source venv313/bin/activate
```

3. **Install dependencies**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install datasets tiktoken numpy tqdm ipywidgets
```

### Usage

1. **Open the notebook**
```bash
jupyter notebook code.ipynb
```

2. **Run cells sequentially**
   - Data loading and tokenization
   - Model architecture setup
   - Training configuration
   - Model training
   - Evaluation and text generation

3. **Monitor training**
   - GPU utilization via `nvidia-smi`
   - Training/validation loss curves
   - Learning rate scheduling

## 🏗️ Model Architecture

### GPT-Style Transformer
- **6 Layers** - Transformer blocks
- **6 Attention Heads** - Multi-head attention
- **384 Embedding Dimensions** - Hidden state size
- **128 Context Length** - Sequence length
- **30M Parameters** - Total trainable parameters

### Key Features
- **Causal Self-Attention** - Autoregressive text generation
- **Layer Normalization** - Training stability
- **Residual Connections** - Gradient flow
- **Weight Tying** - Input/output embedding sharing
- **Flash Attention** - Memory-efficient attention (when available)

## 📊 Training Process

### Configuration
```python
learning_rate = 1e-4
max_iters = 20000
batch_size = 32
gradient_accumulation_steps = 32
warmup_steps = 1000
dropout = 0.1
```

### Training Pipeline
1. **Data Preprocessing** (~10 minutes)
   - Download TinyStories dataset
   - Tokenize with GPT-2 tokenizer
   - Create memory-mapped binary files

2. **Model Training** (~8 hours on GTX 1650 Ti)
   - Mixed precision training (bfloat16)
   - Gradient accumulation for larger effective batch size
   - Learning rate warmup + cosine decay
   - Automatic best model checkpointing

3. **Validation** (Every 500 steps)
   - Evaluation on validation set
   - Loss monitoring and early stopping

## 📈 Performance & Results

### Hardware Performance
- **Training Speed**: ~0.5-1.0 seconds per iteration (GPU)
- **Memory Usage**: ~3.5GB VRAM (GTX 1650 Ti)
- **Total Training Time**: ~8 hours for 20,000 iterations

### Model Performance
- **Validation Loss**: Achieved stable convergence
- **Text Generation**: Coherent short stories with proper grammar
- **Narrative Structure**: Maintains character consistency and plot development

## 🎓 Key Learnings

### Technical Insights
1. **Environment Setup Challenges**
   - Python version compatibility (3.5 → 3.13 upgrade required)
   - CUDA-enabled PyTorch installation complexities
   - Virtual environment management for GPU workloads

2. **Training Optimization**
   - Mixed precision training reduces memory usage by ~40%
   - Gradient accumulation enables larger effective batch sizes on limited VRAM
   - Learning rate scheduling crucial for stable convergence

3. **Data Efficiency**
   - Memory mapping essential for large datasets (>1GB)
   - Efficient tokenization with multiprocessing speeds up preprocessing
   - Binary format reduces I/O overhead during training

### Development Workflow
1. **Modular Architecture** - Separate data processing, model definition, and training
2. **Comprehensive Logging** - Track loss curves, learning rates, and system metrics
3. **Checkpointing Strategy** - Save best models and enable training resumption
4. **GPU Monitoring** - Regular memory and utilization checks

### Challenges Overcome
- **Memory Constraints**: Optimized batch sizes and gradient accumulation
- **Training Stability**: Implemented gradient clipping and learning rate scheduling
- **Data Loading**: Efficient preprocessing pipeline for large text datasets
- **Environment Issues**: Systematic debugging of CUDA and PyTorch setup

## 🔮 Future Improvements

- **Larger Model**: Scale to 100M+ parameters with better hardware
- **Advanced Training**: Implement LoRA, gradient checkpointing
- **Better Evaluation**: Add perplexity metrics, human evaluation
- **Deployment**: Create inference API and web interface
- **Fine-tuning**: Adapt model for specific storytelling styles

## � Acknowledgments

This project was built following the excellent tutorial by **Shravan Koninti**:
- **Tutorial**: [Build a Small Language Model (SLM) from Scratch](https://medium.com/@shravankoninti/build-a-small-language-model-slm-from-scratch-3ddd13fa6470)

Special thanks for the clear explanations of transformer architecture, training procedures, and practical implementation guidance that made this project possible.

## �📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📧 Contact

**Ayush** - [@ayushsj99](https://github.com/ayushsj99)

Project Link: [https://github.com/ayushsj99/My_own_SLM](https://github.com/ayushsj99/My_own_SLM)
