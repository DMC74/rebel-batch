# Installation Guide

## Prerequisites

Before installing Rebel Stream, ensure you have the following prerequisites:

* Python 3.10 or higher
* Anaconda or Miniconda
* Git
* 8GB RAM minimum (16GB recommended)
* (Optional) NVIDIA GPU with CUDA support

## Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/rebel-stream.git
cd rebel-stream
```

2. **Set Up the Environment**

Using the provided setup script:
```bash
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

Or manually:
```bash
# Create and activate conda environment
conda create -n rebel-stream python=3.10
conda activate rebel-stream

# Install PyTorch (CPU version)
conda install pytorch cpuonly -c pytorch

# For GPU support instead:
# conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt
```

3. **Verify Installation**
```bash
# Activate environment
conda activate rebel-stream

# Run a simple test
python -c "from src.streaming_extractor import StreamingRelationExtractor; print('Installation successful!')"
```

## Directory Structure Setup

The installation script will create the following directory structure:
```
data/
├── input_documents/    # Place documents here for processing
├── processed_documents/ # Processed results will appear here
└── archived_documents/  # Processed files are moved here
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, ensure CUDA toolkit is installed and try:
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

2. **Memory Issues**
```python
# Reduce batch size in configuration:
batch_size=4  # Default is 8
```

3. **Import Errors**
```bash
# Ensure PYTHONPATH includes the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Environment Variables

Configure the system using these environment variables:
```bash
export REBEL_BATCH_SIZE=8        # Number of segments to process at once
export REBEL_MAX_LENGTH=1024     # Maximum sequence length
export REBEL_DEVICE=-1           # GPU device ID (-1 for CPU)
```

## Updating

To update to the latest version:
```bash
git pull origin main
./scripts/setup_environment.sh
```