# Rebel Stream

A real-time streaming implementation of the REBEL (Relation Extraction By End-to-end Language generation) model that processes documents as they arrive and extracts structured relations. Uses Ray AI Runtime for distributed processing and efficient scaling.

## Features

- Real-time document processing with distributed computing via Ray
- Streaming architecture for efficient processing
- Support for multiple languages
- Automatic language detection
- Batch processing with parallel execution
- Asynchronous file watching and processing
- Structured output in JSON format
- Automatic file archiving
- GPU acceleration with automatic resource management
- Horizontal scaling across multiple workers

## Requirements

- Python 3.10+
- Anaconda or Miniconda
- 8GB RAM minimum (16GB recommended)
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- Ray AI Runtime


The processed results will appear in `data/processed_documents/` and the original files will be moved to `data/archived_documents/`.

## Ray Configuration

The system uses Ray for distributed processing. Configure Ray settings through environment variables:

- `RAY_NUM_WORKERS`: Number of parallel workers (default: number of available GPUs or 4)
- `RAY_ADDRESS`: Ray cluster address (default: local)
- `RAY_OBJECT_STORE_MEMORY`: Memory limit for Ray's object store (default: 2GB)
- `RAY_GPU_FRACTION`: GPU memory fraction per worker (default: 0.5)

For multi-node deployment:
```bash
# On head node
ray start --head --port=6379

# On worker nodes
ray start --address='<head-node-address>:6379'
```

## Directory Structure

- `data/input_documents/`: Place documents here for processing
- `data/processed_documents/`: Contains extracted relations in JSON format
- `data/archived_documents/`: Contains processed input files
- `src/`: Source code
- `scripts/`: Utility scripts
- `docs/`: Documentation
- `tests/`: Test files

## Output Format

The extracted relations are saved in JSON format:
```json
{
    "doc_id": "example",
    "language": "en_XX",
    "triplets": [
        {
            "head": "Company",
            "head_type": "ORGANIZATION",
            "type": "headquarters_location",
            "tail": "City",
            "tail_type": "LOCATION"
        }
    ],
    "num_segments": 10
}
```

## Configuration

Configure the processor using environment variables:
- `REBEL_BATCH_SIZE`: Number of segments to process at once (default: 8)
- `REBEL_MAX_LENGTH`: Maximum sequence length (default: 1024)
- `REBEL_NUM_WORKERS`: Number of Ray workers (default: auto-detected based on GPUs)
- `REBEL_DEVICE`: GPU device ID or -1 for CPU (default: auto-detected)

## Performance Optimization

### GPU Configuration
For optimal performance with GPUs:
```bash
# Set GPU memory fraction per worker
export RAY_GPU_FRACTION=0.5

# Specify visible GPU devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Start with specific number of GPU workers
python src/ray_relation_extractor.py --num-workers 4
```

### CPU Configuration
For CPU-only deployment:
```bash
# Disable GPU usage
export CUDA_VISIBLE_DEVICES=""

# Set number of CPU workers
export RAY_NUM_CPU=8
```

## Documentation

Detailed documentation is available in the `docs/` directory:
- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api.md)
- [Ray Integration Guide](docs/ray_integration.md)
- [Scaling Guide](docs/scaling.md)

## License

MIT License - see LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@article{huguet2021rebel,
    title={REBEL: Relation Extraction By End-to-end Language generation},
    author={Huguet Cabot, Pere-Lluís and Navigli, Roberto},
    journal={arXiv preprint arXiv:2104.07650},
    year={2021}
}
```

## Troubleshooting

### Common Issues

1. GPU Not Detected
```bash
# Check GPU visibility
python -c "import torch; print(torch.cuda.is_available())"

# Check Ray resources
python -c "import ray; ray.init(); print(ray.available_resources())"
```

2. Memory Issues
```bash
# Adjust Ray object store memory
export RAY_OBJECT_STORE_MEMORY=4000000000  # 4GB
```

3. Worker Initialization Failures
```bash
# Enable detailed Ray logging
export RAY_VERBOSITY=debug
```

