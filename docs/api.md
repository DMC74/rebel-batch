# API Reference

## Core Classes

### BaseRelationExtractor

Base class providing core relation extraction functionality.

```python
class BaseRelationExtractor:
    def __init__(self, 
                 model_name: str = 'Babelscape/mrebel-large-32', 
                 device: int = 0,
                 max_length: int = 200):
```

#### Parameters:
- `model_name` (str): Name of the pretrained model
- `device` (int): GPU device ID (-1 for CPU)
- `max_length` (int): Maximum sequence length

#### Methods:

##### `detect_language(text: str) -> str`
Detect the language of input text.

##### `get_token_length(text: str) -> int`
Get the token length of a text string.

##### `segment_document(text: str) -> List[str]`
Split document into processable segments.

##### `process_segment(segment: str, src_lang: str) -> List[Dict[str, str]]`
Process a single document segment.

### StreamingRelationExtractor

Streaming implementation of the relation extractor.

```python
class StreamingRelationExtractor(BaseRelationExtractor):
    def __init__(self,
                 model_name: str = 'Babelscape/mrebel-large-32',
                 device: int = 0,
                 max_length: int = 200,
                 batch_size: int = 8,
                 queue_size: int = 1000):
```

#### Parameters:
- Inherits from BaseRelationExtractor
- `batch_size` (int): Number of segments to process at once
- `queue_size` (int): Maximum size of processing queue

#### Methods:

##### `async process_stream(document_stream: AsyncGenerator[Dict[str, str], None], output_dir: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]`
Process a stream of documents asynchronously.

Parameters:
- `document_stream`: Async generator yielding documents
- `output_dir`: Optional directory to save results

Returns:
- Async generator yielding processed results

### DirectoryWatcher

Monitors directory for new files and streams them for processing.

```python
class DirectoryWatcher:
    def __init__(self,
                 input_dir: str,
                 file_pattern: str = "*.txt",
                 queue_size: int = 1000):
```

#### Parameters:
- `input_dir` (str): Directory to watch
- `file_pattern` (str): File pattern to match
- `queue_size` (int): Maximum queue size

#### Methods:

##### `start()`
Start watching the directory.

##### `stop()`
Stop watching the directory.

##### `async stream_documents() -> AsyncGenerator[Dict[str, str], None]`
Stream documents from the watched directory.

## Data Structures

### Document Format

Input document dictionary:
```python
{
    'id': str,      # Document identifier
    'text': str,    # Document text content
    'path': str     # Optional file path
}
```

### Result Format

Processing result dictionary:
```python
{
    'doc_id': str,          # Document identifier
    'language': str,        # Detected language code
    'triplets': List[Dict], # Extracted relations
    'num_segments': int,    # Number of segments processed
    'path': str            # Optional original file path
}
```

### Relation Format

Extracted relation dictionary:
```python
{
    'head': str,        # Subject entity
    'head_type': str,   # Subject entity type
    'type': str,        # Relation type
    'tail': str,        # Object entity
    'tail_type': str    # Object entity type
}
```

## Error Handling

### Exception Types

```python
class ProcessingError(Exception):
    """Base class for processing errors"""
    pass

class DocumentError(ProcessingError):
    """Error reading or processing document"""
    pass

class ModelError(ProcessingError):
    """Error in model inference"""
    pass
```

### Error Handling Example

```python
try:
    async for result in extractor.process_stream(document_stream()):
        handle_result(result)
except DocumentError as e:
    logger.error(f"Document processing error: {e}")
except ModelError as e:
    logger.error(f"Model inference error: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| REBEL_BATCH_SIZE | Batch size for processing | 8 |
| REBEL_MAX_LENGTH | Maximum sequence length | 1024 |
| REBEL_DEVICE | GPU device ID (-1 for CPU) | -1 |
| REBEL_QUEUE_SIZE | Maximum queue size | 1000 |

### Configuration Example

```python
import os

os.environ['REBEL_BATCH_SIZE'] = '16'
os.environ['REBEL_DEVICE'] = '0'
```

## Logging

### Log Levels

- DEBUG: Detailed information
- INFO: General information
- WARNING: Warning messages
- ERROR: Error messages
- CRITICAL: Critical errors

### Logging Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rebel_stream.log'),
        logging.StreamHandler()
    ]
)
```

## Performance Considerations

### Memory Usage

- Batch size affects memory usage
- GPU memory requirements
- Queue size impact

### Processing Speed

- GPU vs CPU processing
- Batch size optimization
- Document segmentation impact

## Examples

### Basic Usage

```python
from rebel_stream import StreamingRelationExtractor, DirectoryWatcher
import asyncio

async def main():
    watcher = DirectoryWatcher("input_documents")
    extractor = StreamingRelationExtractor()
    
    watcher.start()
    async for result in extractor.process_stream(
        watcher.stream_documents()
    ):
        print(f"Processed: {result['doc_id']}")

asyncio.run(main())
```

### Advanced Configuration

```python
extractor = StreamingRelationExtractor(
    model_name='Babelscape/mrebel-large-32',
    device=0,
    max_length=2048,
    batch_size=16,
    queue_size=2000
)
```