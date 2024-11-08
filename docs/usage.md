# Usage Guide

## Quick Start

1. **Start the Processor**
```bash
./scripts/start_processor.sh
```

2. **Process Documents**
```bash
# Add files to process
cp your_document.txt data/input_documents/
```

3. **View Results**
```bash
# Check processed results
cat data/processed_documents/your_document_relations.json
```

## Basic Usage

### Processing Single Documents

```python
import asyncio
from src.streaming_extractor import StreamingRelationExtractor

async def process_single_file():
    extractor = StreamingRelationExtractor()
    
    async def document_stream():
        yield {
            'id': 'test_doc',
            'text': 'Apple Inc. is headquartered in Cupertino, California.'
        }
    
    async for result in extractor.process_stream(document_stream()):
        print(f"Found {len(result['triplets'])} relations")

asyncio.run(process_single_file())
```

### Directory Watching

```python
from src.file_watcher import DirectoryWatcher
from src.streaming_extractor import StreamingRelationExtractor
import asyncio

async def watch_directory():
    watcher = DirectoryWatcher("data/input_documents")
    extractor = StreamingRelationExtractor()
    
    watcher.start()
    try:
        async for result in extractor.process_stream(
            watcher.stream_documents(),
            output_dir="data/processed_documents"
        ):
            print(f"Processed: {result['doc_id']}")
    finally:
        watcher.stop()

asyncio.run(watch_directory())
```

## Advanced Usage

### Custom Configuration

```python
from src.streaming_extractor import StreamingRelationExtractor

# Configure for GPU usage
extractor = StreamingRelationExtractor(
    model_name='Babelscape/mrebel-large-32',
    device=0,               # Use first GPU
    max_length=2048,        # Longer sequences
    batch_size=16,          # Larger batches
    queue_size=2000         # Larger queue
)
```

### Processing Multiple Languages

The system automatically detects and processes multiple languages:
```python
documents = [
    {"id": "en", "text": "Apple is based in Cupertino."},
    {"id": "es", "text": "Madrid es la capital de España."},
    {"id": "fr", "text": "Paris est la capitale de la France."}
]
```

### Custom Processing Pipeline

```python
import asyncio
from src.streaming_extractor import StreamingRelationExtractor

async def custom_pipeline():
    extractor = StreamingRelationExtractor()
    
    async def document_stream():
        # Your custom document source
        docs = get_documents_from_source()
        for doc in docs:
            # Custom preprocessing
            processed_text = preprocess(doc.text)
            yield {
                'id': doc.id,
                'text': processed_text
            }
    
    async def process_results(results_stream):
        async for result in results_stream:
            # Custom post-processing
            processed_results = postprocess(result)
            # Custom storage
            store_results(processed_results)
    
    await process_results(
        extractor.process_stream(document_stream())
    )

asyncio.run(custom_pipeline())
```

## Best Practices

1. **Document Preparation**
   - Clean text before processing
   - Keep documents under 100KB for optimal processing
   - Use UTF-8 encoding

2. **Performance Optimization**
   - Use GPU for large-scale processing
   - Adjust batch size based on available memory
   - Monitor system resources

3. **Error Handling**
   - Implement try-catch blocks
   - Log errors appropriately
   - Set up monitoring

4. **Resource Management**
   - Clean up old files regularly
   - Monitor disk space
   - Implement backups

## Monitoring

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### Performance Metrics

Monitor:
- Processing time per document
- Relations extracted per document
- Queue sizes
- Memory usage
- GPU utilization (if applicable)

## Troubleshooting

Common runtime issues and solutions:

1. **Memory Issues**
   - Reduce batch size
   - Process smaller documents
   - Use CPU if GPU memory is limited

2. **Processing Errors**
   - Check input file encoding
   - Verify file permissions
   - Monitor log files

3. **Performance Issues**
   - Enable GPU processing
   - Adjust batch size
   - Check system resources