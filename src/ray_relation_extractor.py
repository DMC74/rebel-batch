# ray_relation_extractor.py
import ray
from typing import AsyncGenerator, Dict, Any, Optional, List
import asyncio
from pathlib import Path
import json
from datetime import datetime
import logging
from transformers import pipeline, AutoTokenizer
import torch
from base_extractor import BaseRelationExtractor
from file_watcher import DirectoryWatcher

# Define GPU resource requirements for the worker
@ray.remote(num_gpus=1)  # Request 1 GPU for this actor
class RelationExtractorWorker:
    """Ray actor for processing text segments."""
    
    def __init__(self, model_name: str, device_id: int, max_length: int):
        # Force GPU usage since we've requested GPU resources
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is not available")
            
        device = f'cuda:{device_id}'
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing worker with device: {device}")
        
        # Set the CUDA device for this worker
        torch.cuda.set_device(device_id)
        
        self.extractor = BaseRelationExtractor(
            model_name=model_name,
            device=device,
            max_length=max_length
        )
    
    def process_segment(self, segment: str, src_lang: str) -> List[Dict[str, str]]:
        """Process a single text segment."""
        return self.extractor.process_segment(segment, src_lang)

@ray.remote
class DocumentProcessor:
    """Ray actor for document preprocessing."""
    
    def __init__(self, model_name: str):
        self.extractor = BaseRelationExtractor(
            model_name=model_name,
            device='cpu'  # Keep preprocessing on CPU as it's less intensive
        )
    
    def preprocess_document(self, doc_id: str, text: str) -> Dict[str, Any]:
        """Preprocess document by detecting language and segmenting."""
        src_lang = self.extractor.detect_language(text)
        segments = self.extractor.segment_document(text)
        return {
            'doc_id': doc_id,
            'segments': segments,
            'src_lang': src_lang
        }

class RayStreamingExtractor:
    """Distributed streaming relation extractor using Ray."""
    
    def __init__(
        self,
        model_name: str = 'Babelscape/mrebel-large-32',
        num_workers: int = 4,
        max_length: int = 1024,
        batch_size: int = 8
    ):
        self.model_name = model_name
        self.num_workers = num_workers
        self.max_length = max_length
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize Ray with GPU support if available
        if not ray.is_initialized():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                self.logger.info(f"Initializing Ray with {num_gpus} GPUs")
                ray.init(num_gpus=num_gpus)
            else:
                self.logger.warning("No GPUs detected, falling back to CPU")
                ray.init()
            
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize Ray actors for processing."""
        self.document_processor = DocumentProcessor.remote(self.model_name)
        
        num_gpus = torch.cuda.device_count()
        self.logger.info(f"Number of available GPUs: {num_gpus}")
        
        if num_gpus == 0:
            self.logger.warning("No GPUs available. Workers will use CPU.")
            self.num_workers = min(self.num_workers, ray.available_resources()['CPU'])
        else:
            # Adjust number of workers based on available GPUs
            self.num_workers = min(self.num_workers, num_gpus)
            self.logger.info(f"Adjusted number of workers to {self.num_workers} based on GPU availability")
        
        self.workers = []
        for i in range(self.num_workers):
            worker = RelationExtractorWorker.remote(
                self.model_name,
                i % max(1, num_gpus),  # Ensure proper GPU device assignment
                self.max_length
            )
            self.workers.append(worker)

    async def _get_ray_async(self, object_ref):
        """Helper method to get Ray objects asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, ray.get, object_ref)

    async def process_stream(
        self,
        document_stream: AsyncGenerator[Dict[str, str], None],
        output_dir: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a stream of documents using distributed workers."""
        try:
            async for document in document_stream:
                doc_id = document.get('id', str(datetime.now().timestamp()))
                text = document.get('text', '').strip()
                
                if not text:
                    self.logger.warning(f"Empty document received: {doc_id}")
                    continue
                
                # Preprocess document using Ray
                preprocessed = await self._get_ray_async(
                    self.document_processor.preprocess_document.remote(doc_id, text)
                )
                
                # Process segments in parallel
                segment_refs = []
                for i in range(0, len(preprocessed['segments']), self.batch_size):
                    batch = preprocessed['segments'][i:i + self.batch_size]
                    for segment in batch:
                        worker = self.workers[len(segment_refs) % self.num_workers]
                        ref = worker.process_segment.remote(
                            segment,
                            preprocessed['src_lang']
                        )
                        segment_refs.append(ref)
                
                # Collect results using Ray
                all_triplets = []
                for refs_batch in self._chunk_refs(segment_refs, self.batch_size):
                    results = await self._get_ray_async(refs_batch)
                    for triplets in results:
                        all_triplets.extend(triplets)
                
                final_result = {
                    'doc_id': doc_id,
                    'language': preprocessed['src_lang'],
                    'triplets': all_triplets,
                    'num_segments': len(preprocessed['segments']),
                    'path': document.get('path')
                }
                
                if output_dir:
                    await self._save_document_results(final_result, output_dir)
                
                yield final_result
                
        except Exception as e:
            self.logger.error(f"Error processing stream: {str(e)}")
            raise
    
    def _chunk_refs(self, refs, size):
        """Split references into chunks for batch processing."""
        for i in range(0, len(refs), size):
            yield refs[i:i + size]
    
    async def _save_document_results(self, result: Dict[str, Any], output_dir: str):
        """Save document results asynchronously."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            output_file = output_path / f"{result['doc_id']}_relations.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving document results: {str(e)}")

async def main():
    input_dir = "data/input_documents"
    output_dir = "data/processed_documents"
    
    watcher = DirectoryWatcher(input_dir)
    extractor = RayStreamingExtractor(
        model_name='Babelscape/mrebel-large-32',
        num_workers=4,
        max_length=1024,
        batch_size=8
    )
    
    try:
        watcher.start()
        
        async for result in extractor.process_stream(
            watcher.stream_documents(),
            output_dir=output_dir
        ):
            print(f"\nProcessed document: {result['doc_id']}")
            print(f"Found {len(result['triplets'])} relations")
            
            input_file = Path(result.get('path', ''))
            if input_file.exists():
                archive_dir = Path("data/archived_documents")
                archive_dir.mkdir(exist_ok=True)
                input_file.rename(archive_dir / input_file.name)
                
    except KeyboardInterrupt:
        print("\nStopping document processing...")
    finally:
        watcher.stop()
        ray.shutdown()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())