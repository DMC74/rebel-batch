# file_watcher.py
import asyncio
from pathlib import Path
import time
from typing import AsyncGenerator, Dict, Set, Optional, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
import logging
from asyncio import Queue
from threading import Thread, Event
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class DocumentFileHandler(FileSystemEventHandler):
    """Handles file system events for new document files."""
    
    def __init__(self, file_queue: asyncio.Queue, file_pattern: str = "*.txt"):
        self.file_queue = file_queue
        self.file_pattern = file_pattern
        self.processed_files: Set[str] = set()
        self.logger = logging.getLogger(__name__)
        self.loop = asyncio.get_event_loop()

    def on_created(self, event):
        if not isinstance(event, FileCreatedEvent):
            return
            
        file_path = Path(event.src_path)
        if not file_path.match(self.file_pattern):
            return
            
        if str(file_path) in self.processed_files:
            return
            
        # Wait briefly to ensure file is completely written
        time.sleep(0.5)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create a future to run the coroutine in the main event loop
            future = asyncio.run_coroutine_threadsafe(
                self.file_queue.put({
                    'id': file_path.stem,
                    'text': content,
                    'path': str(file_path)
                }),
                self.loop
            )
            # Wait for the result with a timeout
            future.result(timeout=1.0)
            
            self.processed_files.add(str(file_path))
            self.logger.info(f"Queued new file: {file_path.name}")
            
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")

class DirectoryWatcher:
    """Watches a directory for new files and streams them for processing."""
    
    def __init__(self, 
                 input_dir: str, 
                 file_pattern: str = "*.txt",
                 queue_size: int = 1000):
        self.input_dir = Path(input_dir)
        self.file_pattern = file_pattern
        self.file_queue = asyncio.Queue(maxsize=queue_size)
        self.stop_event = Event()
        self.observer = None
        self.logger = logging.getLogger(__name__)
        self.loop = asyncio.get_event_loop()

    def start(self):
        """Start watching the directory."""
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        # Process existing files first
        asyncio.create_task(self._process_existing_files())
        
        # Start watching for new files
        self.observer = Observer()
        handler = DocumentFileHandler(self.file_queue, self.file_pattern)
        self.observer.schedule(handler, str(self.input_dir), recursive=False)
        self.observer.start()
        self.logger.info(f"Started watching directory: {self.input_dir}")

    def stop(self):
        """Stop watching the directory."""
        self.stop_event.set()
        if self.observer:
            self.observer.stop()
            self.observer.join()
        self.logger.info("Stopped watching directory")

    async def _process_existing_files(self):
        """Process any existing files in the directory."""
        for file_path in self.input_dir.glob(self.file_pattern):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                await self.file_queue.put({
                    'id': file_path.stem,
                    'text': content,
                    'path': str(file_path)
                })
                self.logger.info(f"Queued existing file: {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"Error reading existing file {file_path}: {str(e)}")

    async def stream_documents(self) -> AsyncGenerator[Dict[str, str], None]:
        """Stream documents from the watched directory."""
        while not self.stop_event.is_set() or not self.file_queue.empty():
            try:
                # Wait for new documents with timeout
                doc = await asyncio.wait_for(self.file_queue.get(), timeout=0.1)
                yield doc
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error streaming document: {str(e)}")
                continue

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example standalone usage
    async def main():
        watcher = DirectoryWatcher("test_input")
        watcher.start()
        
        try:
            async for doc in watcher.stream_documents():
                print(f"Received document: {doc['id']}")
                print(f"Content length: {len(doc['text'])}")
        except KeyboardInterrupt:
            print("\nStopping watcher...")
        finally:
            watcher.stop()
    
    asyncio.run(main())