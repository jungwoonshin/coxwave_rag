
from typing import List, Dict

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the document processor with chunking parameters."""
        pass
        
    def load_from_csv(self, file_path: str) -> List[Dict[str, str]]:
        """Load documents from a CSV file."""
        pass
        
    def load_from_txt(self, file_path: str) -> List[Dict[str, str]]:
        """Load documents from a text file."""
        pass
        
    def load_from_pdf(self, file_path: str) -> List[Dict[str, str]]:
        """Load documents from a PDF file."""
        pass
        
    def load_from_json(self, file_path: str) -> List[Dict[str, str]]:
        """Load documents from a JSON file."""
        pass
        
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Split documents into chunks."""
        pass