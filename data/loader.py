import logging
import pickle
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading and processing FAQ data from a pickle file.
    """
    def __init__(self, data_path: str, augmented_data_path: str = 'dataset/qa_dataset_generated.pkl'):
        """
        Initialize the DataLoader with the path to the pickle file.
        
        Args:
            data_path: Path to the pickle file containing FAQ data
        """
        self.data_path = data_path
        self.augmented_data_path = augmented_data_path
        self.data = None
        self.augmented_data = None

    def load_data(self) -> Dict[str, str]:
        """
        Load the FAQ data from the pickle file.
        
        Returns:
            Dictionary mapping questions to answers
        """
        try:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
                self.augmented_data = pickle.load(open(self.augmented_data_path, 'rb'))
                # combine two dictionaries
                self.data.update(self.augmented_data)
            logger.info(f"Loaded {len(self.data)} FAQ pairs from {self.data_path}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            raise
    
    def get_question_answer_pairs(self) -> List[Tuple[str, str]]:
        """
        Convert the data dictionary to a list of (question, answer) tuples.
        
        Returns:
            List of (question, answer) tuples
        """
        if self.data is None:
            self.load_data()
        
        return [{'question':question, 'answer':answer} for question, answer in self.data.items()]