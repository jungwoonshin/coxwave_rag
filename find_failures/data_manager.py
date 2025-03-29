import logging
import random
from typing import Any, Dict, List, Optional

from data.loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Class for managing test data including loading, sampling, and collection setup.
    """

    def __init__(self, data_path: str, test_sample_size: Optional[int] = None):
        """
        Initialize the data manager.

        Args:
            data_path: Path to the FAQ data pickle file
            test_sample_size: Number of questions to sample (None for all)
        """
        self.data_path = data_path
        self.test_sample_size = test_sample_size
        self.faq_data = self._load_data()

    def _load_data(self) -> Dict[str, str]:
        """
        Load the FAQ data from pickle file.

        Returns:
            Dictionary of question-answer pairs
        """
        try:
            data_loader = DataLoader(self.data_path)
            data = data_loader.load_data()
            logger.info(f"Loaded {len(data)} FAQ entries")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def setup_collection(self, retriever: Any) -> None:
        """
        Set up the retriever collection with FAQ data.

        Args:
            retriever: The retriever to set up
        """
        qa_pairs = []
        for i, (question, answer) in enumerate(self.faq_data.items()):
            qa_pairs.append({"id": i, "question": question, "answer": answer})

        logger.info(f"Setting up retriever with {len(qa_pairs)} QA pairs")
        retriever.setup_collection(qa_pairs)
        logger.info(f"Retriever collection setup complete")

    def get_sample_questions(self) -> List[str]:
        """
        Get a sample of questions for testing.

        Returns:
            List of questions
        """
        all_questions = list(self.faq_data.keys())

        # Sample if needed
        if self.test_sample_size and self.test_sample_size < len(all_questions):
            random.seed(42)  # For reproducibility
            return random.sample(all_questions, self.test_sample_size)

        return all_questions

    def get_answer_for_question(self, question: str) -> str:
        """
        Get the answer for a given question.

        Args:
            question: The question

        Returns:
            The answer
        """
        return self.faq_data.get(question, "")
