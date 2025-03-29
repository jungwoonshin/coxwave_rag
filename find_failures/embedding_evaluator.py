import logging
from typing import Any, Dict, List

import numpy as np

# Import components from your system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingEvaluator:
    """
    Class for evaluating answers using embedding-based similarity.
    """
    def __init__(self, embedder: Any):
        """
        Initialize the evaluator.
        
        Args:
            embedder: The embedding model to use
        """
        self.embedder = embedder
    
    def evaluate_batch(self, generated_answers: List[str], correct_answers: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple generated answers against their correct answers using batch processing.
        
        Args:
            generated_answers: List of answers generated by the RAG system
            correct_answers: List of correct answers from the FAQ
            
        Returns:
            List of dictionaries with evaluation metrics for each answer pair
        """
        # Validate input
        if len(generated_answers) != len(correct_answers):
            raise ValueError("Number of generated answers must match number of correct answers")
        
        # Return empty list if no answers to evaluate
        if len(generated_answers) == 0:
            return []
        
        # Process in batches to optimize embedding generation
        try:
            # Get embeddings for all answers in one batch call
            all_texts = generated_answers + correct_answers
            all_embeddings = self.embedder.get_embeddings_batch(all_texts)
            
            # Split embeddings back into generated and correct
            num_answers = len(generated_answers)
            generated_embeddings = all_embeddings[:num_answers]
            correct_embeddings = all_embeddings[num_answers:]
            
            # Calculate batch cosine similarity
            cosine_similarities = self._compute_batch_cosine_similarity(generated_embeddings, correct_embeddings)
            
            # Calculate batch metrics
            semantic_scores = np.minimum(5, np.maximum(1, cosine_similarities * 5))
            
            # Calculate length ratios
            length_ratios = np.zeros(num_answers)
            for i in range(num_answers):
                if len(correct_answers[i]) > 0:
                    length_ratios[i] = len(generated_answers[i]) / len(correct_answers[i])
            
            # Calculate completeness scores
            completeness_scores = 5 - np.minimum(4, np.abs(length_ratios - 1) * 5)
            completeness_scores = np.maximum(1, completeness_scores)  # Ensure minimum score of 1
            
            # Calculate word overlap metrics (for relevance) and conciseness
            word_overlaps = np.zeros(num_answers)
            conciseness_scores = np.zeros(num_answers)
            
            for i in range(num_answers):
                generated_words = set(generated_answers[i].lower().split())
                correct_words = set(correct_answers[i].lower().split())
                
                # Calculate Jaccard similarity for word overlap
                union_size = len(generated_words.union(correct_words))
                if union_size > 0:
                    word_overlaps[i] = len(generated_words.intersection(correct_words)) / union_size
                
                # Conciseness score - penalize verbosity
                conciseness_penalty = max(0, length_ratios[i] - 1.2)  # Penalty if more than 20% longer
                conciseness_scores[i] = min(5, max(1, 5 - conciseness_penalty * 3))
            
            # Calculate relevance scores
            relevance_scores = np.minimum(5, np.maximum(1, word_overlaps * 5))
            
            # Process results for each pair
            results = []
            for i in range(num_answers):
                # Calculate overall score
                overall_score = round((semantic_scores[i] + completeness_scores[i] + relevance_scores[i] + conciseness_scores[i]) / 4, 2)
                
                # Determine pass/fail threshold
                passed = overall_score >= 4.0
                
                # Generate explanations based on metrics
                reasons = []
                if semantic_scores[i] < 3.5:
                    reasons.append("The answer's semantic meaning differs significantly from the reference.")
                if completeness_scores[i] < 3.5:
                    if length_ratios[i] < 0.8:
                        reasons.append("The answer is too short compared to the reference.")
                    elif length_ratios[i] > 1.2:
                        reasons.append("The answer is unnecessarily verbose.")
                if relevance_scores[i] < 3.5:
                    reasons.append("The answer lacks key terms from the reference.")
                if conciseness_scores[i] < 3.5:
                    reasons.append("The answer could be more concise.")
                    
                # Compile evaluation results
                evaluation = {
                    "correctness": round(float(semantic_scores[i]), 2),
                    "completeness": round(float(completeness_scores[i]), 2),
                    "relevance": round(float(relevance_scores[i]), 2),
                    "conciseness": round(float(conciseness_scores[i]), 2),
                    "overall": overall_score,
                    "passed": passed,
                    "cosine_similarity": round(float(cosine_similarities[i]), 4),
                    "length_ratio": round(float(length_ratios[i]), 2),
                    "word_overlap": round(float(word_overlaps[i]), 4),
                    "reasons": "; ".join(reasons) if reasons else "Answer meets quality thresholds."
                }
                
                results.append(evaluation)
            
            return results
        
        except Exception as e:
            logger.error(f"Error in batch evaluation: {e}")
            # Return error results for all pairs
            return [
                {
                    "correctness": 0,
                    "completeness": 0, 
                    "relevance": 0,
                    "conciseness": 0,
                    "overall": 0,
                    "passed": False,
                    "reasons": f"Batch evaluation error: {str(e)}"
                }
                for _ in range(len(generated_answers))
            ]
    
    def evaluate_single(self, generated_answer: str, correct_answer: str) -> Dict[str, Any]:
        """
        Evaluate a single generated answer against the correct answer.
        
        Args:
            generated_answer: Answer generated by the RAG system
            correct_answer: The correct answer from the FAQ
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = self.evaluate_batch([generated_answer], [correct_answer])
        return results[0] if results else {
            "correctness": 0,
            "completeness": 0, 
            "relevance": 0,
            "conciseness": 0,
            "overall": 0,
            "passed": False,
            "reasons": "Evaluation failed"
        }
    
    def _evaluate_single_answer(self, generated_answer, correct_answer, generated_embedding, correct_embedding):
        """
        Process a single answer evaluation with pre-computed embeddings.
        
        Args:
            generated_answer: The generated answer text
            correct_answer: The correct answer text
            generated_embedding: Pre-computed embedding for generated answer
            correct_embedding: Pre-computed embedding for correct answer
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Calculate cosine similarity
            cosine_similarity = self._compute_cosine_similarity(generated_embedding, correct_embedding)
            
            # Calculate semantic similarity score (0-5 scale)
            semantic_score = min(5, max(1, cosine_similarity * 5))
            
            # Calculate length ratio (to evaluate completeness)
            length_ratio = len(generated_answer) / len(correct_answer) if len(correct_answer) > 0 else 0
            # Penalize if too short or too long (ideal is 0.8-1.2 times the length)
            completeness_score = 5 - min(4, abs(length_ratio - 1) * 5)
            completeness_score = max(1, completeness_score)  # Ensure minimum score of 1
            
            # Calculate word overlap (for relevance)
            generated_words = set(generated_answer.lower().split())
            correct_words = set(correct_answer.lower().split())
            
            # Calculate Jaccard similarity for word overlap
            if len(generated_words.union(correct_words)) > 0:
                word_overlap = len(generated_words.intersection(correct_words)) / len(generated_words.union(correct_words))
            else:
                word_overlap = 0
                
            relevance_score = min(5, max(1, word_overlap * 5))
            
            # Conciseness score - penalize verbosity
            conciseness_penalty = max(0, length_ratio - 1.2)  # Penalty if more than 20% longer
            conciseness_score = min(5, max(1, 5 - conciseness_penalty * 3))
            
            # Calculate overall score
            overall_score = round((semantic_score + completeness_score + relevance_score + conciseness_score) / 4, 2)
            
            # Determine pass/fail threshold
            passed = overall_score >= 4.0
            
            # Generate explanations based on metrics
            reasons = []
            if semantic_score < 3.5:
                reasons.append("The answer's semantic meaning differs significantly from the reference.")
            if completeness_score < 3.5:
                if length_ratio < 0.8:
                    reasons.append("The answer is too short compared to the reference.")
                elif length_ratio > 1.2:
                    reasons.append("The answer is unnecessarily verbose.")
            if relevance_score < 3.5:
                reasons.append("The answer lacks key terms from the reference.")
            if conciseness_score < 3.5:
                reasons.append("The answer could be more concise.")
                
            # Compile evaluation results
            evaluation = {
                "correctness": round(semantic_score, 2),  # Using semantic similarity as correctness
                "completeness": round(completeness_score, 2),
                "relevance": round(relevance_score, 2),
                "conciseness": round(conciseness_score, 2),
                "overall": overall_score,
                "passed": passed,
                "cosine_similarity": round(cosine_similarity, 4),  # Add raw similarity for debugging
                "length_ratio": round(length_ratio, 2),  # Add raw ratio for debugging
                "word_overlap": round(word_overlap, 4),  # Add raw overlap for debugging
                "reasons": "; ".join(reasons) if reasons else "Answer meets quality thresholds."
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error in embedding-based evaluation: {e}")
            return {
                "correctness": 0,
                "completeness": 0, 
                "relevance": 0,
                "conciseness": 0,
                "overall": 0,
                "passed": False,
                "reasons": f"Evaluation error: {str(e)}"
            }
    
    def _compute_cosine_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity value between 0 and 1
        """
        # Convert to numpy arrays if they aren't already
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0  # Handle zero vectors
            
        return dot_product / (norm1 * norm2)
    
    def _compute_batch_cosine_similarity(self, embeddings1, embeddings2):
        """
        Compute cosine similarity between two sets of embeddings in batch mode.
        
        Args:
            embeddings1: First set of embedding vectors (n x dim)
            embeddings2: Second set of embedding vectors (n x dim)
            
        Returns:
            Array of cosine similarity values between corresponding pairs
        """
        # Convert to numpy arrays
        vecs1 = np.array(embeddings1)
        vecs2 = np.array(embeddings2)
        
        # Compute norms for all vectors at once
        norms1 = np.linalg.norm(vecs1, axis=1)
        norms2 = np.linalg.norm(vecs2, axis=1)
        
        # Calculate dot products for corresponding pairs
        dot_products = np.sum(vecs1 * vecs2, axis=1)
        
        # Calculate similarities (handling zero norms)
        denominators = norms1 * norms2
        similarities = np.zeros_like(denominators)
        non_zero_mask = denominators > 0
        similarities[non_zero_mask] = dot_products[non_zero_mask] / denominators[non_zero_mask]
        
        return similarities