import nltk
nltk.download('punkt_tab')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union

class BLEUEvaluator:
    """
    Class for evaluating text similarity using BLEU score.
    """
    
    def __init__(self, weights=(0.25, 0.25, 0.25, 0.25)):
        """
        Initialize the BLEU score evaluator with customizable n-gram weights.
        
        Args:
            weights (tuple): Weights for unigrams, bigrams, trigrams, and 4-grams.
                            Default is equal weighting (0.25, 0.25, 0.25, 0.25).
        """
        self.weights = weights
        self.smoothing_function = SmoothingFunction().method1
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text to tokenize.
        
        Returns:
            list: List of tokens (words).
        """
        return nltk.word_tokenize(text.lower())
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """
        Calculate BLEU score between reference and candidate texts.
        
        Args:
            reference (str): Reference text (e.g., correct answer).
            candidate (str): Candidate text (e.g., generated answer).
        
        Returns:
            float: BLEU score from 0 to 1.
        """
        reference_tokens = self.tokenize(reference)
        candidate_tokens = self.tokenize(candidate)
        
        # BLEU requires a list of references (even if we have just one)
        references = [reference_tokens]
        
        # Calculate BLEU score with smoothing to handle zero counts
        score = sentence_bleu(references, candidate_tokens, 
                              weights=self.weights, 
                              smoothing_function=self.smoothing_function)
        
        return score
    
    def evaluate_pair(self, reference: str, candidate: str) -> Dict[str, Any]:
        """
        Evaluate a single pair of reference and candidate texts.
        
        Args:
            reference (str): Reference text (correct answer).
            candidate (str): Candidate text (generated answer).
            
        Returns:
            Dict: Evaluation results with various metrics.
        """
        bleu_score = self.calculate_bleu(reference, candidate)
        
        # Scale BLEU score (0-1) to 1-5 scale
        scaled_score = 1 + bleu_score * 4
        
        # Generate explanations based on BLEU score
        reasons = []
        if bleu_score < 0.5:
            reasons.append("The answer differs significantly from the reference.")
        elif bleu_score < 0.7:
            reasons.append("The answer captures some key elements but misses others.")
            
        # Determine pass/fail threshold
        passed = bleu_score >= 0.6
        
        # Compile evaluation results
        evaluation = {
            "correctness": round(scaled_score, 2),
            "completeness": round(scaled_score, 2),
            "relevance": round(scaled_score, 2),
            "conciseness": round(scaled_score, 2),
            "overall": round(scaled_score, 2),
            "passed": passed,
            "bleu_score": round(bleu_score, 4),
            "reasons": "; ".join(reasons) if reasons else "Answer meets quality thresholds."
        }
        
        return evaluation


def evaluate_batch(data: Union[pd.DataFrame, List[Dict]], output_path: str = None, 
                  weights: tuple = (0.25, 0.25, 0.25, 0.25)) -> pd.DataFrame:
    """
    Evaluate a batch of text pairs using BLEU score.
    
    Args:
        data: DataFrame or list of dictionaries with columns/keys:
              'prompt' (optional), 'response_llm_a' (reference), 'response_llm_b' (candidate)
        output_path: Optional path to save results
        weights: BLEU n-gram weights
        
    Returns:
        DataFrame with original data plus evaluation results
    """
    # Convert list to DataFrame if needed
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Validate required columns
    required_cols = ['response_llm_a', 'response_llm_b']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Initialize evaluator
    evaluator = BLEUEvaluator(weights=weights)
    
    # Calculate BLEU scores
    bleu_scores = []
    for _, row in df.iterrows():
        reference = row['response_llm_a']
        candidate = row['response_llm_b']
        bleu_score = evaluator.calculate_bleu(reference, candidate)
        bleu_scores.append(bleu_score)
    
    # Add scores to DataFrame
    df['bleu_score'] = bleu_scores
    
    # Calculate pass/fail based on threshold
    df['passed'] = df['bleu_score'] >= 0.6
    
    # Add summary statistics
    avg_bleu = np.mean(bleu_scores)
    median_bleu = np.median(bleu_scores)
    std_bleu = np.std(bleu_scores)
    
    print(f"Evaluation complete!")
    print(f"Average BLEU score: {avg_bleu:.4f}")
    print(f"Median BLEU score: {median_bleu:.4f}")
    print(f"Standard deviation: {std_bleu:.4f}")
    print(f"Pass rate: {sum(df['passed']) / len(df) * 100:.2f}%")
    
    # Save results if output path is provided
    if output_path:
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.json'):
            df.to_json(output_path, orient='records', indent=2)
        else:
            # Default to CSV
            df.to_csv(output_path + '.csv', index=False)
            
        print(f"Results saved to {output_path}")
    
    return df