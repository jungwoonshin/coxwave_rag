import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

# Import components from your system
from data.loader import DataLoader
from embedding.embedder import TextEmbedder
from llm.openai_model import OpenAIModel
from rag.retriever import ChromaRetriever
from utils.prompt import PromptBuilder
from find_failures.data_manager import DataManager
from llm.llama_model import LlamaModel
from find_failures.embedding_evaluator import EmbeddingEvaluator
from find_failures.reporter import TestResultReporter

from tqdm import tqdm
from config.setting import (DEFAULT_BATCH_SIZE, DEFAULT_LLM_MODEL, DEFAULT_NUM_VARIANTS,
    DEFAULT_TEST_SAMPLE_SIZE, EMBEDDING_MODEL_NAME, HF_TOKEN, OPENAI_API_KEY, OUTPUT_DIR)
from find_failures.failure_analyzer import FailureAnalyzer
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGSystemTester:
    """
    Class for testing and analyzing RAG system failures on FAQ data.
    """
    def __init__(
        self,
        data_path: str,
        llm_model_name: str = DEFAULT_LLM_MODEL,
        embedding_model_name: str = "text-embedding-3-small",
        output_dir: str = "test_results",
        test_sample_size: int = None,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize the tester with the necessary components.
        
        Args:
            data_path: Path to the FAQ data pickle file
            llm_model_name: Name of the LLM to use
            embedding_model_name: Name of the embedding model
            output_dir: Directory to save test results
            test_sample_size: Number of questions to test (None for all)
            similarity_threshold: Threshold for semantic similarity evaluation
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.test_sample_size = test_sample_size
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        logger.info("Initializing components...")
        self.embedder = TextEmbedder(embedding_model_name, OPENAI_API_KEY)
        self.retriever = ChromaRetriever(embedder=self.embedder)
        self.llm_model = LlamaModel(llm_model_name, HF_TOKEN)
        self.data_manager = DataManager(data_path, test_sample_size)
        self.evaluator = EmbeddingEvaluator(self.embedder)
        self.variant_generator = OpenAIModel(embedding_model_name, OPENAI_API_KEY)
        self.reporter = TestResultReporter(OUTPUT_DIR)  # Placeholder for the reporter component
        self.failure_analyzer = FailureAnalyzer(self.llm_model)  # Placeholder for the failure analysis component
        
        # Load FAQ data
        self.faq_data = self._load_data()
        
        # Setup the collection with FAQ data
        self._setup_collection()
        
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
    
    def _setup_collection(self):
        """
        Set up the retriever collection with FAQ data.
        """
        qa_pairs = []
        for i, (question, answer) in enumerate(self.faq_data.items()):
            qa_pairs.append({
                "id": i,
                "question": question,
                "answer": answer
            })
        
        logger.info(f"Setting up retriever with {len(qa_pairs)} QA pairs")
        self.retriever.setup_collection(qa_pairs)
        logger.info(f"Retriever collection setup complete")
    
    def generate_question_variants(self, questions: List[str], num_variants: int = 3) -> Dict[str, List[str]]:
        """
        Generate variants of the original questions to test robustness.
        
        Args:
            questions: List of original questions
            num_variants: Number of variants to generate per question
            
        Returns:
            Dictionary mapping original questions to their variants
        """
        variants = {}
        
        # System message for the LLM to generate variants
        system_message = """
        Generate {num_variants} different versions of the given question. 
        The variants should:
        0. Use Korean
        1. Keep the same meaning but use different wording
        2. Vary in formality (formal, casual, etc.)
        3. Include some with different sentence structures
        4. Potentially include common typing mistakes or colloquial expressions
        
        Return only a JSON array of strings with the variant questions.
        """
        
        logger.info(f"Generating variants for {len(questions)} questions")
        
        for i, question in enumerate(questions):
            if i % 10 == 0:
                logger.info(f"Generated variants for {i}/{len(questions)} questions")
            
            # Get variants from LLM
            try:
                prompt = f"Original question: {question}\n\nGenerate {num_variants} different versions of this question that keep the same meaning but use different words, structures, or formality levels."
                
                response = self.llm_model.generate_with_system_message(
                    query=prompt,
                    system_message=system_message.format(num_variants=num_variants),
                    temperature=0.7
                )
                
                # Parse the response
                try:
                    # Try to parse as a JSON array
                    question_variants = json.loads(response)
                    if not isinstance(question_variants, list):
                        raise ValueError("Response is not a list")
                except (json.JSONDecodeError, ValueError):
                    # Fallback to parsing line by line
                    question_variants = []
                    for line in response.split("\n"):
                        line = line.strip()
                        if line and not line.startswith("[") and not line.startswith("]"):
                            # Remove any numbering or quotes
                            line = line.lstrip("0123456789. \"'")
                            line = line.rstrip("\"'")
                            if line:
                                question_variants.append(line)
                
                # Keep only up to num_variants
                question_variants = question_variants[:num_variants]
                
                # Store variants
                variants[question] = question_variants
                
            except Exception as e:
                logger.warning(f"Error generating variants for question '{question}': {e}")
                variants[question] = []
        
        logger.info(f"Generated variants for all questions")
        return variants
    
    def evaluate_answers_batch(self, generated_answers: List[str], correct_answers: List[str]) -> List[Dict[str, Any]]:
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
            all_embeddings = self.embedder.get_embeddings(all_texts)
            
            # Split embeddings back into generated and correct
            num_answers = len(generated_answers)
            generated_embeddings = all_embeddings[:num_answers]
            correct_embeddings = all_embeddings[num_answers:]
            
            # Process each pair
            results = []
            for i in range(num_answers):
                generated_answer = generated_answers[i]
                correct_answer = correct_answers[i]
                generated_embedding = generated_embeddings[i]
                correct_embedding = correct_embeddings[i]
                
                # Process individual evaluation
                results.append(self._evaluate_single_answer(
                    generated_answer,
                    correct_answer,
                    generated_embedding,
                    correct_embedding
                ))
            
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

    def evaluate_answer(self, generated_answer: str, correct_answer: str) -> Dict[str, Any]:
        """
        Evaluate a single generated answer against the correct answer using embedding-based similarity.
        This is a convenience wrapper around the batch method for backward compatibility.
        
        Args:
            generated_answer: Answer generated by the RAG system
            correct_answer: The correct answer from the FAQ
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = self.evaluate_answers_batch([generated_answer], [correct_answer])
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
            
            # Calculate semantic similarity score (0-5 scale to match previous evaluation)
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
        import numpy as np

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
        import numpy as np

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
    
    def analyze_failure(self, question: str, retrieved_docs: List[Dict], correct_answer: str, 
                      generated_answer: str, evaluation: Dict) -> Dict:
        """
        Analyze why the system failed to answer correctly.
        
        Args:
            question: The test question
            retrieved_docs: The documents retrieved by the RAG system
            correct_answer: The correct answer from the FAQ
            generated_answer: The answer generated by the LLM
            evaluation: The evaluation results
            
        Returns:
            Dictionary with failure analysis
        """
        # Check if correct answer was in retrieved documents
        correct_in_retrieved = False
        retrieval_rank = -1
        
        for i, doc in enumerate(retrieved_docs):
            if doc["answer"] == correct_answer:
                correct_in_retrieved = True
                retrieval_rank = i
                break
        
        # Use LLM to analyze the failure
        system_message = """
        You are an expert in diagnosing failures in RAG (Retrieval Augmented Generation) systems.
        Analyze why the system failed to generate the correct answer based on the provided information.
        
        Consider the following possible failure modes:
        1. Retrieval Failure: The system didn't retrieve documents containing the correct answer
        2. Relevance Failure: The system retrieved documents but ranked them poorly
        3. Context Utilization Failure: The LLM didn't effectively use the retrieved context
        4. Hallucination: The LLM generated information not present in the context
        5. Other Failure: Any other failure mode you can identify
        
        Return your analysis as a JSON object with the following fields:
        - primary_failure_mode: one of the failure modes listed above
        - explanation: detailed explanation of what went wrong
        - suggestions: suggestions for improving system performance
        """
        
        # Format retrieved documents for the prompt
        retrieved_context = ""
        for i, doc in enumerate(retrieved_docs):
            retrieved_context += f"Document {i+1}:\n"
            retrieved_context += f"Question: {doc['question']}\n"
            retrieved_context += f"Answer: {doc['answer']}\n"
            retrieved_context += f"Score: {doc['score']}\n\n"
        
        query = f"""
        Test Question: {question}
        
        Retrieved Documents:
        {retrieved_context}
        
        Correct Answer: {correct_answer}
        
        Generated Answer: {generated_answer}
        
        Evaluation:
        Correctness: {evaluation['correctness']}
        Completeness: {evaluation['completeness']}
        Relevance: {evaluation['relevance']}
        Conciseness: {evaluation['conciseness']}
        Overall: {evaluation['overall']}
        Passed: {evaluation['passed']}
        
        Additional Information:
        Was correct answer in retrieved documents: {"Yes" if correct_in_retrieved else "No"}
        If yes, rank of correct document: {retrieval_rank if retrieval_rank >= 0 else "N/A"}
        
        Provide your analysis in JSON format.
        """
        
        try:
            response = self.llm_model.generate_with_system_message(
                query=query,
                system_message=system_message,
                temperature=0.3
            )
            
            # Parse the JSON response
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                # Fallback with basic info
                analysis = {
                    "primary_failure_mode": "Retrieval Failure" if not correct_in_retrieved else "Context Utilization Failure",
                    "explanation": "Failed to parse analysis response. Raw response: " + response[:100] + "...",
                    "suggestions": "Improve the retrieval system to better match questions to answers."
                }
            
            # Add retrieval stats
            analysis["correct_in_retrieved"] = correct_in_retrieved
            analysis["retrieval_rank"] = retrieval_rank if retrieval_rank >= 0 else None
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing failure: {e}")
            return {
                "primary_failure_mode": "Analysis Error",
                "explanation": f"Error during failure analysis: {str(e)}",
                "suggestions": "Check system logs for error details.",
                "correct_in_retrieved": correct_in_retrieved,
                "retrieval_rank": retrieval_rank if retrieval_rank >= 0 else None
            }
    
    def run_test(self, include_variants: bool = True, num_variants: int = 2, batch_size: int = 20) -> Dict:
        """
        Run the test on the FAQ dataset using batch processing.
        
        Args:
            include_variants: Whether to include question variants in the test
            num_variants: Number of variants to generate per question
            batch_size: Number of questions to process in each batch
            
        Returns:
            Dictionary with test results
        """
        # Prepare test questions
        original_questions = self.data_manager.get_sample_questions()
        
        test_questions = []
        variant_map = {}  # Maps variant questions to original questions
        
        # Add original questions to test set
        for question in original_questions:
            test_questions.append(question)
            variant_map[question] = question  # Map to itself
        
        # Generate and add variants if needed
        if include_variants:
            logger.info(f"Generating {num_variants} variants for {len(original_questions)} questions")
            variants_dict = self.variant_generator.generate_variants(original_questions, num_variants)
            
            for original, variants in tqdm(variants_dict.items(), total=len(variants_dict), desc="Generating variants"):
                for variant in variants:
                    test_questions.append(variant)
                    variant_map[variant] = original
            
            logger.info(f"Total test questions including variants: {len(test_questions)}")
        
        # Run tests in batches
        results = []
        success_count = 0
        failure_count = 0
        
        logger.info(f"Starting test with {len(test_questions)} questions in batches of {batch_size}")
        
        # Process questions in batches
        for batch_start in range(0, len(test_questions), batch_size):
            batch_end = min(batch_start + batch_size, len(test_questions))
            batch_questions = test_questions[batch_start:batch_end]
            batch_size_actual = len(batch_questions)
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: questions {batch_start+1}-{batch_end} of {len(test_questions)}")
            
            # Prepare batch data
            batch_original_questions = [variant_map[q] for q in batch_questions]
            batch_correct_answers = [self.data_manager.get_answer_for_question(q) for q in batch_original_questions]
            batch_is_variant = [q != variant_map[q] for q in batch_questions]
            
            # Track timing for entire batch
            batch_start_time = time.time()
            
            # 1. Retrieve documents (still done individually for now)
            batch_retrieved_docs = []
            for question in batch_questions:
                retrieved_docs = self.retriever.retrieve(question, top_k=3)
                batch_retrieved_docs.append(retrieved_docs)
            logger.info(f"Retrieved documents for batch {batch_start//batch_size + 1}")

            # 2. Generate answers (individually for each question)
            prompt_template = PromptBuilder.get_rag_prompt(with_followup=False)
            batch_generated_answers = []
            logger.info(f'len(batch_questions): {len(batch_questions)}')
            for i, question in enumerate(tqdm(batch_questions, total=len(batch_questions), desc="Generating answers")):
                generated_answer = self.llm_model.generate_rag_response(
                    query=question,
                    retrieved_docs=batch_retrieved_docs[i],
                    prompt_template=prompt_template,
                    stream=False
                )
                batch_generated_answers.append(generated_answer)
            
            logger.info(f"Generated answers for batch {batch_start//batch_size + 1}")
            # 3. Evaluate answers in batch
            batch_evaluations = self.evaluator.evaluate_batch(
                batch_generated_answers, 
                batch_correct_answers
            )
            
            logger.info(f"Evaluated answers for batch {batch_start//batch_size + 1}")
            # Total processing time for the batch
            batch_process_time = time.time() - batch_start_time
            # Estimate individual times (distribute evenly across the batch)
            individual_process_times = [batch_process_time / batch_size_actual] * batch_size_actual
            
            # 4. Analyze failures
            batch_analyses = [None] * batch_size_actual
            failed_indices = [i for i, eval_result in enumerate(batch_evaluations) if not eval_result["passed"]]
            
            if failed_indices:
                for i in failed_indices:
                    batch_analyses[i] = self.failure_analyzer.analyze(
                        question=batch_questions[i],
                        retrieved_docs=batch_retrieved_docs[i],
                        correct_answer=batch_correct_answers[i],
                        generated_answer=batch_generated_answers[i],
                        evaluation=batch_evaluations[i]
                    )
            
            # 5. Record results for this batch
            for i in range(batch_size_actual):
                global_idx = batch_start + i
                result = {
                    "test_id": global_idx,
                    "question": batch_questions[i],
                    "original_question": batch_original_questions[i],
                    "is_variant": batch_is_variant[i],
                    "correct_answer": batch_correct_answers[i],
                    "generated_answer": batch_generated_answers[i],
                    "top_retrieved_questions": [doc["question"] for doc in batch_retrieved_docs[i][:3]],
                    "top_retrieved_answers": [doc["answer"] for doc in batch_retrieved_docs[i][:3]],
                    "top_retrieved_scores": [doc["score"] for doc in batch_retrieved_docs[i][:3]],
                    "process_time_seconds": individual_process_times[i],
                    "evaluation": batch_evaluations[i],
                    "failure_analysis": batch_analyses[i]
                }
                
                results.append(result)
                
                # Count successes and failures
                if batch_evaluations[i]["passed"]:
                    success_count += 1
                else:
                    failure_count += 1
        
        # Generate test configuration dictionary
        test_config = {
            "data_path": self.data_path,
            "llm_model": DEFAULT_LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "test_sample_size": self.test_sample_size,
            "include_variants": include_variants,
            "num_variants": num_variants,
            "batch_size": batch_size,
            "device": self.llm_model.device
        }
        
        # Generate the report
        report = self.reporter.generate_report(results, test_config)
        
        # Save the report
        self.reporter.save_report(report, self.llm_model_name)
        
        # Print summary
        self.reporter.print_summary(report)
        
        return report
    
    def _save_report(self, report: Dict):
        """
        Save the test report to files.
        
        Args:
            report: The test report dictionary
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save full report
        full_report_path = self.output_dir / f"rag_test_report_{timestamp}.json"
        with open(full_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save summary
        summary = {
            "timestamp": report["timestamp"],
            "test_config": report["test_config"],
            "statistics": report["statistics"]
        }
        
        summary_path = self.output_dir / f"rag_test_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save failure cases separately
        failures = [r for r in report["results"] if not r["evaluation"]["passed"]]
        if failures:
            failures_path = self.output_dir / f"rag_test_failures_{timestamp}.json"
            with open(failures_path, 'w') as f:
                json.dump(failures, f, indent=2)
        
        logger.info(f"Report saved to {full_report_path}")
        logger.info(f"Summary saved to {summary_path}")
        if failures:
            logger.info(f"Failures saved to {failures_path}")
