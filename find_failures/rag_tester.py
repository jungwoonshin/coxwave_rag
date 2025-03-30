import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

# Import components from your system
from data.loader import DataLoader
from embedding.embedder import TextEmbedder
from llm.openai_model import OpenAIModel
from llm.llama_model import LlamaModel
from rag.retriever import ChromaRetriever
from utils.prompt import PromptBuilder
from find_failures.data_manager import DataManager
from find_failures.reporter import TestResultReporter
from find_failures.failure_analyzer import FailureAnalyzer
from find_failures.bleu_evaluator import BLEUEvaluator, evaluate_batch

from config.setting import (DEFAULT_BATCH_SIZE, DEFAULT_LLM_MODEL, OPENAI_LLM_MODEL_NAME,
    DEFAULT_TEST_SAMPLE_SIZE, EMBEDDING_MODEL_NAME, HF_TOKEN, OPENAI_API_KEY, OUTPUT_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_llm_response(response: str) -> list:
    """
    Parse LLM responses that contain JSON arrays wrapped in markdown code blocks.
    Extracts just the list of questions from responses like:
    ```json
    ["question1", "question2", "question3"]
    ```
    
    Args:
        response: The raw response string from the LLM
        
    Returns:
        List of extracted questions
    """
    # Remove markdown code block indicators and any surrounding whitespace
    cleaned_response = response.strip()
    if cleaned_response.startswith('```'):
        # Find the end of the first line which may contain the language specifier
        first_line_end = cleaned_response.find('\n')
        if first_line_end != -1:
            # Skip the first line which contains ```json or just ```
            start_index = first_line_end + 1
        else:
            # Fallback if there's no newline
            start_index = cleaned_response.find('```') + 3
            
        # Find the closing code block
        end_index = cleaned_response.rfind('```')
        if end_index == -1:  # If no closing block, take the whole string
            end_index = len(cleaned_response)
            
        # Extract just the JSON content
        json_content = cleaned_response[start_index:end_index].strip()
    else:
        # If no code block markers, use the whole string
        json_content = cleaned_response
    
    try:
        # Try to parse as JSON
        import json
        questions = json.loads(json_content)
        if isinstance(questions, list):
            return questions
        else:
            return []
    except json.JSONDecodeError:
        # Fallback: try to extract list items manually
        import re
        # Look for quoted strings within brackets
        pattern = r'"([^"]*)"'
        matches = re.findall(pattern, json_content)
        return matches if matches else []
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
        bleu_threshold: float = 0.6
    ):
        """
        Initialize the tester with the necessary components.
        
        Args:
            data_path: Path to the FAQ data pickle file
            llm_model_name: Name of the LLM to use
            embedding_model_name: Name of the embedding model
            output_dir: Directory to save test results
            test_sample_size: Number of questions to test (None for all)
            bleu_threshold: Threshold for BLEU score evaluation (0.0-1.0)
        """
        self.data_path = data_path
        self.llm_model_name = llm_model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.test_sample_size = test_sample_size
        self.bleu_threshold = bleu_threshold
        
        # Initialize components
        logger.info("Initializing components...")
        self.embedder = TextEmbedder(embedding_model_name, OPENAI_API_KEY)
        self.retriever = ChromaRetriever(embedder=self.embedder)
        self.llm_model = LlamaModel(llm_model_name, HF_TOKEN)
        self.openai_model = OpenAIModel(OPENAI_LLM_MODEL_NAME, OPENAI_API_KEY)
        self.data_manager = DataManager(data_path, test_sample_size)
        
        self.variant_generator = OpenAIModel(embedding_model_name, OPENAI_API_KEY)
        self.reporter = TestResultReporter(OUTPUT_DIR)
        self.failure_analyzer = FailureAnalyzer(self.llm_model)
        
        # Load FAQ data and setup collection
        self.faq_data = self._load_data()
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

        
        logger.info(f"Generating variants for {len(questions)} questions")
        
        for i, question in enumerate(tqdm(questions, desc="Generating question variants")):
            if i % 10 == 0:
                logger.info(f"Generated variants for {i}/{len(questions)} questions")
            
            # Get variants from LLM
            try:
                # prompt = f"Original question: {question}\n\nGenerate {num_variants} different versions of this question that keep the same meaning but use different words, structures, or formality levels. Use Korean language."
                
                prompt = PromptBuilder().get_variant_generation_prompt(2, question)
                
                response = self.openai_model.generate(prompt)
                
                # Keep only up to num_variants
                question_variants = parse_llm_response(response)
                
                # Store variants
                variants[question] = question_variants
                
            except Exception as e:
                logger.warning(f"Error generating variants for question '{question}': {e}")
                variants[question] = []
        
        logger.info(f"Generated variants for all questions")
        return variants
    
    def evaluate_answers_batch(self, generated_answers: List[str], correct_answers: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple generated answers against their correct answers using BLEU score.
        
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
        
        try:
            # Ensure all answers are strings (not generators or other objects)
            generated_answers_str = [str(ans) if not isinstance(ans, str) else ans for ans in generated_answers]
            correct_answers_str = [str(ans) if not isinstance(ans, str) else ans for ans in correct_answers]
            
            # Create a data structure that the evaluate_batch function expects
            evaluation_data = []
            for i in range(len(generated_answers_str)):
                evaluation_data.append({
                    "prompt": f"Question {i+1}",  # Placeholder since we don't need it for evaluation
                    "response_llm_a": correct_answers_str[i],  # Reference/correct answer
                    "response_llm_b": generated_answers_str[i]  # Generated answer to evaluate
                })
            
            # Convert to DataFrame for the evaluate_batch function
            df = pd.DataFrame(evaluation_data)
            
            # Run batch evaluation with BLEU
            results_df = evaluate_batch(df)
            
            # Process results into our expected format
            results = []
            for i, row in results_df.iterrows():
                bleu_score = row['bleu_score']
                
                # Scale BLEU score (0-1) to our 1-5 scale
                scaled_score = 1 + bleu_score * 4  # Maps 0->1, 1->5
                
                # Generate explanations based on BLEU score
                reasons = []
                if bleu_score < 0.5:
                    reasons.append("The answer differs significantly from the reference.")
                elif bleu_score < 0.7:
                    reasons.append("The answer captures some key elements but misses others.")
                
                # Determine pass/fail threshold
                passed = bleu_score >= self.bleu_threshold
                
                # Compile evaluation results
                evaluation = {
                    "correctness": round(scaled_score, 2),  # Using BLEU as correctness
                    "completeness": round(scaled_score, 2),  # Simplifying by using same score
                    "relevance": round(scaled_score, 2),     # Simplifying by using same score
                    "conciseness": round(scaled_score, 2),   # Simplifying by using same score
                    "overall": round(scaled_score, 2),
                    "passed": passed,
                    "bleu_score": round(bleu_score, 4),  # Add raw BLEU for debugging
                    "reasons": "; ".join(reasons) if reasons else "Answer meets quality thresholds."
                }
                
                results.append(evaluation)
            
            return results
        
        except Exception as e:
            logger.error(f"Error in BLEU-based evaluation: {e}")
            
            # Return error results with default values for all pairs
            return [
                {
                    "correctness": 2.5,  # Default middle value
                    "completeness": 2.5, 
                    "relevance": 2.5,
                    "conciseness": 2.5,
                    "overall": 2.5,
                    "passed": False,
                    "reasons": f"Evaluation error: {str(e)}"
                }
                for _ in range(len(generated_answers))
            ]

    def evaluate_answer(self, generated_answer: str, correct_answer: str) -> Dict[str, Any]:
        """
        Evaluate a single generated answer against the correct answer using BLEU score.
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
        # Prepare test questions - limit to 20 as requested
        original_questions = self.data_manager.get_sample_questions()[:5]
        
        test_questions = original_questions.copy()
        variant_map = {q: q for q in original_questions}  # Maps variant questions to original questions
        
        # Generate and add variants if needed
        if include_variants and num_variants > 0:
            logger.info(f"Generating {num_variants} variants for {len(original_questions)} questions")
            variants_dict = self.generate_question_variants(original_questions, num_variants)
            
            # Log original and variant questions together
            variant_log_path = self.output_dir / f"question_variants_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(variant_log_path, 'w', encoding='utf-8') as f:
                for original, variants in variants_dict.items():
                    f.write(f"Original: {original}\n")
                    for i, variant in enumerate(variants):
                        f.write(f"Variant {i+1}: {variant}\n")
                    f.write("\n")
                    
                    # Add variants to test questions
                    for variant in variants:
                        test_questions.append(variant)
                        variant_map[variant] = original
            
            logger.info(f"Total test questions including variants: {len(test_questions)}")
            logger.info(f"Question variants logged to {variant_log_path}")
        
        # Initialize OpenAI model for answer generation
        
        # Initialize results
        results = []
        test_start_time = time.time()
        
        # Process questions in batches
        for batch_start in range(0, len(test_questions), batch_size):
            batch_end = min(batch_start + batch_size, len(test_questions))
            batch_questions = test_questions[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start // batch_size + 1}/{(len(test_questions) + batch_size - 1) // batch_size} ({len(batch_questions)} questions)")
            
            # Process each question in the batch
            batch_results = []
            for i, question in enumerate(tqdm(batch_questions, desc=f"Processing batch {batch_start // batch_size + 1}")):
                original_question = variant_map[question]
                correct_answer = self.faq_data[original_question]
                
                # Track if this is an original or variant question
                is_original = (question == original_question)
                
                # Process start time
                process_start_time = time.time()
                
                # Step 1: Retrieve relevant documents
                retrieved_docs = self.retriever.retrieve(question, top_k=3)
                
                # Step 2: Build prompt with context
                prompt_builder = PromptBuilder()
                context = "Context Information:\n"
                for i, doc in enumerate(retrieved_docs):
                    context += f"Document {i+1}:\n"
                    context += f"Question: {doc['question']}\n"
                    context += f"Answer: {doc['answer']}\n\n"
                
                prompt = prompt_builder.get_rag_prompt(query=question, context=context)
                
                # Step 3: Generate answer using OpenAI model
                try:
                    openai_answer = self.openai_model.generate(prompt)
                except Exception as e:
                    logger.error(f"Error generating OpenAI answer for question '{question}': {e}")
                    openai_answer = "Error: Failed to generate answer."
                
                
                # Calculate process time
                process_time = time.time() - process_start_time
                
                # Compile result
                result = {
                    "test_id": len(results) + len(batch_results),
                    "variant_question": question,
                    "original_question": original_question,
                    "is_variant": not is_original,
                    "retrieved_docs": [
                        {
                            "question": doc["question"],
                            "answer": doc["answer"],
                            "score": doc["score"]
                        }
                        for doc in retrieved_docs
                    ],
                    "correct_answer": correct_answer,
                    "openai_answer": openai_answer,
                    "process_time": process_time
                }
                
                batch_results.append(result)
            
            # Add batch results to overall results
            results.extend(batch_results)
            
            # Save intermediate results after each batch
            intermediate_results_path = self.output_dir / f"rag_test_batch_{batch_start // batch_size + 1}_results.json"
            with open(intermediate_results_path, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Batch {batch_start // batch_size + 1} results saved to {intermediate_results_path}")
        
        # Save OpenAI model outputs 
        model_outputs_path = self.output_dir / f"openai_outputs_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(model_outputs_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"Question: {result['variant_question']}\n")
                f.write(f"Original Question: {result['original_question']}\n")
                f.write(f"Correct Answer: {result['correct_answer']}\n")
                f.write(f"OpenAI Answer: {result['openai_answer']}\n")
                f.write("\n" + "-"*80 + "\n\n")
        
        return None

    def _save_report(self, report: Dict):
        """
        Save the test report to files.
        
        Args:
            report: The test report dictionary
        """
        timestamp = report["timestamp"]
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save full report
        full_report_path = self.output_dir / f"rag_test_report_{timestamp}.json"
        with open(full_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary = {
            "timestamp": report["timestamp"],
            "test_config": report["test_config"],
            "statistics": report["statistics"]
        }
        
        summary_path = self.output_dir / f"rag_test_summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save failure cases separately
        failures = [r for r in report["results"] if not r["evaluation"]["passed"]]
        if failures:
            failures_path = self.output_dir / f"rag_test_failures_{timestamp}.json"
            with open(failures_path, 'w', encoding='utf-8') as f:
                json.dump(failures, f, indent=2, ensure_ascii=False)
        
        # Save successful cases separately
        successes = [r for r in report["results"] if r["evaluation"]["passed"]]
        if successes:
            successes_path = self.output_dir / f"rag_test_successes_{timestamp}.json"
            with open(successes_path, 'w', encoding='utf-8') as f:
                json.dump(successes, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Report saved to {full_report_path}")
        logger.info(f"Summary saved to {summary_path}")
        if failures:
            logger.info(f"Failures saved to {failures_path}")
        if successes:
            logger.info(f"Successes saved to {successes_path}")
    
    def _print_summary(self, report: Dict):
        """
        Print a summary of the test results.
        
        Args:
            report: The test report dictionary
        """
        stats = report["statistics"]
        config = report["test_config"]
        
        print("\n" + "="*80)
        print(f"RAG SYSTEM TEST SUMMARY ({report['timestamp']})")
        print("="*80)
        
        print("\nTEST CONFIGURATION:")
        print(f"- Data: {config['data_path']}")
        print(f"- LLM: {config['llm_model']}")
        print(f"- Embedding Model: {config['embedding_model']}")
        print(f"- Sample Size: {config['test_sample_size'] or 'All'}")
        print(f"- Variants: {'Yes' if config['include_variants'] else 'No'} ({config['num_variants']} per question)")
        
        print("\nRESULTS SUMMARY:")
        print(f"- Total Questions: {stats['total_questions']}")
        print(f"  - Original Questions: {stats['original_questions_count']}")
        print(f"  - Variant Questions: {stats['variant_questions_count']}")
        print(f"- Success Rate: {stats['overall_success_rate']*100:.2f}% ({stats['success_count']}/{stats['total_questions']})")
        print(f"  - Original Questions: {stats['original_success_rate']*100:.2f}%")
        print(f"  - Variant Questions: {stats['variant_success_rate']*100:.2f}%")
        print(f"- Average Processing Time: {stats['avg_process_time']:.2f} seconds")
        
        print("\nFAILURE ANALYSIS:")
        failures = [r for r in report["results"] if not r["evaluation"]["passed"]]
        
        if not failures:
            print("- No failures observed.")
        else:
            failure_modes = {}
            for failure in failures:
                analysis = failure.get("failure_analysis", {})
                mode = analysis.get("primary_failure_mode", "Unknown")
                failure_modes[mode] = failure_modes.get(mode, 0) + 1
            
            print(f"- Failure distribution ({len(failures)} failures):")
            for mode, count in sorted(failure_modes.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(failures) * 100
                print(f"  - {mode}: {percentage:.1f}% ({count}/{len(failures)})")
        
        print("\nREPORT FILES:")
        print(f"- Full Report: {self.output_dir}/rag_test_report_{report['timestamp']}.json")
        print(f"- Summary: {self.output_dir}/rag_test_summary_{report['timestamp']}.json")
        
        if failures:
            print(f"- Failures: {self.output_dir}/rag_test_failures_{report['timestamp']}.json")
        
        print("="*80 + "\n")
    
    def run_evaluation_only(self, questions: List[str], generated_answers: List[str], correct_answers: List[str]) -> Dict:
        """
        Run evaluation only on provided questions and answers.
        Useful for evaluating external RAG systems or comparing different models.
        
        Args:
            questions: List of test questions
            generated_answers: List of answers generated by the system
            correct_answers: List of correct answers
            
        Returns:
            Dictionary with evaluation results
        """
        if len(questions) != len(generated_answers) or len(questions) != len(correct_answers):
            raise ValueError("Number of questions, generated answers, and correct answers must match")
        
        # Evaluate answers
        evaluations = self.evaluate_answers_batch(generated_answers, correct_answers)
        
        # Compile results
        results = []
        for i, (question, generated, correct, evaluation) in enumerate(zip(questions, generated_answers, correct_answers, evaluations)):
            result = {
                "test_id": i,
                "question": question,
                "correct_answer": correct,
                "generated_answer": generated,
                "evaluation": evaluation
            }
            results.append(result)
        
        # Calculate statistics
        total = len(results)
        success_count = sum(1 for r in results if r["evaluation"]["passed"])
        failure_count = total - success_count
        
        statistics = {
            "total_questions": total,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / total if total > 0 else 0,
            "avg_bleu_score": sum(r["evaluation"]["bleu_score"] for r in results) / total if total > 0 else 0,
        }
        
        # Generate the report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        report = {
            "timestamp": timestamp,
            "test_type": "evaluation_only",
            "statistics": statistics,
            "results": results
        }
        
        return report