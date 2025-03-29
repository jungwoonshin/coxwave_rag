import json
import logging
import time
from pathlib import Path
from typing import Dict, List

# Import components from your system
from config.setting import HF_TOKEN, OPENAI_API_KEY
from embedding.embedder import TextEmbedder
from llm.llama_model import LlamaModel
from rag.retriever import ChromaRetriever
from utils.prompt import PromptBuilder
from find_failures.data_manager import DataManager
from find_failures.variant_generator import VariantGenerator
from find_failures.embedding_evaluator import EmbeddingEvaluator
from find_failures.failure_analyzer import FailureAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestResultReporter:
    """
    Class for generating, saving, and summarizing test results.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the reporter.

        Args:
            output_dir: Directory to save test results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def generate_report(self, results: List[Dict], test_config: Dict) -> Dict:
        """
        Generate a full test report.

        Args:
            results: List of test results
            test_config: Test configuration parameters

        Returns:
            Report dictionary
        """
        # Calculate aggregate statistics
        total_tests = len(results)
        success_count = sum(1 for r in results if r["evaluation"]["passed"])
        failure_count = total_tests - success_count
        success_rate = success_count / total_tests if total_tests > 0 else 0

        # Group failures by type
        failure_types = {}
        for result in results:
            if result["failure_analysis"]:
                failure_mode = result["failure_analysis"]["primary_failure_mode"]
                failure_types[failure_mode] = failure_types.get(failure_mode, 0) + 1

        # Sort failure types by frequency
        sorted_failure_types = sorted(
            [(mode, count) for mode, count in failure_types.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Calculate metrics for variants vs. original questions
        variant_results = [r for r in results if r["is_variant"]]
        original_results = [r for r in results if not r["is_variant"]]

        variant_success_rate = (
            sum(1 for r in variant_results if r["evaluation"]["passed"])
            / len(variant_results)
            if variant_results
            else 0
        )
        original_success_rate = (
            sum(1 for r in original_results if r["evaluation"]["passed"])
            / len(original_results)
            if original_results
            else 0
        )

        # Compile statistics
        statistics = {
            "total_tests": total_tests,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_rate,
            "failure_types": {mode: count for mode, count in sorted_failure_types},
            "variant_vs_original": {
                "variant_count": len(variant_results),
                "original_count": len(original_results),
                "variant_success_rate": variant_success_rate,
                "original_success_rate": original_success_rate,
                "success_rate_difference": original_success_rate - variant_success_rate,
            },
        }

        # Compile final report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_config": test_config,
            "statistics": statistics,
            "results": results,
        }

        return report

    def save_report(self, report: Dict, model_name: str) -> Dict[str, Path]:
        """
        Save the test report to files.

        Args:
            report: The test report dictionary
            model_name: Name of the model used for testing

        Returns:
            Dictionary mapping file types to file paths
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name_short = model_name.split("/")[-1]  # Extract model name for filename

        # Save full report
        full_report_path = (
            self.output_dir / f"rag_test_{model_name_short}_report_{timestamp}.json"
        )
        with open(full_report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Save summary
        summary = {
            "timestamp": report["timestamp"],
            "test_config": report["test_config"],
            "statistics": report["statistics"],
        }

        summary_path = (
            self.output_dir / f"rag_test_{model_name_short}_summary_{timestamp}.json"
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save failure cases separately
        failures = [r for r in report["results"] if not r["evaluation"]["passed"]]
        failures_path = None
        if failures:
            failures_path = (
                self.output_dir
                / f"rag_test_{model_name_short}_failures_{timestamp}.json"
            )
            with open(failures_path, "w") as f:
                json.dump(failures, f, indent=2)

        logger.info(f"Report saved to {full_report_path}")
        logger.info(f"Summary saved to {summary_path}")
        if failures_path:
            logger.info(f"Failures saved to {failures_path}")

        file_paths = {"full_report": full_report_path, "summary": summary_path}
        if failures_path:
            file_paths["failures"] = failures_path

        return file_paths

    def print_summary(self, report: Dict) -> None:
        """
        Print a summary of the test results to console.

        Args:
            report: The test report dictionary
        """
        print("\n=== Test Results Summary ===")
        print(f"Total tests: {report['statistics']['total_tests']}")
        print(f"Success rate: {report['statistics']['success_rate']*100:.2f}%")

        # Print comparison between original and variant questions if applicable
        if report["statistics"]["variant_vs_original"]["variant_count"] > 0:
            var_stats = report["statistics"]["variant_vs_original"]
            print("\n=== Original vs Variant Questions ===")
            print(
                f"Original questions success rate: {var_stats['original_success_rate']*100:.2f}%"
            )
            print(
                f"Variant questions success rate: {var_stats['variant_success_rate']*100:.2f}%"
            )
            print(
                f"Success rate difference: {abs(var_stats['success_rate_difference'])*100:.2f}%"
            )

        # Print top failure modes
        if report["statistics"]["failure_count"] > 0:
            print("\n=== Top Failure Modes ===")
            for mode, count in report["statistics"]["failure_types"].items():
                print(
                    f"{mode}: {count} failures ({count/report['statistics']['failure_count']*100:.2f}%)"
                )


class RAGSystemTester:
    """
    Main class for testing and analyzing RAG system failures on FAQ data using Llama models.
    """

    def __init__(
        self,
        data_path: str,
        llm_model_name: str = "meta-llama/Llama-3.2-8B-Instruct",
        embedding_model_name: str = "text-embedding-3-small",
        output_dir: str = "test_results",
        test_sample_size: int = None,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize the tester with the necessary components.

        Args:
            data_path: Path to the FAQ data pickle file
            llm_model_name: Name of the Llama model to use
            embedding_model_name: Name of the embedding model
            output_dir: Directory to save test results
            test_sample_size: Number of questions to test (None for all)
            similarity_threshold: Threshold for semantic similarity evaluation
        """
        logger.info("Initializing RAG System Tester components...")

        # Initialize models and components
        self.embedder = TextEmbedder(embedding_model_name, OPENAI_API_KEY)
        self.retriever = ChromaRetriever(embedder=self.embedder)
        self.llm_model = LlamaModel(llm_model_name, HF_TOKEN)

        # Initialize specialized classes
        self.data_manager = DataManager(data_path, test_sample_size)
        self.variant_generator = VariantGenerator(self.llm_model)
        self.evaluator = EmbeddingEvaluator(self.embedder)
        self.failure_analyzer = FailureAnalyzer(self.llm_model)
        self.reporter = TestResultReporter(output_dir)

        # Set up the retriever collection
        self.data_manager.setup_collection(self.retriever)

        # Store configuration
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.data_path = data_path
        self.test_sample_size = test_sample_size
        self.similarity_threshold = similarity_threshold

        logger.info("RAG System Tester initialization complete")

    def run_test(
        self, include_variants: bool = True, num_variants: int = 2, batch_size: int = 20
    ) -> Dict:
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
            logger.info(
                f"Generating {num_variants} variants for {len(original_questions)} questions"
            )
            variants_dict = self.variant_generator.generate_variants(
                original_questions, num_variants
            )

            for original, variants in variants_dict.items():
                for variant in variants:
                    test_questions.append(variant)
                    variant_map[variant] = original

            logger.info(
                f"Total test questions including variants: {len(test_questions)}"
            )

        # Run tests in batches
        results = []
        success_count = 0
        failure_count = 0

        logger.info(
            f"Starting test with {len(test_questions)} questions in batches of {batch_size}"
        )

        # Process questions in batches
        for batch_start in range(0, len(test_questions), batch_size):
            batch_end = min(batch_start + batch_size, len(test_questions))
            batch_questions = test_questions[batch_start:batch_end]
            batch_size_actual = len(batch_questions)

            logger.info(
                f"Processing batch {batch_start//batch_size + 1}: questions {batch_start+1}-{batch_end} of {len(test_questions)}"
            )

            # Prepare batch data
            batch_original_questions = [variant_map[q] for q in batch_questions]
            batch_correct_answers = [
                self.data_manager.get_answer_for_question(q)
                for q in batch_original_questions
            ]
            batch_is_variant = [q != variant_map[q] for q in batch_questions]

            # Track timing for entire batch
            batch_start_time = time.time()

            # 1. Retrieve documents (still done individually for now)
            batch_retrieved_docs = []
            for question in batch_questions:
                retrieved_docs = self.retriever.retrieve(question, top_k=3)
                batch_retrieved_docs.append(retrieved_docs)

            # 2. Generate answers (individually for each question)
            prompt_template = PromptBuilder.get_rag_prompt(with_followup=False)
            batch_generated_answers = []
            for i, question in enumerate(batch_questions):
                generated_answer = self.llm_model.generate_rag_response(
                    query=question,
                    retrieved_docs=batch_retrieved_docs[i],
                    prompt_template=prompt_template,
                    stream=False,
                )
                batch_generated_answers.append(generated_answer)

            # 3. Evaluate answers in batch
            batch_evaluations = self.evaluator.evaluate_batch(
                batch_generated_answers, batch_correct_answers
            )

            # Total processing time for the batch
            batch_process_time = time.time() - batch_start_time
            # Estimate individual times (distribute evenly across the batch)
            individual_process_times = [
                batch_process_time / batch_size_actual
            ] * batch_size_actual

            # 4. Analyze failures
            batch_analyses = [None] * batch_size_actual
            failed_indices = [
                i
                for i, eval_result in enumerate(batch_evaluations)
                if not eval_result["passed"]
            ]

            if failed_indices:
                for i in failed_indices:
                    batch_analyses[i] = self.failure_analyzer.analyze(
                        question=batch_questions[i],
                        retrieved_docs=batch_retrieved_docs[i],
                        correct_answer=batch_correct_answers[i],
                        generated_answer=batch_generated_answers[i],
                        evaluation=batch_evaluations[i],
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
                    "top_retrieved_questions": [
                        doc["question"] for doc in batch_retrieved_docs[i][:3]
                    ],
                    "top_retrieved_answers": [
                        doc["answer"] for doc in batch_retrieved_docs[i][:3]
                    ],
                    "top_retrieved_scores": [
                        doc["score"] for doc in batch_retrieved_docs[i][:3]
                    ],
                    "process_time_seconds": individual_process_times[i],
                    "evaluation": batch_evaluations[i],
                    "failure_analysis": batch_analyses[i],
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
            "llm_model": self.llm_model_name,
            "embedding_model": self.embedding_model_name,
            "test_sample_size": self.test_sample_size,
            "include_variants": include_variants,
            "num_variants": num_variants,
            "batch_size": batch_size,
            "device": self.llm_model.device,
        }

        # Generate the report
        report = self.reporter.generate_report(results, test_config)

        # Save the report
        self.reporter.save_report(report, self.llm_model_name)

        # Print summary
        self.reporter.print_summary(report)

        return report
