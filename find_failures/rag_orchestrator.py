# rag_orchestrator.py
"""
Main orchestrator class for executing RAG tests with various configurations.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from config.setting import (
    DEFAULT_LLM_MODEL, EMBEDDING_MODEL_NAME, 
    DEFAULT_TEST_SAMPLE_SIZE, DEFAULT_BATCH_SIZE, DEFAULT_NUM_VARIANTS
)
from find_failures.rag_tester import RAGSystemTester

logger = logging.getLogger(__name__)

class RAGTestOrchestrator:
    """
    Orchestrator for running RAG tests with various configurations.
    Designed for testing different LLM models, variants, batch sizes, etc.
    """
    
    def __init__(self, data_path: str, output_dir: str = "test_results"):
        """
        Initialize the orchestrator.
        
        Args:
            data_path: Path to the FAQ data pickle file
            output_dir: Directory to save test results
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
    
    def run_model_comparison(self, 
                            model_names: List[str],
                            sample_size: int = DEFAULT_TEST_SAMPLE_SIZE,
                            include_variants: bool = True,
                            num_variants: int = DEFAULT_NUM_VARIANTS,
                            batch_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, Any]:
        """
        Run tests on multiple LLM models for comparison.
        
        Args:
            model_names: List of model names to test
            sample_size: Number of questions to sample
            include_variants: Whether to include question variants
            num_variants: Number of variants per question
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with comparative results
        """
        model_results = {}
        
        for model_name in model_names:
            logger.info(f"Testing model: {model_name}")
            
            # Initialize the tester with this model
            tester = RAGSystemTester(
                data_path=self.data_path,
                llm_model_name=model_name,
                embedding_model_name=EMBEDDING_MODEL_NAME,
                output_dir=str(self.output_dir),
                test_sample_size=sample_size
            )
            
            # Run the test
            start_time = time.time()
            report = tester.run_test(
                include_variants=include_variants,
                num_variants=num_variants,
                batch_size=batch_size
            )
            total_time = time.time() - start_time
            
            # Store the results
            model_results[model_name] = {
                "success_rate": report["statistics"]["success_rate"],
                "total_tests": report["statistics"]["total_tests"],
                "success_count": report["statistics"]["success_count"],
                "failure_count": report["statistics"]["failure_count"],
                "processing_time": total_time,
                "avg_time_per_test": total_time / report["statistics"]["total_tests"],
                "variant_vs_original": report["statistics"].get("variant_vs_original", {}),
                "failure_types": report["statistics"].get("failure_types", {})
            }
            
            # Save individual report reference
            model_results[model_name]["report_path"] = str(self.output_dir / f"rag_test_{model_name.split('/')[-1]}_report_{time.strftime('%Y%m%d_%H%M%S')}.json")
        
        # Generate comparative analysis
        comparison = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_config": {
                "data_path": self.data_path,
                "sample_size": sample_size,
                "include_variants": include_variants,
                "num_variants": num_variants,
                "batch_size": batch_size,
                "models_tested": model_names
            },
            "model_results": model_results,
            "ranking": sorted(
                [(model, results["success_rate"]) for model, results in model_results.items()],
                key=lambda x: x[1],
                reverse=True
            )
        }
        
        # Save comparison report
        comparison_path = self.output_dir / f"rag_model_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Model comparison saved to: {comparison_path}")
        
        # Print comparison summary
        print("\n=== Model Comparison Summary ===")
        print(f"Models tested: {', '.join(model_names)}")
        print("\nSuccess Rates:")
        for model, rate in comparison["ranking"]:
            print(f"  {model}: {rate*100:.2f}%")
        
        print("\nAverage Time Per Test (seconds):")
        for model in model_names:
            print(f"  {model}: {model_results[model]['avg_time_per_test']:.2f}")
        
        self.results.append(comparison)
        return comparison
    
    def run_batch_size_experiment(self,
                                 model_name: str = DEFAULT_LLM_MODEL,
                                 batch_sizes: List[int] = [1, 5, 10, 20, 50],
                                 sample_size: int = DEFAULT_TEST_SAMPLE_SIZE) -> Dict[str, Any]:
        """
        Run tests with different batch sizes to find optimal performance.
        
        Args:
            model_name: LLM model to use
            batch_sizes: List of batch sizes to test
            sample_size: Number of questions to sample
            
        Returns:
            Dictionary with batch size experiment results
        """
        batch_results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Initialize the tester
            tester = RAGSystemTester(
                data_path=self.data_path,
                llm_model_name=model_name,
                output_dir=str(self.output_dir),
                test_sample_size=sample_size
            )
            
            # Run the test with this batch size
            start_time = time.time()
            report = tester.run_test(
                include_variants=False,  # Simplify by not using variants
                batch_size=batch_size
            )
            total_time = time.time() - start_time
            
            # Store the results
            batch_results[batch_size] = {
                "success_rate": report["statistics"]["success_rate"],
                "total_tests": report["statistics"]["total_tests"],
                "processing_time": total_time,
                "avg_time_per_test": total_time / report["statistics"]["total_tests"],
                "avg_time_per_batch": total_time / (report["statistics"]["total_tests"] / batch_size)
            }
        
        # Generate experiment results
        experiment = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_config": {
                "data_path": self.data_path,
                "model": model_name,
                "sample_size": sample_size,
                "batch_sizes_tested": batch_sizes
            },
            "batch_results": batch_results,
            "optimal_batch_size": min(
                [(size, results["avg_time_per_test"]) for size, results in batch_results.items()],
                key=lambda x: x[1]
            )[0]
        }
        
        # Save experiment report
        experiment_path = self.output_dir / f"rag_batch_experiment_{time.strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(experiment_path, 'w') as f:
            json.dump(experiment, f, indent=2)
        
        logger.info(f"Batch size experiment saved to: {experiment_path}")
        
        # Print experiment summary
        print("\n=== Batch Size Experiment Summary ===")
        print(f"Model tested: {model_name}")
        print(f"Optimal batch size: {experiment['optimal_batch_size']}")
        
        print("\nPerformance by Batch Size:")
        for size in batch_sizes:
            print(f"  Batch size {size}: {batch_results[size]['avg_time_per_test']:.4f} sec/test")
        
        self.results.append(experiment)
        return experiment
    
    def run_variant_sensitivity_experiment(self,
                                         model_name: str = DEFAULT_LLM_MODEL,
                                         num_variants_list: List[int] = [1, 3, 5, 10],
                                         sample_size: int = DEFAULT_TEST_SAMPLE_SIZE,
                                         batch_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, Any]:
        """
        Run tests with different numbers of variants to assess model robustness.
        
        Args:
            model_name: LLM model to use
            num_variants_list: List of variant counts to test
            sample_size: Number of questions to sample
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with variant sensitivity results
        """
        variant_results = {}
        
        for num_variants in num_variants_list:
            logger.info(f"Testing with {num_variants} variants per question")
            
            # Initialize the tester
            tester = RAGSystemTester(
                data_path=self.data_path,
                llm_model_name=model_name,
                output_dir=str(self.output_dir),
                test_sample_size=sample_size
            )
            
            # Run the test with this number of variants
            report = tester.run_test(
                include_variants=True,
                num_variants=num_variants,
                batch_size=batch_size
            )
            
            # Extract variant vs. original stats
            var_stats = report["statistics"].get("variant_vs_original", {})
            
            # Store the results
            variant_results[num_variants] = {
                "overall_success_rate": report["statistics"]["success_rate"],
                "original_success_rate": var_stats.get("original_success_rate", 0),
                "variant_success_rate": var_stats.get("variant_success_rate", 0),
                "success_rate_difference": var_stats.get("success_rate_difference", 0),
                "total_tests": report["statistics"]["total_tests"],
                "original_count": var_stats.get("original_count", 0),
                "variant_count": var_stats.get("variant_count", 0)
            }
        
        # Generate experiment results
        experiment = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_config": {
                "data_path": self.data_path,
                "model": model_name,
                "sample_size": sample_size,
                "batch_size": batch_size,
                "num_variants_tested": num_variants_list
            },
            "variant_results": variant_results,
            "robustness_score": min(
                [abs(results["success_rate_difference"]) for _, results in variant_results.items()]
            ),
            "most_sensitive_variant_count": max(
                [(count, abs(results["success_rate_difference"])) for count, results in variant_results.items()],
                key=lambda x: x[1]
            )[0]
        }
        
        # Save experiment report
        experiment_path = self.output_dir / f"rag_variant_sensitivity_{time.strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(experiment_path, 'w') as f:
            json.dump(experiment, f, indent=2)
        
        logger.info(f"Variant sensitivity experiment saved to: {experiment_path}")
        
        # Print experiment summary
        print("\n=== Variant Sensitivity Experiment Summary ===")
        print(f"Model tested: {model_name}")
        print(f"Robustness score: {experiment['robustness_score']:.4f}")
        print(f"Most sensitive at: {experiment['most_sensitive_variant_count']} variants")
        
        print("\nSuccess Rate Gap (Original - Variant):")
        for count in num_variants_list:
            diff = variant_results[count]["success_rate_difference"]
            print(f"  {count} variants: {diff*100:.2f}% ({'more robust' if diff < 0 else 'less robust'})")
        
        self.results.append(experiment)
        return experiment