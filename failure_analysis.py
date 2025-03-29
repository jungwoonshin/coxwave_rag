# main.py
# Main script to run RAG system testing with refactored architecture

import argparse
import logging
import os
import sys

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.setting import (DEFAULT_BATCH_SIZE, DEFAULT_LLM_MODEL,
                            DEFAULT_NUM_VARIANTS, DEFAULT_TEST_SAMPLE_SIZE,
                            EMBEDDING_MODEL_NAME, OUTPUT_DIR)
from find_failures.rag_orchestrator import RAGTestOrchestrator
from find_failures.rag_tester import RAGSystemTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test RAG system with Llama models")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default='dataset/data.pkl',
        help="Path to the FAQ data pickle file"
    )
    
    # Add subparsers for different test modes
    subparsers = parser.add_subparsers(dest="command", help="Test command")
    # Set the default command to "test" if none is specified
    subparsers.required = False
    
    # Basic test command
    basic_parser = subparsers.add_parser("test", help="Run a basic RAG test")
    basic_parser.add_argument(
        "--llm_model", 
        type=str, 
        default=DEFAULT_LLM_MODEL,
        help=f"Llama model to use (default: {DEFAULT_LLM_MODEL})"
    )
    basic_parser.add_argument(
        "--embedding_model", 
        type=str, 
        default=EMBEDDING_MODEL_NAME,
        help=f"Embedding model to use (default: {EMBEDDING_MODEL_NAME})"
    )
    basic_parser.add_argument(
        "--sample_size", 
        type=int, 
        default=DEFAULT_TEST_SAMPLE_SIZE,
        help=f"Number of questions to test (default: {DEFAULT_TEST_SAMPLE_SIZE}, 0 for all)"
    )
    basic_parser.add_argument(
        "--batch_size", 
        type=int, 
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for processing questions (default: {DEFAULT_BATCH_SIZE})"
    )
    basic_parser.add_argument(
        "--include_variants", 
        action="store_true",
        help="Include question variants in the test"
    )
    basic_parser.add_argument(
        "--num_variants", 
        type=int, 
        default=DEFAULT_NUM_VARIANTS,
        help=f"Number of variants to generate per question (default: {DEFAULT_NUM_VARIANTS})"
    )
    
    # Model comparison command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple LLM models")
    compare_parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[DEFAULT_LLM_MODEL],
        help="List of models to compare (space separated)"
    )
    compare_parser.add_argument(
        "--sample_size", 
        type=int, 
        default=DEFAULT_TEST_SAMPLE_SIZE,
        help=f"Number of questions to test (default: {DEFAULT_TEST_SAMPLE_SIZE})"
    )
    compare_parser.add_argument(
        "--include_variants", 
        # action="store_true",
        default=True,
        help="Include question variants in the test"
    )
    
    # Batch size experiment command
    batch_parser = subparsers.add_parser("batch_experiment", help="Find optimal batch size")
    batch_parser.add_argument(
        "--llm_model", 
        type=str, 
        default=DEFAULT_LLM_MODEL,
        help=f"Llama model to use (default: {DEFAULT_LLM_MODEL})"
    )
    batch_parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 50],
        help="List of batch sizes to test (space separated, default: 1 5 10 20 50)"
    )
    batch_parser.add_argument(
        "--sample_size", 
        type=int, 
        default=DEFAULT_TEST_SAMPLE_SIZE,
        help=f"Number of questions to test (default: {DEFAULT_TEST_SAMPLE_SIZE})"
    )
    
    # Variant sensitivity experiment command
    variant_parser = subparsers.add_parser("variant_experiment", help="Test model sensitivity to variants")
    variant_parser.add_argument(
        "--llm_model", 
        type=str, 
        default=DEFAULT_LLM_MODEL,
        help=f"Llama model to use (default: {DEFAULT_LLM_MODEL})"
    )
    variant_parser.add_argument(
        "--variant_counts",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="List of variant counts to test (space separated, default: 1 3 5 10)"
    )
    variant_parser.add_argument(
        "--sample_size", 
        type=int, 
        default=DEFAULT_TEST_SAMPLE_SIZE,
        help=f"Number of questions to test (default: {DEFAULT_TEST_SAMPLE_SIZE})"
    )
    
    # Common arguments for all commands
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=OUTPUT_DIR,
        help=f"Directory to save test results (default: {OUTPUT_DIR})"
    )
    
    return parser.parse_args()

def main():
    """Run the RAG system test based on command line arguments."""
    args = parse_args()
    
    if not hasattr(args, 'command') or not args.command:
        args.command = "test"
        # Set default values for "test" command if they're not provided
        if not hasattr(args, 'llm_model'):
            args.llm_model = DEFAULT_LLM_MODEL
        if not hasattr(args, 'embedding_model'):
            args.embedding_model = EMBEDDING_MODEL_NAME
        if not hasattr(args, 'batch_size'):
            args.batch_size = DEFAULT_BATCH_SIZE
        if not hasattr(args, 'include_variants'):
            args.include_variants = False
        if not hasattr(args, 'num_variants'):
            args.num_variants = DEFAULT_NUM_VARIANTS
        
        logger.info(f"No command specified. Defaulting to 'test' command with model: {args.llm_model}")
    
    # Convert sample_size=0 to None (test all questions)
    if hasattr(args, 'sample_size'):
        sample_size = args.sample_size if args.sample_size > 0 else None
    else:
        sample_size = DEFAULT_TEST_SAMPLE_SIZE
    
    try:
        if args.command == "test":
            # Run basic test
            logger.info(f"Starting basic RAG system test with model: {args.llm_model}")
            
            # Initialize the tester
            tester = RAGSystemTester(
                data_path=args.data_path,
                llm_model_name=args.llm_model,
                embedding_model_name=args.embedding_model,
                output_dir=args.output_dir,
                test_sample_size=sample_size
            )
            
            # Run the test
            tester.run_test(
                include_variants=args.include_variants,
                num_variants=args.num_variants,
                batch_size=args.batch_size
            )
            
        elif args.command == "compare":
            # Run model comparison
            logger.info(f"Starting model comparison with {len(args.models)} models")
            
            # Initialize the orchestrator
            orchestrator = RAGTestOrchestrator(
                data_path=args.data_path,
                output_dir=args.output_dir
            )
            
            # Run the comparison
            orchestrator.run_model_comparison(
                model_names=args.models,
                sample_size=sample_size,
                include_variants=args.include_variants
            )
            
        elif args.command == "batch_experiment":
            # Run batch size experiment
            logger.info(f"Starting batch size experiment with model: {args.llm_model}")
            
            # Initialize the orchestrator
            orchestrator = RAGTestOrchestrator(
                data_path=args.data_path,
                output_dir=args.output_dir
            )
            
            # Run the experiment
            orchestrator.run_batch_size_experiment(
                model_name=args.llm_model,
                batch_sizes=args.batch_sizes,
                sample_size=sample_size
            )
            
        elif args.command == "variant_experiment":
            # Run variant sensitivity experiment
            logger.info(f"Starting variant sensitivity experiment with model: {args.llm_model}")
            
            # Initialize the orchestrator
            orchestrator = RAGTestOrchestrator(
                data_path=args.data_path,
                output_dir=args.output_dir
            )
            
            # Run the experiment
            orchestrator.run_variant_sensitivity_experiment(
                model_name=args.llm_model,
                num_variants_list=args.variant_counts,
                sample_size=sample_size
            )
            
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
    except Exception as e:
        logger.error(f"Error running RAG system test: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())