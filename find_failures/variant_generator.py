import json
import logging
from typing import Any, Dict, List

# Import components from your system
from utils.prompt import PromptBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VariantGenerator:
    """
    Class for generating question variants using an LLM.
    """
    def __init__(self, llm_model: Any):
        """
        Initialize the variant generator.
        
        Args:
            llm_model: The LLM to use for generation
        """
        self.llm_model = llm_model
    
    def generate_variants(self, questions: List[str], num_variants: int = 3) -> Dict[str, List[str]]:
        """
        Generate variants of the original questions.
        
        Args:
            questions: List of original questions
            num_variants: Number of variants to generate per question
            
        Returns:
            Dictionary mapping original questions to their variants
        """
        variants = {}
        
        # Get the variant generation prompt template
        variant_prompt_template = PromptBuilder.get_variant_generation_prompt()
        
        logger.info(f"Generating variants for {len(questions)} questions")
        
        for i, question in enumerate(questions):
            if i % 10 == 0:
                logger.info(f"Generated variants for {i}/{len(questions)} questions")
            
            # Get variants from LLM
            try:
                prompt = variant_prompt_template.format(question=question, num_variants=num_variants)
                
                # Generate with lower temperature for more controlled output
                response = self.llm_model.generate(
                    prompt=prompt,
                    temperature=0.5,
                    max_tokens=128,
                    stream=False
                )
                
                # Parse the response
                question_variants = self._parse_variant_response(response, num_variants)
                
                # Store variants
                variants[question] = question_variants
                
            except Exception as e:
                logger.warning(f"Error generating variants for question '{question}': {e}")
                variants[question] = []
        
        logger.info(f"Generated variants for all questions")
        return variants
    
    def _parse_variant_response(self, response: str, num_variants: int) -> List[str]:
        """
        Parse the LLM response to extract variants.
        
        Args:
            response: The LLM response
            num_variants: Maximum number of variants to extract
            
        Returns:
            List of variant questions
        """
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
        return question_variants[:num_variants]