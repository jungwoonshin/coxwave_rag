class PromptBuilder:
    """
    Class for creating prompts for the LLM.
    """
    @staticmethod
    def get_rag_prompt(with_followup: bool = True, history='', context=None, query=None) -> str:
        """
        Get the RAG prompt template.
        
        Args:
            with_followup: Whether to include follow-up question in the prompt
            history: Conversation history
            context: Context information from retrieved documents
            query: User's current query
            
        Returns:
            Formatted prompt string
        """
        followup_instruction = """
    답변을 제공한 후에, 사용자가 관련 주제에 대해 더 알고 싶을 수 있는 내용을 물어보세요.
    """ if with_followup else ""
        
        # Create the prompt template with explicit placeholders
        prompt_template = f"""
당신은 스마트 스토어 FAQ를 위한 챗봇입니다. 주어진 정보를 바탕으로 사용자의 질문에 정확하게 답하세요.

아래는 이전 대화 내용입니다:
{history}

아래는 관련 FAQ 정보입니다:
{context}

사용자 질문: {query}

지침:
1. 주어진 FAQ 정보를 바탕으로만 답변하세요.
2. 관련 FAQ 정보에 해당 내용이 없는 경우 "찾을 수 없는 정보입니다. 다시 질문해 주세요"라고 답변하세요.
3. 만약 사용자가 스마트 스토어 FAQ과 관련이 없는 질문을 한다면, "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다"라고 답변하세요.
4. 간결하고 이해하기 쉬운 언어로 답변하세요.
5. 사용자의 질문에 대해 답을 해준 뒤, 질의응답 맥락에서 사용자가 궁금해할만한 다른 내용을 물어봐야 합니다.
6. FAQ 내용을 그대로 복사하지 말고, 자연스러운 대화체로 정보를 전달하세요.{followup_instruction}

챗봇의 답변:
"""        
        # If parameters aren't provided, return the template with placeholders
        return prompt_template
    
    @staticmethod
    def get_unrelated_message() -> str:
        """
        Get the message to return for unrelated queries.
        
        Returns:
            Unrelated query response message
        """
        return "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다"
    
    @staticmethod
    def is_relevant(scores: list) -> bool:
        """
        Determine if the retrieved documents are relevant to the query.
        
        Args:
            scores: List of similarity scores from the retriever
            
        Returns:
            Boolean indicating whether the query is relevant
        """
        # If no documents were retrieved or the best score is too low,
        # consider the query unrelated to the FAQ domain
        if not scores or min(scores) > 0.5:  # Using L2 distance, lower is better
            return False
        return True
    
    @staticmethod
    def get_variant_generation_prompt(num_variants, question) -> str:
        """
        Get a prompt template for generating question variants.
        
        Returns:
            Prompt template string for generating variants
        """
        return f"""Generate {num_variants} different versions of the given question. 
The variants should:
0. Use Korean Language
1. Keep the same meaning but use different wording
2. Vary in formality (formal, casual, etc.)
3. Include some with different sentence structures
4. Potentially include common typing mistakes or colloquial expressions

Original question: {question}

Return only a JSON array of strings with the variant questions. Format: ["variant 1", "variant 2", "variant 3"]
"""
    
    @staticmethod
    def get_failure_analysis_prompt() -> str:
        """
        Get a prompt template for analyzing RAG failures.
        
        Returns:
            Prompt template string for failure analysis
        """
        return """You are an expert in diagnosing failures in RAG (Retrieval Augmented Generation) systems.
Analyze why the system failed to generate the correct answer based on the provided information.

Consider the following possible failure modes:
1. Retrieval Failure: The system didn't retrieve documents containing the correct answer
2. Relevance Failure: The system retrieved documents but ranked them poorly
3. Context Utilization Failure: The LLM didn't effectively use the retrieved context
4. Hallucination: The LLM generated information not present in the context
5. Other Failure: Any other failure mode you can identify

Test Question: {question}

Retrieved Documents:
{retrieved_context}

Correct Answer: {correct_answer}

Generated Answer: {generated_answer}

Evaluation:
{evaluation_details}

Additional Information:
Was correct answer in retrieved documents: {correct_in_retrieved}
If yes, rank of correct document: {retrieval_rank}

Return your analysis as a JSON object with the following fields:
- primary_failure_mode: one of the failure modes listed above
- explanation: detailed explanation of what went wrong
- suggestions: suggestions for improving system performance
"""