class PromptBuilder:
    """
    Class for creating prompts for the LLM.
    """
    @staticmethod
    def get_rag_prompt(with_followup: bool = True) -> str:
        """
        Get the RAG prompt template.
        
        Args:
            with_followup: Whether to include follow-up question in the prompt
            
        Returns:
            Prompt template string
        """
        followup_instruction = """
답변을 제공한 후에, 사용자가 관련 주제에 대해 더 알고 싶을 수 있는 내용을 물어보세요.
""" if with_followup else ""
        
        return f"""
당신은 스마트 스토어 FAQ를 위한 챗봇입니다. 주어진 정보를 바탕으로 사용자의 질문에 정확하게 답하세요.

아래는 이전 대화 내용입니다:
{{history}}

아래는 관련 FAQ 정보입니다:
{{context}}

사용자 질문: {{query}}

지침:
1. 주어진 FAQ 정보를 바탕으로만 답변하세요.
2. 답변을 할 수 없거나 관련 정보가 없는 경우 "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다"라고 답변하세요.
3. 간결하고 이해하기 쉬운 언어로 답변하세요.
4. FAQ 내용을 그대로 복사하지 말고, 자연스러운 대화체로 정보를 전달하세요.{followup_instruction}

챗봇의 답변:
"""
    
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
        if not scores or min(scores) > 0.8:  # Using L2 distance, lower is better
            return False
        return True