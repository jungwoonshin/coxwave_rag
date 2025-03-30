import json
import os
import pickle
import logging
from typing import List, Dict
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# OpenAI API 사용을 위한 설정
import openai
from openai import OpenAI

# 기존 코드에서 사용되는 함수 재활용
def parse_llm_response(response: str) -> list:
    """
    Parse LLM responses that contain JSON arrays wrapped in markdown code blocks.
    
    Args:
        response: The raw response string from the LLM
        
    Returns:
        List of extracted items
    """
    # Remove markdown code block indicators and surrounding whitespace
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
        items = json.loads(json_content)
        if isinstance(items, list):
            return items
        else:
            return []
    except json.JSONDecodeError:
        # Fallback: try to extract list items manually
        import re
        # Look for quoted strings within brackets
        pattern = r'"([^"]*)"'
        matches = re.findall(pattern, json_content)
        return matches if matches else []

class QuestionAnswerGenerator:
    """
    답변 기반 질문을 생성하고 결과를 저장하는 클래스
    """
    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "gpt-4",
        output_dir: str = "dataset",
        num_questions_per_answer: int = 3,
        verbose: bool = True
    ):
        """
        매개변수:
            openai_api_key: OpenAI API 키
            model_name: 사용할 OpenAI 모델 이름
            output_dir: 결과 저장 경로
            num_questions_per_answer: 답변당 생성할 질문 수
        """
        self.api_key = openai_api_key
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.num_questions_per_answer = num_questions_per_answer
        self.verbose = verbose
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=openai_api_key)
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        if self.verbose:
            self.logger.info(f"QuestionAnswerGenerator 초기화 완료")
            self.logger.info(f"- 모델: {model_name}")
            self.logger.info(f"- 출력 경로: {output_dir}")
            self.logger.info(f"- 답변당 질문 수: {num_questions_per_answer}")
    
    def generate_questions_from_answer(self, answer: str) -> List[str]:
        """
        답변을 기반으로 질문 생성
        
        매개변수:
            answer: 질문을 생성할 답변 텍스트
            
        반환:
            생성된 질문 목록
        """
        prompt = f"""
        다음 답변을 보고, 이 답변에 해당하는 적절한 질문을 {self.num_questions_per_answer}개 생성해주세요.
        질문은 한국어로 작성해주세요. 다양한 표현과 구조를 사용해주세요.
        
        답변:
        ```
        {answer}
        ```
        
        JSON 형식으로 질문 목록만 반환해주세요:
        ```json
        ["질문1", "질문2", "질문3"]
        ```
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "당신은 주어진 답변에 적합한 질문을 생성하는 도우미입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            questions = parse_llm_response(content)
            
            return questions[:self.num_questions_per_answer]  # 요청한 개수만큼만 반환
            
        except Exception as e:
            self.logger.error(f"Error generating questions: {e}")
            return []
    
    def generate_dataset(self, answers: List[str]) -> List[Dict[str, str]]:
        """
        답변 목록으로부터 질문-답변 데이터셋 생성
        
        매개변수:
            answers: 답변 목록
            
        반환:
            질문-답변 쌍 목록 (dictionary 형태)
        """
        qa_pairs = []
        
        for answer_idx, answer in enumerate(tqdm(answers, desc="Generating questions")):
            questions = self.generate_questions_from_answer(answer)
            
            for question in questions:
                qa_pairs.append({
                    "question": question,
                    "answer": answer
                })
                
            # 로그 출력
            if (answer_idx + 1) % 5 == 0 or answer_idx == len(answers) - 1:
                self.logger.info(f"Generated {len(qa_pairs)} Q&A pairs from {answer_idx + 1}/{len(answers)} answers")
        
        return qa_pairs
    
    def save_dataset(self, qa_pairs: List[Dict[str, str]], filename: str = "qa_dataset.pkl"):
        """
        생성된 데이터셋을 pickle 파일로 저장
        
        매개변수:
            qa_pairs: 저장할 질문-답변 쌍 목록
            filename: 저장할 파일 이름
        """
        file_path = self.output_dir / filename
        
        # pickle 형식으로 저장
        with open(file_path, 'wb') as f:
            pickle.dump(qa_pairs, f)
            
        self.logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {file_path}")
        
        # JSON 형식으로도 저장 (확인용)
        json_filename = f"{filename.rsplit('.', 1)[0]}.json"
        json_path = self.output_dir / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Also saved as JSON to {json_path}")
        
        return str(file_path)