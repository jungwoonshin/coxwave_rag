from question_from_answer.generate import QuestionAnswerGenerator

from config.setting import OPENAI_API_KEY, OPENAI_LLM_MODEL_NAME
import pickle

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
        qa_pairs = {}
        
        for answer_idx, answer in enumerate(tqdm(answers, desc="Generating questions")):
            questions = self.generate_questions_from_answer(answer)
            
            for question in questions:
                qa_pairs[question] = answer
                
            # 로그 출력
            if (answer_idx + 1) % 5 == 0 or answer_idx == len(answers) - 1:
                self.logger.info(f"Generated {len(qa_pairs)} Q&A pairs from {answer_idx + 1}/{len(answers)} answers")
        
        return qa_pairs
    
    def save_dataset(self, qa_pairs: Dict[str, str], filename: str = "qa_dataset.pkl"):
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
        
        qa_pairs_json = [{"question": q, "answer": a} for q, a in qa_pairs.items()]
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs_json, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Also saved as JSON to {json_path}")
        
        return str(file_path)

# 메인 실행 부분
if __name__ == "__main__":
    # 데이터 소스 설정
    USE_EXISTING_DATA = True  # 기존 데이터를 사용할지 여부
    EXISTING_DATA_PATH = "dataset/data.pkl"  # 기존 데이터 경로
    SAMPLE_SIZE = 500  # 처리할 데이터 샘플 수 (전체 데이터 사용 시 None)
    
    # OpenAI API 키 설정 (환경 변수에서 가져오거나 직접 입력)
    
    if USE_EXISTING_DATA:
        try:
            print(f"기존 데이터 파일 '{EXISTING_DATA_PATH}'을 로드합니다.")
            with open(EXISTING_DATA_PATH, 'rb') as f:
                existing_data = pickle.load(f)
            
            # 데이터 구조 확인 (dictionary 형태: {'질문내용':'답변내용',...})
            if isinstance(existing_data, dict):
                print(f"데이터 구조: 딕셔너리 (키-값 쌍 {len(existing_data)}개)")
                
                # 딕셔너리에서 답변만 추출 (값만 추출)
                answers = list(existing_data.values())
                
                # 샘플 크기 제한
                if SAMPLE_SIZE is not None and len(answers) > SAMPLE_SIZE:
                    print(f"{SAMPLE_SIZE}개의 샘플만 사용합니다 (총 {len(answers)}개 중)")
                    # 처음 SAMPLE_SIZE개 항목만 선택
                    answers = answers[:SAMPLE_SIZE]
                
                print(f"총 {len(answers)}개의 답변을 로드했습니다.")
                answers_to_use = answers
            else:
                print(f"예상치 못한 데이터 구조입니다: {type(existing_data)}")
                print("샘플 데이터를 대신 사용합니다.")
                # 샘플 답변 몇 개만 제공
                answers_to_use = [
                    "RAG 시스템은 검색과 생성을 결합한 AI 시스템입니다.",
                    "임베딩은 텍스트를 벡터 공간에 표현하는 방법입니다."
                ]
        except Exception as e:
            print(f"데이터 로드 중 오류 발생: {e}")
            print("샘플 데이터를 대신 사용합니다.")
            # 샘플 답변 몇 개만 제공
            answers_to_use = [
                "RAG 시스템은 검색과 생성을 결합한 AI 시스템입니다.",
                "임베딩은 텍스트를 벡터 공간에 표현하는 방법입니다."
            ]
    else:
        # 샘플 답변 목록 (10개)
        answers_to_use = [
            "RAG(Retrieval-Augmented Generation) 시스템은 대규모 언어 모델과 검색 엔진을 결합한 시스템입니다.",
            "BLEU 점수는 기계 번역 품질을 평가하는 메트릭으로, n-gram의 일치도를 측정합니다.",
            # 더 많은 샘플 답변...
        ]
    
    # 생성기 초기화 및 실행
    generator = QuestionAnswerGenerator(
        openai_api_key=OPENAI_API_KEY,
        model_name=OPENAI_LLM_MODEL_NAME,
        num_questions_per_answer=3,  # 답변당 3개 질문 생성
        output_dir="dataset",  # 저장 경로 설정
        verbose=True  # 상세 로그 출력
    )
    
    # 데이터셋 생성
    qa_pairs = generator.generate_dataset(answers_to_use)
    
    # 저장
    filename = "qa_dataset_generated.pkl"
    dataset_path = generator.save_dataset(qa_pairs, filename=filename)
    
    print(f"생성 완료! 데이터셋이 {dataset_path}에 저장되었습니다.")
    print(f"총 {len(qa_pairs)}개의 질문-답변 쌍이 생성되었습니다.")