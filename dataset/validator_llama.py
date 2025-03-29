import pickle
import json
import os
import logging
from typing import Dict, List, Tuple, Set, Union, Generator
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
from collections import Counter

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAQValidator:
    """
    FAQ 데이터셋을 검증하고 분류하는 클래스
    """
    def __init__(self, llama_model, similarity_threshold=0.85):
        """
        초기화 함수
        
        Args:
            llama_model: LlamaModel 인스턴스
            similarity_threshold: 유사도 임계값 (기본값: 0.85)
        """
        self.llama_model = llama_model
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer='char_wb', ngram_range=(2, 4))
        
    def load_data(self, filepath: str, sample=False) -> Dict:
        """
        pickle 파일에서 FAQ 데이터 로드
        
        Args:
            filepath: pickle 파일 경로
            
        Returns:
            로드된 FAQ 데이터 (Dictionary)
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            if sample:
                data = dict(list(data.items())[:20])
            logger.info(f"{len(data)} 개의 FAQ 항목을 로드했습니다.")
            return data
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {e}")
            raise
    
    def check_structural_completeness(self, data: Dict) -> List[str]:
        """
        구조적 완전성 확인 (누락된 질문이나 답변)
        
        Args:
            data: FAQ 데이터셋
            
        Returns:
            구조적 오류가 있는 질문 목록
        """
        incomplete_items = []
        
        for question, answer in data.items():
            if not question or not answer:
                incomplete_items.append(question)
            elif isinstance(question, str) and isinstance(answer, str):
                if question.strip() == "" or answer.strip() == "":
                    incomplete_items.append(question)
            else:
                incomplete_items.append(question)
                
        logger.info(f"구조적 오류가 있는 항목: {len(incomplete_items)}개")
        return incomplete_items
    
    def find_duplicates(self, data: Dict) -> Dict[str, List[str]]:
        """
        중복 질문 찾기
        
        Args:
            data: FAQ 데이터셋
            
        Returns:
            유사한 질문들의 그룹 (Dictionary)
        """
        questions = list(data.keys())
        
        # TF-IDF 벡터화
        try:
            X = self.vectorizer.fit_transform(questions)
            similarity_matrix = cosine_similarity(X)
        except Exception as e:
            logger.error(f"유사도 계산 중 오류 발생: {e}")
            return {}
        
        # 유사한 질문 그룹화
        duplicate_groups = {}
        processed = set()
        
        for i in range(len(questions)):
            if i in processed:
                continue
                
            similar_indices = [j for j in range(len(questions)) if 
                              i != j and 
                              similarity_matrix[i, j] > self.similarity_threshold]
            
            if similar_indices:
                group_key = questions[i]
                duplicate_groups[group_key] = [questions[j] for j in similar_indices]
                processed.add(i)
                processed.update(similar_indices)
        
        logger.info(f"중복 그룹 수: {len(duplicate_groups)}")
        return duplicate_groups
    
    def _collect_generator_output(self, generator) -> str:
        """
        제너레이터의 출력을 문자열로 수집
        
        Args:
            generator: 텍스트 청크를 생성하는 제너레이터
            
        Returns:
            수집된 전체 텍스트
        """
        collected_text = ""
        try:
            for chunk in generator:
                collected_text += chunk
            return collected_text
        except Exception as e:
            logger.error(f"제너레이터 처리 중 오류: {e}")
            return collected_text
    
    def _parse_llm_json(self, text: str) -> Dict:
        """
        LLM의 출력에서 JSON을 추출하고 파싱
        
        Args:
            text: LLM 응답 텍스트
            
        Returns:
            파싱된 JSON 객체
        """
        # 여러 정규식 패턴으로 시도
        patterns = [
            r'({[\s\S]*})',  # 기본 패턴
            r'```json\s*([\s\S]*?)\s*```',  # 코드 블록 패턴
            r'({[\s\S]*"판정"[\s\S]*})'  # "판정" 키워드가 포함된 패턴
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                json_str = match.group(1).strip()
                try:
                    # 따옴표 일관성 수정 시도
                    json_str = json_str.replace("'", '"')
                    # 줄바꿈 및 공백 처리
                    json_str = re.sub(r'\s+', ' ', json_str)
                    # 후행 쉼표 제거
                    json_str = re.sub(r',\s*}', '}', json_str)
                    
                    # 특수 문자 이스케이프 처리
                    for ch in ['\n', '\r', '\t']:
                        json_str = json_str.replace(ch, '')
                    
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 실패, 다른 패턴 시도: {e}")
                    continue
        
        # 모든 패턴 실패 시 수동 파싱 시도
        try:
            lines = text.split('\n')
            result = {}
            
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip().strip('"\'').replace(' ', '_')
                    value = parts[1].strip().strip(',').strip('"\'')
                    
                    # 숫자 값 변환
                    try:
                        if value.isdigit():
                            value = int(value)
                        elif value.replace('.', '', 1).isdigit():
                            value = float(value)
                    except:
                        pass
                    
                    if key:
                        result[key] = value
            
            # 필수 키 확인
            if "정확성" in result and "판정" in result:
                return result
        except Exception as e:
            logger.error(f"수동 파싱 실패: {e}")
        
        # 모든 시도 실패 시 기본값 반환
        return {
            "정확성": 0, "완결성": 0, "명확성": 0, "연관성": 0, "언어적_품질": 0,
            "실패_이유": "파싱 오류",
            "판정": "실패",
            "제거할_내용": ""
        }
    
    def evaluate_content_quality(self, question: str, answer: str) -> Tuple[bool, Dict]:
        """
        LlamaModel을 사용하여 콘텐츠 품질 평가
        
        Args:
            question: 질문
            answer: 답변
            
        Returns:
            평가 통과 여부, 평가 결과 상세 정보
        """
        prompt_template = """당신은 FAQ 데이터셋의 품질을 평가하는 전문가입니다. 주어진 질문과 답변 쌍을 분석하고 다음 기준에 따라 평가해주세요:

1. 정확성 (1-5): 답변이 질문에 정확하게 대응하는지
2. 완결성 (1-5): 답변이 질문을 충분히 해결하는지
3. 명확성 (1-5): 질문과 답변이 명확하게 표현되었는지
4. 연관성 (1-5): 질문과 답변이 서로 연관되어 있는지
5. 언어적 품질 (1-5): 문법, 맞춤법, 용어 일관성 등

각 항목에 1-5점 사이의 점수를 매기고, 점수가 3점 미만인 항목이 있으면 간략한 이유를 설명해주세요.
마지막에 "통과" 또는 "실패"로 전체 판정을 내려주세요. (3점 미만 항목이 있으면 실패)
관련성이 없는 부분이 있다면 어떤 부분인지 "제거할 내용:"으로 시작하여 설명해주세요.

아래는 반드시 유효한 JSON 형식으로 작성해주세요:
{{
    "정확성": 점수,
    "완결성": 점수,
    "명확성": 점수, 
    "연관성": 점수,
    "언어적_품질": 점수,
    "실패_이유": "해당되는 경우 이유 설명",
    "판정": "통과" 또는 "실패",
    "제거할_내용": "관련성 없는 내용 설명 또는 빈 문자열"
}}

질문: {question}

답변: {answer}

평가 결과(반드시 유효한 JSON 형식으로만 작성):"""
        
        prompt = prompt_template.format(question=question, answer=answer)
        
        try:
            for attempt in range(3):  # 최대 3번 재시도
                try:
                    # 명시적으로 stream=False 설정
                    response = self.llama_model.generate(
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=1024,
                        stream=False
                    )
                    
                    # 응답이 제너레이터인 경우 처리
                    if hasattr(response, '__iter__') and not isinstance(response, (str, bytes)):
                        logger.warning("응답이 제너레이터 형태임, 문자열로 변환합니다.")
                        response = self._collect_generator_output(response)
                    
                    # 응답 파싱
                    result = self._parse_llm_json(response)
                    
                    # 결과 검증
                    if "정확성" in result and "판정" in result:
                        passed = result["판정"] == "통과"
                        return passed, result
                    else:
                        logger.warning(f"유효한 결과를 찾을 수 없음, 재시도 중 ({attempt+1}/3)")
                        time.sleep(1)
                        continue
                        
                except Exception as e:
                    logger.warning(f"평가 처리 중 오류, 재시도 중 ({attempt+1}/3): {e}")
                    time.sleep(1)  # 잠시 대기
                    continue
                    
            # 최대 재시도 후에도 실패
            logger.error("평가 최대 재시도 횟수 초과")
            return False, {"판정": "실패", "실패_이유": "응답 처리 오류"}
            
        except Exception as e:
            logger.error(f"평가 API 호출 중 오류: {str(e)}")
            return False, {"판정": "실패", "실패_이유": f"API 오류: {str(e)}"}
    
    def clean_irrelevant_content(self, answer: str, removal_info: str) -> str:
        """
        관련 없는 내용 제거
        
        Args:
            answer: 원본 답변
            removal_info: 제거할 내용 정보
            
        Returns:
            정제된 답변
        """
        if not removal_info or removal_info.strip() == "":
            return answer
            
        try:
            # 구체적인 제거 내용이 있는 경우
            cleaned_answer = answer
            
            # 단락 또는 문장 기반 제거 시도
            paragraphs = answer.split('\n\n')
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            
            # 긴 문자열 먼저 찾아보기
            for paragraph in paragraphs:
                if paragraph.strip() in removal_info:
                    cleaned_answer = cleaned_answer.replace(paragraph, "")
                    
            # 문장 단위로 찾아보기
            for sentence in sentences:
                if sentence.strip() in removal_info:
                    cleaned_answer = cleaned_answer.replace(sentence, "")
            
            # 공백 정리
            cleaned_answer = re.sub(r'\n{3,}', '\n\n', cleaned_answer)
            cleaned_answer = cleaned_answer.strip()
            
            return cleaned_answer
            
        except Exception as e:
            logger.error(f"내용 정제 중 오류: {str(e)}")
            return answer  # 오류 시 원본 반환
    
    def validate_dataset(self, data: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        전체 데이터셋 검증
        
        Args:
            data: FAQ 데이터셋
            
        Returns:
            검증 통과 데이터, 실패 데이터, 정제된 데이터
        """
        passed_data = {}
        failed_data = {}
        cleaned_data = {}
        
        total = len(data)
        logger.info(f"총 {total}개 항목 검증 시작")
        
        # 구조적 완전성 확인
        incomplete_items = self.check_structural_completeness(data)
        for question in incomplete_items:
            failed_data[question] = data[question]
        
        # 중복 확인 (정보용)
        duplicate_groups = self.find_duplicates(data)
        
        # 품질 평가
        count = 0
        for question, answer in data.items():
            if question in failed_data:
                continue
                
            count += 1
            if count % 50 == 0:
                logger.info(f"진행 중: {count}/{total}")
                
            try:
                passed, result = self.evaluate_content_quality(question, answer)
                
                if passed:
                    # 관련 없는 내용 제거
                    if "제거할_내용" in result and result["제거할_내용"]:
                        cleaned_answer = self.clean_irrelevant_content(answer, result["제거할_내용"])
                        cleaned_data[question] = cleaned_answer
                    else:
                        cleaned_data[question] = answer
                        
                    passed_data[question] = answer
                else:
                    failed_data[question] = answer
                    
            except Exception as e:
                logger.error(f"항목 '{question[:30]}...' 평가 중 오류: {str(e)}")
                failed_data[question] = answer
        
        logger.info(f"검증 완료: 통과 {len(passed_data)}개, 실패 {len(failed_data)}개")
        return passed_data, failed_data, cleaned_data
    
    def save_results(self, passed_data: Dict, failed_data: Dict, cleaned_data: Dict, 
                    output_dir: str = "output"):
        """
        결과 저장
        
        Args:
            passed_data: 통과한 데이터
            failed_data: 실패한 데이터
            cleaned_data: 정제된 데이터
            output_dir: 출력 디렉토리
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 통과 데이터 저장
        with open(os.path.join(output_dir, "passed_data.pkl"), 'wb') as f:
            pickle.dump(passed_data, f)
            
        # 정제된 데이터 저장
        with open(os.path.join(output_dir, "cleaned_data.pkl"), 'wb') as f:
            pickle.dump(cleaned_data, f)
            
        # 실패 데이터 저장
        with open(os.path.join(output_dir, "failed_data.pkl"), 'wb') as f:
            pickle.dump(failed_data, f)
            
        # 텍스트 파일로도 저장 (사람이 읽기 쉽게)
        with open(os.path.join(output_dir, "passed_data.txt"), 'w', encoding='utf-8') as f:
            for i, (question, answer) in enumerate(passed_data.items(), 1):
                f.write(f"## {i}. 질문: {question}\n\n")
                f.write(f"답변: {answer}\n\n")
                f.write("-" * 80 + "\n\n")
                
        with open(os.path.join(output_dir, "failed_data.txt"), 'w', encoding='utf-8') as f:
            for i, (question, answer) in enumerate(failed_data.items(), 1):
                f.write(f"## {i}. 질문: {question}\n\n")
                f.write(f"답변: {answer}\n\n")
                f.write("-" * 80 + "\n\n")
                
        with open(os.path.join(output_dir, "cleaned_data.txt"), 'w', encoding='utf-8') as f:
            for i, (question, answer) in enumerate(cleaned_data.items(), 1):
                f.write(f"## {i}. 질문: {question}\n\n")
                f.write(f"답변: {answer}\n\n")
                f.write("-" * 80 + "\n\n")
                
        logger.info(f"결과가 {output_dir} 디렉토리에 저장되었습니다.")
        
        # 통계 정보 생성
        stats = {
            "총_데이터_수": len(passed_data) + len(failed_data),
            "통과_데이터_수": len(passed_data),
            "실패_데이터_수": len(failed_data),
            "정제된_데이터_수": len(cleaned_data),
            "통과율": round(len(passed_data) / (len(passed_data) + len(failed_data)) * 100, 2)
        }
        
        with open(os.path.join(output_dir, "statistics.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
