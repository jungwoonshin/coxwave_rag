
import logging
import os
from typing import Dict

from config.setting import HF_TOKEN, OPENAI_API_KEY
from dataset.validator_openai import FAQValidator
from llm.llama_model import LlamaModel

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def llama_main():
    # Hugging Face 토큰 설정
    hf_token = HF_TOKEN
    if not hf_token:
        raise ValueError("HF_TOKEN 환경 변수가 설정되지 않았습니다.")
    
    # LlamaModel 초기화
    model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"  # 또는 다른 Llama 모델
    llama_model = LlamaModel(model_name=model_name, token=hf_token)
    
    # FAQ 검증기 초기화
    validator = FAQValidator(llama_model=llama_model, similarity_threshold=0.85)
    
    # FAQ 데이터 로드
    data_path = "dataset/data.pkl"
    data = validator.load_data(data_path, sample=True)

    # 데이터셋 검증
    passed_data, failed_data, cleaned_data = validator.validate_dataset(data)
    
    # 결과 저장
    output_dir = "dataset/output"
    validator.save_results(passed_data, failed_data, cleaned_data, output_dir)
    
    logger.info("FAQ 검증 완료")

def openai_main():
    """
    메인 함수
    """
    # OpenAI API 키 설정
    api_key = OPENAI_API_KEY
    if not api_key:
        logger.error("OPENAI_API_KEY 환경 변수를 설정해주세요.")
        return
    
    # 모델 초기화
    try:
        from llm.openai_model import OpenAIModel  # 제공된 클래스 임포트
        model = OpenAIModel(model_name="gpt-4o-mini", api_key=api_key)
        logger.info("OpenAI 모델 초기화 완료")
    except Exception as e:
        logger.error(f"모델 초기화 중 오류: {str(e)}")
        return
    
    # 데이터셋 경로
    dataset_path = "dataset/data.pkl"
    output_dir = "dataset/validated"
    
    # 검증기 초기화 및 실행
    validator = FAQValidator(openai_model=model)
    
    try:
        # 데이터 로드
        data = validator.load_data(dataset_path, sample=True)
        
        # 데이터셋 검증
        passed_data, failed_data, cleaned_data = validator.validate_dataset(data)
        
        # 결과 저장
        validator.save_results(passed_data, failed_data, cleaned_data, output_dir)
        
        logger.info("FAQ 데이터셋 검증 및 정제 완료")
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")

def load_data(filepath: str, sample=False) -> Dict:
        """
        pickle 파일에서 FAQ 데이터 로드
        
        Args:
            filepath: pickle 파일 경로
            
        Returns:
            로드된 FAQ 데이터 (Dictionary)
        """
        import pickle
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            if sample:
                # 샘플 데이터 로드 (10개)
                data = dict(list(data.items())[:10])
            logger.info(f"{len(data)} 개의 FAQ 항목을 로드했습니다.")
            return data
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {e}")
            raise

if __name__ == "__main__":
    # openai_main()

    passed_data = "dataset/validated/passed_data.pkl"
    failed_data = "dataset/validated/failed_data.pkl"
    cleaned_data = "dataset/validated/cleaned_data.pkl"
    passed_data = load_data(passed_data)
    failed_data = load_data(failed_data)
    cleaned_data = load_data(cleaned_data)
    # 데이터 출력
    print("Passed Data:", passed_data)
    print("Failed Data:", failed_data)
    print("cleaned Data:", cleaned_data)