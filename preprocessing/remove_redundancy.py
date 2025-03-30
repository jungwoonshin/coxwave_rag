import json
import pickle
from pathlib import Path

# 경로 설정
dataset_dir = Path("dataset")
input_file = dataset_dir / "data.pkl"
output_file = dataset_dir / "cleaned_data.pkl"
json_output_file = dataset_dir / "cleaned_data.json"
json_original_input_file = dataset_dir / "original_data.json"
# 디렉터리 생성
dataset_dir.mkdir(exist_ok=True, parents=True)

def clean_faq_dict(faq_dict):
    """
    FAQ 데이터 딕셔너리에서 불필요한 내용을 제거하고 개행 문자와 특수 공백을 일반 공백으로 대체하는 함수
    중복된 질문이 있을 경우 마지막에 점(.)을 추가하여 구분함
    """
    print("FAQ 데이터 정리 시작...")
    
    # 결과를 저장할 딕셔너리
    cleaned_data = {}
    
    import re
    
    # 각 FAQ 항목을 처리
    for question, answer in faq_dict.items():
        # 원본 답변 저장
        original_answer = answer
        
        # "위 도움말이 도움이 되었나요" 이전 부분만 사용
        pattern_help = r'위\s*도움말이\s*도움이\s*되었나요'
        match = re.search(pattern_help, answer)
        if match:
            cleaned_answer = answer[:match.start()]
        else:
            # 패턴을 찾지 못한 경우 전체 텍스트 사용
            cleaned_answer = answer
        
        # 특수 공백 문자(\xa0)를 일반 공백으로 대체
        cleaned_answer = cleaned_answer.replace('\xa0', ' ')
        
        # 모든 개행 문자를 공백으로 대체
        cleaned_answer = cleaned_answer.replace('\n', ' ')
        
        # 연속된 여러 공백을 하나의 공백으로 축소
        cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer)
        
        # 앞뒤 공백 제거
        cleaned_answer = cleaned_answer.strip()
        
        # 질문이 이미 딕셔너리에 있는지 확인
        original_question = question
        while question in cleaned_data:
            # 중복된 질문이 있으면 끝에 점(.) 추가
            question = question + "."
        
        # 정리된 데이터 추가
        cleaned_data[question] = cleaned_answer
        
        # 원래 질문과 달라졌으면 로그 출력
        if question != original_question:
            print(f"중복 질문 처리: '{original_question}' -> '{question}'")
        
        # 원래 답변과 크게 달라졌으면 로그 출력 (길이가 절반 이상 줄어든 경우)
        if len(cleaned_answer) < len(original_answer) * 0.5:
            print(f"경고: 답변이 크게 줄어듦 - 질문: '{question}'")
            print(f"  원래 길이: {len(original_answer)}, 정리 후 길이: {len(cleaned_answer)}")
    
    print(f"정리 완료: {len(cleaned_data)}개 항목")
    return cleaned_data

def main():
    """메인 실행 함수"""
    try:
        # 입력 파일 읽기
        print(f"{input_file} 파일 읽는 중...")
        
        # pickle 파일 로드
        with open(input_file, "rb") as f:
            data = pickle.load(f)
        
        if not isinstance(data, dict):
            print("데이터가 딕셔너리 형식이 아닙니다!")
            return
        
        print(f"총 {len(data)}개 FAQ 항목 발견")
        
        # 원본 데이터 크기 확인
        original_size = sum(len(answer) for answer in data.values())
        print(f"원본 데이터 크기: {original_size} 문자")
        
        # 데이터 정리
        cleaned_data = clean_faq_dict(data)
        
        # 정리된 데이터 크기 확인
        cleaned_size = sum(len(answer) for answer in cleaned_data.values())
        reduction_percentage = ((1 - cleaned_size / original_size) * 100) if original_size else 0
        print(f"정리 후 크기: {cleaned_size} 문자 ({reduction_percentage:.2f}% 감소)")
        
        # 결과 저장 (pickle 형식)
        print(f"정리된 데이터를 {output_file}에 pickle 형식으로 저장 중...")
        with open(output_file, "wb") as f:
            pickle.dump(cleaned_data, f)
        
        # JSON 형식으로도 저장 (확인용)
        print(f"정리된 데이터를 JSON 형식으로 저장 중...")
        
        # 원본 데이터를 리스트 형태로 변환
        original_json_data = [{'question': question, 'answer': answer} for question, answer in data.items()]
        original_json_file = dataset_dir / "original_data.json"
        
        # 정리된 데이터를 리스트 형태로 변환
        cleaned_json_data = [{'question': question, 'answer': answer} for question, answer in cleaned_data.items()]
        
        # 각각 저장
        with open(original_json_file, "w", encoding="utf-8") as f:
            json.dump(original_json_data, f, ensure_ascii=False, indent=2)
        
        with open(json_output_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_json_data, f, ensure_ascii=False, indent=2)
        
        print(f"원본 데이터를 {json_output_file}에 JSON 형식으로 저장 중...")
        json_data = [{'question': question, 'answer':answer} for question, answer in data.items()]
        with open(json_original_input_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print("데이터 정리 및 저장 완료!")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()