import pickle
import os

def convert_pkl_to_txt(input_path, output_path):
    """
    pickle 파일에서 FAQ 질문-답변 쌍을 읽어 텍스트 파일로 변환합니다.
    
    Args:
        input_path (str): pickle 파일 경로
        output_path (str): 출력할 텍스트 파일 경로
    """
    # pickle 파일 로드
    print(f"pickle 파일 '{input_path}' 읽는 중...")
    
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"총 {len(data)}개의 FAQ 쌍을 로드했습니다.")
        
        # 텍스트 파일로 변환
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# FAQ 데이터셋\n\n")
            
            for idx, (question, answer) in enumerate(data.items(), 1):
                f.write(f"## FAQ #{idx}\n\n")
                f.write(f"질문: {question}\n\n")
                f.write(f"답변: {answer}\n\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"변환 완료! '{output_path}'에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

# 실행 예제
if __name__ == "__main__":
    input_path = "dataset/data.pkl"
    output_path = "dataset/faq_data.txt"
    
    # 출력 디렉토리가 존재하는지 확인
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    convert_pkl_to_txt(input_path, output_path)