import os
import csv
from dotenv import load_dotenv
from ollama import Client
import time

load_dotenv()
HOST_URL = os.getenv('HOST_URL')

if HOST_URL is None:
    raise ValueError("HOST_URL이 .env 파일에 정의되어 있지 않습니다.")

client = Client(host=HOST_URL)

# 테스트할 모델 리스트 (원하는 모델들을 여기에 추가)
MODELS_TO_TEST = [
    # "bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh",
    # "erwan2/DeepSeek-Janus-Pro-7B",
    "llama3.2-vision",
    # "llama3.2-vision:90b",
    # "llava-llama3",
]

def encode_image_to_binary(image_path):
    """이미지를 바이너리 형태로 읽어서 반환하는 함수"""
    with open(image_path, 'rb') as f:
        return f.read()

def load_questions(filepath):
    """질문 텍스트 파일을 읽어 리스트로 반환하는 함수 (각 줄 하나의 질문)"""
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(line)
    return questions

def run_tests():
    image_files = sorted(os.listdir(os.path.join("data", "images")), 
                    key=lambda x: int(''.join(filter(str.isdigit, x))))
    if not image_files:
        print("이미지 파일을 찾을 수 없습니다. data/images/ 폴더를 확인하세요.")
        return

    questions = load_questions(os.path.join("data", "questions.txt"))
    if not questions:
        print("질문 파일에 내용이 없습니다. data/questions.txt 파일을 확인하세요.")
        return

    results = []

    # 모델, 이미지, 질문별로 테스트 실행
    for model in MODELS_TO_TEST:
        for image_path, question in zip(image_files, questions):
            image_data = encode_image_to_binary(os.path.join('data','images',image_path))
            print(f"모델: {model} / 이미지: {os.path.basename(image_path)} / 질문: {question}")
            try:
                start_time = time.time()
                response = client.chat(
                    model=model,
                    messages=[
                        {
                            'role': 'user',
                            'content': question,
                            'images': [image_data]
                        }
                    ],
                    options={
                        'temperature': 0.3,
                        'top_p': 0.1,
                        'frequency_penalty': 0.5,
                        'presence_penalty': 0.5,
                        'stop': ['!', '?'],
                        'max_tokens': 200
                    }
                )
                    # 응답 메시지 추출 (없으면 빈 문자열)
                inference_time = round(time.time() - start_time, 2)
                response_text = response.get('message', {}).get('content', '')
            except Exception as e:
                response_text = f"Error: {str(e)}"
                inference_time = 0.0
            
            results.append({
                "model": model,
                "image": os.path.basename(image_path),
                "question": question,
                "response": response_text,
                "inference_time": inference_time
            })

    # 결과를 CSV 파일로 저장 (한 파일에 모든 결과가 기록됨)
    csv_file = "test_results.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=["model", "image", "question", "response","inference_time"])
        writer.writeheader()
        writer.writerows(results)

    print(f"테스트 결과가 {csv_file} 파일에 저장되었습니다.")

if __name__ == '__main__':
    run_tests()