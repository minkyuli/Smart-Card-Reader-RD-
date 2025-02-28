import os
import csv
import time
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# 테스트할 모델 리스트
MODELS_TO_TEST = {
    "qwen": [
        "Qwen/Qwen2.5-VL-3B-Instruct",
        # "Qwen/Qwen2.5-VL-7B-Instruct",
        # "Qwen/Qwen2.5-VL-72B-Instruct"
    ]
}

class QwenModel:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model = None
        self.processor = None
        
    def initialize(self):
        """Qwen 모델 초기화"""
        print(f"\n{self.checkpoint} 모델 초기화 중...")
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.checkpoint,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.checkpoint)
            print(f"{self.checkpoint} 모델 초기화 완료")
            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU 메모리 부족: {self.checkpoint} 모델 스킵")
            else:
                print(f"모델 초기화 실패: {str(e)}")
            return False
            
    def clear(self):
        """GPU 메모리 해제"""
        if self.model is not None:
            del self.model
            del self.processor
            torch.cuda.empty_cache()
            self.model = None
            self.processor = None

def load_data():
    """테스트용 이미지와 질문 로드"""
    image_files = sorted(os.listdir(os.path.join("data", "images")), 
                    key=lambda x: int(''.join(filter(str.isdigit, x))))
    if not image_files:
        raise FileNotFoundError("이미지 파일을 찾을 수 없습니다.")

    with open(os.path.join("data", "questions.txt"), 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    if not questions:
        raise FileNotFoundError("질문 파일이 비어있습니다.")

    return image_files, questions

def run_inference(qwen_instance, image_path, question):
    """Qwen 모델 추론 실행"""
    print(f"\n=== {qwen_instance.checkpoint} 추론 시작 ===")
    if torch.cuda.is_available():
        print(f"GPU 메모리: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    start_time = time.time()
    try:
        image = Image.open(image_path)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"image": image_path},
            ]},
        ]
        
        text = qwen_instance.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_instance.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')

        output_ids = qwen_instance.model.generate(**inputs, max_new_tokens=200)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        response_text = qwen_instance.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        inference_time = round(time.time() - start_time, 2)
        print(f"추론 시간: {inference_time}초")
        
        return response_text, inference_time
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU 메모리 부족으로 추론 실패")
            return None, None
        raise

def run_tests():
    results = []
    try:
        image_files, questions = load_data()
    except FileNotFoundError as e:
        print(f"데이터 로드 실패: {str(e)}")
        return

    for checkpoint in MODELS_TO_TEST['qwen']:
        qwen_instance = QwenModel(checkpoint)
        if not qwen_instance.initialize():
            continue

        for image_path, question in zip(image_files, questions):
            full_image_path = os.path.join('data', 'images', image_path)
            print(f"모델: {checkpoint} / 이미지: {os.path.basename(image_path)}")
            
            response_text, inference_time = run_inference(qwen_instance, full_image_path, question)
            if response_text is None:  # GPU 메모리 부족으로 실패
                break
            
            results.append({
                "model": checkpoint,
                "image": os.path.basename(image_path),
                "question": question,
                "response": response_text,
                "inference_time": inference_time
            })
        
        qwen_instance.clear()

    # 결과 저장
    if results:
        with open("qwen_results.csv", mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=["model", "image", "question", "response", "inference_time"])
            writer.writeheader()
            writer.writerows(results)
        print("\n테스트 결과가 저장되었습니다.")

if __name__ == '__main__':
    run_tests()