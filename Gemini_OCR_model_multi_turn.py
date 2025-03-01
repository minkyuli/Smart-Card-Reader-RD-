#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install --upgrade google-cloud-aiplatform')
get_ipython().system('pip install --upgrade vertexai')


# In[23]:


import os
import csv
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
import time
import re
import shutil


# In[24]:


# 이미지 파일 경로, 프로젝트 ID, 리전, 시스템 지침 설정 (실제 경로 및 정보로 변경)
# image_file_path = "data/images/image_1.png"  # 예시 파일명
project_id = "nimble-analyst-452123-t9"  # 실제 Google Cloud 프로젝트 ID
location = "us-central1"  # 실제 Google Cloud 리전
system_instruction = """당신은 스마트 명함 관리 서비스를 위한 AI 어시스턴트입니다. 주요 기능은 명함 이미지를 처리하고, 인맥 만남에 대한 맥락 정보를 수집하며, 연락처 정보를 관리하는 것입니다. 다음 지침을 따르세요:
1. 시스템 정의:
   텍스트나 이미지 데이터 형태의 사용자 입력을 받습니다. 응답은 친근하고 대화체여야 하며, 항상 새로운 정보를 처리하거나 다음 단계로 넘어갈 준비가 되어 있어야 합니다.
2. 이미지 처리:
   명함 이미지를 받으면 정보를 분석하고, 다음 세부 정보를 JSON 형식으로 추출하세요:
   <extracted_info>
   {
     "name": "",
     "position": "",
     "company": "",
     "phone": "",
     "email": "",
     "address": "",
     "website": "",
     "other_details": []
   }
   </extracted_info>
3. 맥락 정보 수집:
   이미지 처리 후, 추가 맥락 정보를 수집합니다. 예: "이 사람을 어디서 만났나요?", "무엇을 논의했나요?" 등 친근하게 물어보고, 연락처 정보에 추가합니다.
4. 정보 요약 및 확인:
   추출된 명함 정보와 맥락 정보를 요약해 사용자에게 제시하고 확인을 요청하세요.
5. 상호작용 마무리:
   정보가 정확하면 "이 연락처를 데이터베이스에 추가했습니다. 다른 명함 처리할까요?"로 응답하세요.
6. 만약 이미지가 명함이 아니라면, 사용자의 질문이나 요청 사항으로부터 명함 정보를 만들어주세요.
7. 모든 답변은 한국어로 답변해주세요.

친근한 어조를 유지하며, 사용자의 질문이나 설명 요청에 대응하세요。""" # 시스템 지침 추가


# In[25]:


MODELS_TO_TEST = [
    "gemini-pro-vision",
#    "gemini-2.0-flash",
]


# In[26]:


def delete_folder(folder_path):
    """주어진 경로의 폴더를 삭제합니다.

    Args:
        folder_path (str): 삭제할 폴더의 경로.
    """
    try:
        shutil.rmtree(folder_path)
        print(f"폴더 '{folder_path}' 삭제 완료")
    except FileNotFoundError:
        print(f"폴더 '{folder_path}'를 찾을 수 없습니다.")
    except Exception as e:
        print(f"폴더 삭제 중 오류 발생: {e}")

# 삭제할 폴더 경로 예시
folder_to_delete = "data/images/.ipynb_checkpoints"  # 현재 디렉토리의 'checkpoints' 폴더 삭제

# 폴더 삭제 함수 호출
delete_folder(folder_to_delete)

# 폴더가 존재하는지 확인 후 삭제하는 방법
if os.path.exists(folder_to_delete):
    delete_folder(folder_to_delete)
else:
    print(f"폴더 '{folder_to_delete}'가 존재하지 않습니다.")


# In[27]:


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
    # print(image_files)
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
            
            # Vertex AI 초기화
            vertexai.init(project=project_id, location=location)

            # Gemini Pro Vision 모델 로드
            model = GenerativeModel("gemini-pro-vision")
            
            turns = question.split('|')
            if len(turns) != 3:
                print(f"잘못된 질문 형식: {question_line}")
                continue
            
            image_data = encode_image_to_binary(os.path.join("data", "images", image_path))
            print(f"모델: {model} / 이미지: {os.path.basename(image_path)} / 질문: {question}")        
        
            try:
                # 이미지를 Gemini 모델에 전달할 수 있는 형태로 변환
                img_part = Part.from_data(data=image_data, mime_type="image/jpeg")  # 이미지 형식에 맞게 변경

                # Generation config 설정
                generation_config = GenerationConfig(temperature=0.2, top_p=1, top_k=32)

                contents = []

                if system_instruction:
                    contents.append(system_instruction)

                contents.append(img_part)
    
                print(f"이미지: {os.path.basename(image_path)} 와의 대화를 시작합니다.")
                
                question_parts = turns
                print(question_parts)
                turn_idx = 1
                for question_part in question_parts:
                    question_part = question_part.strip()
                    if question_part:
                        contents.append(question_part)

                        start_time = time.time()
                    print(question_part)
                    response = model.generate_content(
                        contents,
                        generation_config=generation_config,
                    )

                    inference_time = round(time.time() - start_time, 2)
                    text = response.text
                    print(f"질문: {question_part}")
                    print(f"답변: {text} (응답 시간: {inference_time}초)")
                    # print(text)
                    
                    results.append({
                        "model": model,
                        "image": os.path.basename(image_path),
                        "turn": turn_idx,
                        "question": question,
                        "response": text,
                        "inference_time": inference_time
                    })
                    
                    turn_idx += 1

            except Exception as e:
                print(f"Error: {str(e)}")
                return None
            
            

    # 결과를 CSV 파일로 저장 (한 파일에 모든 결과가 기록됨)
    csv_file = "test_results.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=["model", "image", "turn", "question", "response","inference_time"])
        writer.writeheader()
        writer.writerows(results)

    print(f"테스트 결과가 {csv_file} 파일에 저장되었습니다.")


# In[ ]:


if __name__ == '__main__':
    if os.path.exists(folder_to_delete):
        delete_folder(folder_to_delete)
    else:
        print(f"폴더 '{folder_to_delete}'가 존재하지 않습니다.")
    run_tests()


# In[ ]:





# In[ ]:




