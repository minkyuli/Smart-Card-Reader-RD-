#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install --upgrade google-cloud-aiplatform')
get_ipython().system('pip install --upgrade vertexai')


# In[7]:


import os
import csv
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
import time
import re
import shutil


# In[8]:


MODELS_TO_TEST = [
    "gemini-pro-vision",
#    "gemini-2.0-flash",
]


# In[9]:


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


# In[19]:


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
    # Vertex AI 초기화
    vertexai.init(project=project_id, location=location)

    # Gemini Pro Vision 모델 로드
    model = GenerativeModel("gemini-pro-vision")
    
    image_files = sorted(os.listdir(os.path.join("data", "images")), 
                key=lambda x: int(''.join(filter(str.isdigit, x))))
    print(image_files)
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
            image_data = encode_image_to_binary(os.path.join("data", "images", image_path))
            print(f"모델: {model} / 이미지: {os.path.basename(image_path)} / 질문: {question}")        
        
            try:
                # Vertex AI 초기화
                vertexai.init(project=project_id, location=location)

                # Gemini Pro Vision 모델 로드
                model = GenerativeModel("gemini-pro-vision")

                # 이미지 파일 읽기
                # with open(image_path, "rb") as image_file:
                #     img_data = image_file.read()

                # 이미지를 Gemini 모델에 전달할 수 있는 형태로 변환
                img_part = Part.from_data(data=image_data, mime_type="image/jpeg")  # 이미지 형식에 맞게 변경

#                 # 프롬프트 설정
#                 prompt = "이 명함 이미지에서 이름, 전화번호, 주소, 회사명을 추출하여 JSON 형식으로 알려주세요."

                # Generation config 설정
                generation_config = GenerationConfig(temperature=0.2, top_p=1, top_k=32)

                contents = []

                if system_instruction:
                    contents.append(system_instruction)

                contents.append(img_part)
                contents.append(question)
    

                start_time = time.time()
                response = model.generate_content(
                    contents,
                    generation_config=generation_config,
                )

                inference_time = round(time.time() - start_time, 2)
                text = response.text
                # print(text)

            except Exception as e:
                print(f"Error: {str(e)}")
                return None
            
            results.append({
                "model": model,
                "image": os.path.basename(image_path),
                "question": question,
                "response": text,
                "inference_time": inference_time
            })

    # 결과를 CSV 파일로 저장 (한 파일에 모든 결과가 기록됨)
    csv_file = "test_results.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=["model", "image", "question", "response","inference_time"])
        writer.writeheader()
        writer.writerows(results)

    print(f"테스트 결과가 {csv_file} 파일에 저장되었습니다.")


# In[20]:


# 이미지 파일 경로, 프로젝트 ID, 리전, 시스템 지침 설정 (실제 경로 및 정보로 변경)
# image_file_path = "data/images/image_1.png"  # 예시 파일명
project_id = "project_id"  # 실제 Google Cloud 프로젝트 ID
location = "location"  # 실제 Google Cloud 리전
system_instruction = """당신은 스마트 명함 관리 서비스를 위한 AI 어시스턴트입니다. 당신의 주요 기능은 명함 이미지를 처리하고, 인맥 만남에 대한 맥락 정보를 수집하며, 연락처 정보를 관리하는 것입니다. 다음 지침을 주의 깊게 따르세요:

1. 시스템 정의:
   텍스트나 이미지 데이터 형태의 사용자 입력을 받게 됩니다. 응답은 친근하고 대화체여야 합니다. 항상 새로운 정보를 처리하거나 상호작용의 다음 단계로 넘어갈 준비가 되어 있어야 합니다.

2. 이미지 처리:
   이미지 데이터를 받으면 명함 정보를 분석하세요. 이미지에 명함이 포함되어 있다면 다음 세부 정보를 추출하여 JSON 형식으로 작성하세요:
   이미지를 처리하고 추출된 정보를 다음 형식으로 출력하세요:
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
   이미지 처리 후, 인맥 만남에 대한 추가적인 맥락 정보를 수집하세요. 다음과 같은 질문을 친근한 방식으로 사용자에게 물어보세요:
   - 이 사람을 어디서 만났나요?
   - 어떤 종류의 이벤트나 환경이었나요?
   - 그들과 무엇을 논의했나요?
   - 그들에 대해 기억나는 독특한 특징이나 주제가 있나요?
   - 향후 교류를 위해 특별히 기억하고 싶은 것이 있나요?
   이 정보를 수집하고 연락처 세부 정보에 추가할 준비를 하세요.

4. 정보 요약 및 확인:
   추출된 명함 세부 정보와 맥락 정보를 포함한 모든 수집된 정보를 요약하세요. 이 요약을 사용자에게 제시하고 확인을 요청하세요. 예를 들면:
   "좋습니다! 새 연락처에 대해 수집한 정보는 다음과 같습니다:
   [명함 세부 정보 및 맥락 정보 요약]
   이 정보가 맞나요? 추가하거나 변경하고 싶은 내용이 있나요?"

5. 상호작용 마무리 및 다음 단계:
   사용자가 정보가 정확하다고 확인하면 다음과 같이 응답하세요:
   "좋습니다! 이 연락처를 데이터베이스에 추가했습니다. 처리하고 싶은 다른 명함이 있으신가요?"
   사용자가 '아니오'라고 대답하면 감사 인사를 하고 대화를 종료하세요. '네'라고 대답하면 다음 이미지 처리를 준비하세요.

상호작용 전반에 걸쳐 도움이 되고 친근한 어조를 유지하세요. 프로세스나 수집된 정보에 대해 사용자가 가질 수 있는 질문이나 명확한 설명 요청에 대응할 준비가 되어 있어야 합니다.

사용자에게 인사하고 처리할 명함 이미지가 있는지 물어보며 상호작용을 시작하세요:""" # 시스템 지침 추가


# In[21]:


if __name__ == '__main__':
    run_tests()


# In[ ]:




