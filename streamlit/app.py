import os
import json
import tempfile
import streamlit as st
from pathlib import Path
from PIL import Image
# from dotenv import load_dotenv
from google import genai  # 최신 Google Gemini SDK 사용
from google.genai import types

# 페이지 설정
st.set_page_config(
    page_title="스마트 명함 관리 서비스",
    page_icon="📇",
    layout="centered"
)

# 환경 변수 로드
# load_dotenv()
api_key = "GOOGLE_API_KEY"
if not api_key:
    raise ValueError("GOOGLE_API_KEY가 .env 파일에 정의되어 있지 않습니다.")

# Google Gemini 클라이언트 초기화
client = genai.Client(api_key=api_key)

# 시스템 지침
SYSTEM_INSTRUCTION = """
당신은 스마트 명함 관리 서비스를 위한 AI 어시스턴트입니다. 주요 기능은 명함 이미지를 처리하고, 인맥 만남에 대한 맥락 정보를 수집하며, 연락처 정보를 관리하는 것입니다. 다음 지침을 따르세요:

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
   추출된 명함 정보와 맥락 정보를 요약해 사용자에게 제시하고 확인을 요청하세요. 예:
   "좋습니다! 새 연락처에 대해 수집한 정보는 다음과 같습니다: [요약] 이 정보가 맞나요? 추가하거나 변경하고 싶은 내용이 있나요?"

5. 상호작용 마무리:
   정보가 정확하면 추출된 연락처 정보를 데이터베이스에 Dictionary 형태로 저장을 합니다. 그 후 다른 명함 처리할까요?"로 응답하고, 사용자가 '네'면 다음 이미지, '아니오'면 종료합니다.

친근한 어조를 유지하며, 사용자의 질문이나 설명 요청에 대응하세요.
"""

# 세션 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'selected_images' not in st.session_state:
    st.session_state.selected_images = []
if 'image_processed' not in st.session_state:
    st.session_state.image_processed = False

# 채팅 메시지 표시 함수
def display_chat():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "images" in message and message["images"]:
                for img_path in message["images"]:
                    st.image(img_path, width=200)

# Google Gemini로 응답 생성 함수 (수정됨)
def generate_response(user_input, images=None):
    contents = []  # 텍스트 입력 추가

    # 이전 대화 기록 추가
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            contents.append(f"user: {message['content']}")
        else:
            contents.append(f"assistant: {message['content']}")
            
    contents.append(f"user: {user_input}")

    
    # 이미지가 아직 처리되지 않았고, 이미지가 제공된 경우에만 포함
    if images and not st.session_state.image_processed:
        for img_path in images:
            img = Image.open(img_path)  # PIL Image 객체로 열기
            contents.append(img)  # 이미지 추가
        st.session_state.image_processed = True  # 이미지 처리 완료 플래그 설정
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',  # 비전 모델 사용
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,  # 시스템 지침 추가
            max_output_tokens=1000,  # 출력 토큰 제한
            temperature=0.7,  # 응답의 창의성 조정
        )
    )
    return response.text

# UI 구성
st.title("📇 스마트 명함 관리 서비스")
st.write("명함 이미지를 업로드하고 대화로 정보를 정리하세요!")

# 이미지 업로드 섹션 (사이드바)
with st.sidebar:
    st.header("이미지 업로드")
    uploaded_files = st.file_uploader(
        "명함 이미지 업로드 (여러 개 가능)", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True,
        help="드래그앤드롭 또는 클릭으로 파일 선택"
    )
    
    if uploaded_files:
        st.session_state.uploaded_images = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.read())
                st.session_state.uploaded_images.append(tmp.name)
        st.success(f"{len(uploaded_files)}개의 이미지가 업로드되었습니다.")
        st.session_state.image_processed = False  # 새로운 이미지 업로드 시 플래그 초기화

# 채팅 UI
chat_container = st.container()
with chat_container:
    display_chat()

# 사용자 입력 창
user_input = st.chat_input("메시지를 입력하세요")

# 이미지 선택 및 제출 로직
if st.session_state.uploaded_images:
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("📷", help="업로드한 이미지 첨부"):
            st.session_state.selected_images = st.multiselect(
                "첨부할 이미지를 선택하세요",
                options=st.session_state.uploaded_images,
                format_func=lambda x: Path(x).name,
                key="image_select"
            )

if user_input:
    images_to_send = st.session_state.selected_images if st.session_state.selected_images else st.session_state.uploaded_images
    if not st.session_state.image_processed and images_to_send:  # 이미지가 처리되지 않았고 이미지가 있는 경우
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "images": images_to_send
        })
        response = generate_response(user_input, images_to_send)
    else:  # 이미지가 처리된 경우 텍스트만 처리
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        response = generate_response(user_input)
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })
    st.session_state.selected_images = []  # 선택 초기화
    st.rerun()

# 대화 기록 저장 버튼
if st.button("대화 기록 저장"):
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
    st.success("대화 기록이 저장되었습니다!")
