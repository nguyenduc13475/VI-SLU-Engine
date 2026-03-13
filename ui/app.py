import streamlit as st
import requests
import json

# Setup Page Configuration
st.set_page_config(
    page_title="Vi-SLU API Demo",
    page_icon="🤖",
    layout="centered"
)

st.title("🎙️ Vietnamese SLU (IoT Execution Plan)")
st.markdown("""
Ứng dụng minh họa kiến trúc **Stateless API** cho Smart Home. 
Nhập lệnh tự nhiên vào, API sẽ trả về **Kế hoạch thực thi (JSON)** để Gateway/Edge xử lý.
""")

# Input section
st.subheader("Nhập câu lệnh điều khiển:")
user_input = st.text_input(
    "Ví dụ:", 
    value="bật đèn và quạt nhanh lên sau 10 giây nữa",
    placeholder="Nhập câu lệnh của bạn vào đây..."
)

API_URL = "http://localhost:8000/api/v1/parse"

if st.button("🚀 Gửi tới API", type="primary"):
    if user_input.strip():
        with st.spinner("Đang phân tích NLP..."):
            try:
                # Call FastAPI endpoint
                payload = {"text": user_input}
                response = requests.post(API_URL, json=payload, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display Results using columns
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.success("Nhận diện Intent:")
                        for intent in data.get("intents", []):
                            st.info(f"🏷️ {intent}")
                            
                    with col2:
                        st.success("Execution Plan (JSON cho thiết bị trạm):")
                        st.json(data.get("execution_plan", []))
                        
                else:
                    st.error(f"Lỗi từ server API: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Không thể kết nối tới API. Vui lòng kiểm tra xem bạn đã chạy `python -m src.api.main` chưa!")
    else:
        st.warning("Vui lòng nhập câu lệnh!")

st.markdown("---")
st.caption("Developed with Clean Architecture & FastAPI")