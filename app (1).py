import streamlit as st
import pandas as pd
import math
from collections import Counter
import os


def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def predict_knn(input_data, train_data, k=5): # Mặc định k=5 nếu không truyền vào
    distances = []
    for item in train_data:
        dist = euclidean_distance(input_data, item["features"])
        distances.append((dist, item["label"]))
    distances.sort(key=lambda x: x[0])
    neighbors = [dist[1] for dist in distances[:k]]
    return Counter(neighbors).most_common(1)[0][0]

# GIAO DIỆN
st.set_page_config(page_title="Sàng lọc Bệnh lý", layout="centered", initial_sidebar_state="collapsed")

# Tiêu đề chính ngay màn hình
st.title("Trợ lý Sàng lọc Bệnh lý")
st.markdown("Hệ thống sử dụng thuật toán KNN (với K=5) để phân loại mức độ bệnh lý dựa trên triệu chứng.")
st.divider()

# Nhận diện đường dẫn file CSV (chạy được Colab & Git)
if os.path.exists('/content/covid_symptoms_severity_prediction.csv'):
    csv_path = '/content/covid_symptoms_severity_prediction.csv'
else:
    # Trên GitHub, file CSV phải nằm ngang hàng với app.py
    csv_path = 'covid_symptoms_severity_prediction.csv'

try:
    df = pd.read_csv(csv_path)
    
    # Tiền xử lý dữ liệu (Mẹ đã tối ưu để code chạy nhanh hơn)
    train_data = [{"features": [r['fever'], r['cough'], r['fatigue'], r['shortness_of_breath']], 
                   "label": "Cần nhập viện" if r['hospitalized'] == 1 else "Theo dõi tại nhà"} 
                  for _, r in df.iterrows()]

    # KHU VỰC NHẬP LIỆU (Nằm ngay màn hình chính)
    st.subheader("1. Chọn các triệu chứng bạn đang gặp phải:")
    
    # Chia thành 2 cột cho đẹp mắt
    col1, col2 = st.columns(2)
    
    with col1:
        sot = st.checkbox("Sốt (Fever)", value=False)
        ho = st.checkbox("Ho (Cough)", value=False)
        
    with col2:
        met_moi = st.checkbox("Mệt mỏi (Fatigue)", value=False)
        kho_tho = st.checkbox("Khó thở (Shortness of Breath)", value=False)

    st.divider()
    
    # KHU VỰC DỰ ĐOÁN
    if st.button("Đưa ra chẩn đoán"):
        # Checkbox (True/False) thành (1/0)
        user_input = [int(sot), int(ho), int(met_moi), int(kho_tho)]
        
        # Gọi hàm dự đoán, fix cứng K=5
        result = predict_knn(user_input, train_data, k=5)
        
        # Hiển thị kết quả nổi bật
        st.subheader("2. Kết quả phân tích:")
        if result == "Cần nhập viện":
            st.error(f"## Dự báo: {result}")
            st.write("Dữ liệu cho thấy triệu chứng của bạn giống với các ca cần chăm sóc y tế chuyên sâu.")
        else:
            st.success(f"## Dự báo: {result}")
            st.write("Các triệu chứng hiện tại có mức độ tương đồng thấp với các ca nhập viện.")
            
except Exception as e:
    st.error(f"Lỗi: Không tìm thấy file dữ liệu tại '{csv_path}")
