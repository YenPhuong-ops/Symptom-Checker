import streamlit as st
import pandas as pd
import math
from collections import Counter
import os


def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def predict_knn(input_data, train_data, k=5):
    distances = []
    for item in train_data:
        dist = euclidean_distance(input_data, item["features"])
        distances.append((dist, item["label"]))
    distances.sort(key=lambda x: x[0])
    neighbors = [dist[1] for dist in distances[:k]]
    return Counter(neighbors).most_common(1)[0][0]

# --- GIAO DIỆN ---
st.set_page_config(page_title="Sàng lọc Bệnh lý", layout="centered")
st.title("Trợ lý Sàng lọc Bệnh lý")

# MẸO: Tự động nhận diện đường dẫn (Chạy được cả Colab và Git)
if os.path.exists('/content/covid_symptoms_severity_prediction.csv'):
    csv_path = '/content/covid_symptoms_severity_prediction.csv'
else:
    csv_path = 'covid_symptoms_severity_prediction.csv'

try:
    df = pd.read_csv(csv_path)
    
    # Chuẩn bị dữ liệu cho KNN
    train_data = []
    for _, r in df.iterrows():
        train_data.append({
            "features": [r['fever'], r['cough'], r['fatigue'], r['shortness_of_breath']], 
            "label": "Cần nhập viện" if r['hospitalized'] == 1 else "Theo dõi tại nhà"
        })

    st.sidebar.header("Chọn triệu chứng")
    f1 = st.sidebar.toggle("Sốt")
    f2 = st.sidebar.toggle("Ho")
    f3 = st.sidebar.toggle("Mệt mỏi")
    f4 = st.sidebar.toggle("Khó thở")
    k_val = st.sidebar.slider("Chọn K", 1, 15, 5)

    if st.button("Dự đoán ngay"):
        user_input = [int(f1), int(f2), int(f3), int(f4)]
        result = predict_knn(user_input, train_data, k=k_val)
        
        if result == "Cần nhập viện":
            st.error(f"### Kết quả: {result}")
        else:
            st.success(f"### Kết quả: {result}")
            
except Exception as e:
    st.error(f"Lỗi: Không tìm thấy file CSV trên GitHub.")
