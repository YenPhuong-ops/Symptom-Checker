import streamlit as st
import pandas as pd
import math
from collections import Counter
import os

# 1. Hàm tính khoảng cách Euclid
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# 2. Hàm dự đoán KNN
def predict_knn(input_data, train_data, k=5):
    distances = []
    for item in train_data:
        dist = euclidean_distance(input_data, item["features"])
        distances.append((dist, item["label"]))
    distances.sort(key=lambda x: x[0])
    neighbors = [dist[1] for dist in distances[:k]]
    return Counter(neighbors).most_common(1)[0][0]

# GIAO DIỆN
st.set_page_config(page_title="Sàng lọc Bệnh lý")
st.title("🩺 Trợ lý Sàng lọc Bệnh lý (KNN)")

# Đường dẫn file trên Colab
csv_path = '/content/covid_symptoms_severity_prediction.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    # Tiền xử lý dữ liệu nhanh
    train_data = []
    for _, r in df.iterrows():
        train_data.append({
            "features": [r['fever'], r['cough'], r['fatigue'], r['shortness_of_breath']], 
            "label": "Cần nhập viện" if r['hospitalized'] == 1 else "Theo dõi tại nhà"
        })

    st.sidebar.header("Chọn triệu chứng")
    f1 = st.sidebar.toggle("Sốt (Fever)")
    f2 = st.sidebar.toggle("Ho (Cough)")
    f3 = st.sidebar.toggle("Mệt mỏi (Fatigue)")
    f4 = st.sidebar.toggle("Khó thở (Shortness of Breath)")
    
    k_val = st.sidebar.slider("Chọn K (số hàng xóm)", 1, 21, 5, step=2)

    if st.button("Phân tích tình trạng"):
        user_input = [int(f1), int(f2), int(f3), int(f4)]
        result = predict_knn(user_input, train_data, k=k_val)
        
        st.divider()
        if result == "Cần nhập viện":
            st.error(f"### Dự đoán: {result}")
            st.write("Hệ thống nhận thấy triệu chứng của bạn giống với các ca cần chăm sóc y tế.")
        else:
            st.success(f"### Dự đoán: {result}")
            st.write("Các triệu chứng hiện tại có mức độ tương đồng thấp với các ca nhập viện.")
else:
    st.error(f"Không tìm thấy file dữ liệu tại: {csv_path}")
    st.info("Hãy đảm bảo bạn đã upload file CSV lên thư mục 'content' của Colab.")
