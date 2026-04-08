import streamlit as st
import pandas as pd
import math
from collections import Counter

# Chép lại hàm đã viết ở Cell 2 vào đây để app.py hiểu được
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
st.title("Trợ lý Sàng lọc Bệnh lý")

# Nạp dữ liệu trong app
df = pd.read_csv('covid_symptoms_severity_prediction.csv')
train_data = [{"features": [r['fever'], r['cough'], r['fatigue'], r['shortness_of_breath']], 
               "label": "Cần nhập viện" if r['hospitalized'] == 1 else "Theo dõi tại nhà"} 
              for _, r in df.iterrows()]

st.sidebar.header("Triệu chứng")
f1 = st.sidebar.checkbox("Sốt")
f2 = st.sidebar.checkbox("Ho")
f3 = st.sidebar.checkbox("Mệt mỏi")
f4 = st.sidebar.checkbox("Khó thở")

if st.button("Dự đoán"):
    result = predict_knn([int(f1), int(f2), int(f3), int(f4)], train_data)
    st.subheader(f"Kết quả: {result}")
