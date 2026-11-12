"""Giao diện Streamlit cho inference khách hàng mới."""
from pathlib import Path
import pandas as pd
import streamlit as st
from src.predict import load_model, predict_customer

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Customer Segmentation - Lab03", page_icon="🧩", layout="wide")
st.title("🧩 Customer Segmentation – K-means")
st.caption("Dự đoán cụm khách hàng mới bằng mô hình K-means đã huấn luyện trong pipeline Lab03.")

@st.cache_resource
def get_artifacts():
    return load_model()

try:
    model, scaler = get_artifacts()
    st.success(f"Đã tải mô hình K-means với K = {model.n_clusters}.")
except Exception as error:
    st.error(f"Không thể tải mô hình: {error}")
    st.info("Chạy `python -m src.pipeline` tại thư mục gốc Lab03 trước.")
    st.stop()

st.subheader("Thông tin khách hàng")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Tuổi (Age)", min_value=1, max_value=100, value=30)
with col2:
    income = st.number_input("Thu nhập hằng năm (Annual Income)", min_value=0, max_value=500000, value=75000, step=1000)
with col3:
    spending = st.slider("Điểm chi tiêu (Spending Score)", min_value=0, max_value=100, value=65)

st.caption("Hệ thống sử dụng đúng trung bình và độ lệch chuẩn đã lưu từ tập huấn luyện (Z-score).")
if st.button("🔍 Dự đoán cụm khách hàng", type="primary", use_container_width=True):
    try:
        result = predict_customer(age, income, spending, model=model, scaler=scaler)
        cluster = result["output"]["cluster"]
        st.success(f"Khách hàng được gán vào Cluster {cluster}")
        st.metric("Thời gian dự đoán", f"{result['output']['prediction_time_seconds']:.6f} giây")

        profile_path = ROOT / "data/processed/cluster_profile.csv"
        if profile_path.exists():
            profile = pd.read_csv(profile_path)
            selected = profile.loc[profile["Cluster"] == cluster]
            if not selected.empty:
                st.subheader(f"Đặc điểm của Cluster {cluster}")
                cols = [c for c in ["Age_mean", "Annual Income_mean", "Spending Score_mean"] if c in selected.columns]
                st.dataframe(selected[cols].rename(columns={
                    "Age_mean": "Tuổi trung bình",
                    "Annual Income_mean": "Thu nhập trung bình",
                    "Spending Score_mean": "Điểm chi tiêu trung bình"
                }).round(2), use_container_width=True, hide_index=True)

        with st.expander("Chi tiết inference"):
            st.write("**Đầu vào:**", result["input"])
            st.write("**Giá trị sau chuẩn hóa:**", result["output"]["scaled_features"])
            st.write("**Thời gian dự đoán:**", result["output"]["prediction_time_seconds"])
            st.caption("Cluster là mã nhóm do thuật toán tạo ra, không phải thứ hạng tốt/xấu.")
        st.info("Đã lưu lịch sử vào `reports/inference_history.csv`.")
    except Exception as error:
        st.error(f"Không thể dự đoán: {error}")

st.divider()
st.subheader("Lịch sử inference")
history_path = (ROOT / "reports" / "customer_segmentation" / "inference_history.csv")
if history_path.exists():
    history = pd.read_csv(history_path)
    st.dataframe(history.tail(20), use_container_width=True, hide_index=True)
else:
    st.caption("Chưa có lịch sử dự đoán.")
