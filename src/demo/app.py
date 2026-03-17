import streamlit as st
import sys
import os
import cv2
import numpy as np
import pandas as pd
import tempfile
import base64

# Add demo directory to path for drift_reports imports
sys.path.insert(0, os.path.dirname(__file__))

from drift_reports.tabular_drift_report import TabularDriftReport
from drift_reports.image_drift_report import ImageDriftReport
from drift_reports.video_drift_report import VideoDriftReport

MODEL_PATH = "models/encoder_with_binary_head.pth"

st.set_page_config(page_title="Data Quality Estimator", page_icon="\U0001f4ca", layout="wide")

# ---- Sidebar ----
with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "icsa_logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<img src="data:image/png;base64,{logo_b64}" style="width:50%; height:auto;">',
            unsafe_allow_html=True,
        )
    st.title("Data Quality Estimator")
    st.divider()

    data_type = st.selectbox("Data Type", ["Tabular", "Image", "Video"])

    if data_type == "Tabular":
        analysis_method = st.selectbox("Analysis Method", ["Statistical", "MMD"])
    else:
        analysis_method = st.selectbox("Analysis Method", ["FLIQE", "OnlineFLIQE"])

    st.divider()

    if data_type == "Tabular":
        st.markdown("**Upload Reference & Test Datasets**")
        ref_file = st.file_uploader("Reference Dataset", type=["csv", "xlsx"], key="ref_file")
        test_file = st.file_uploader("Test Dataset", type=["csv", "xlsx"], key="test_file")
    elif data_type == "Image":
        uploaded_files = st.file_uploader(
            "Upload Images",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
            accept_multiple_files=True,
            key="image_files",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov", "mkv", "wmv"],
            key="video_file",
        )

# ---- Main Content ----
if data_type == "Tabular":
    if ref_file and test_file:
        ref_df = pd.read_csv(ref_file)
        test_df = pd.read_csv(test_file)
        common_cols = list(ref_df.columns.intersection(test_df.columns))

        with st.sidebar:
            st.divider()
            dt_columns = st.multiselect("Datetime Columns", common_cols)
            remaining = [c for c in common_cols if c not in dt_columns]
            cat_columns = st.multiselect("Categorical Columns", remaining)

        report = TabularDriftReport(
            ref_df,
            test_df,
            dt_columns=dt_columns if dt_columns else None,
            cat_columns=cat_columns if cat_columns else None,
        )
        report.generate_streamlit_report(analysis_method)
    else:
        st.info("Please upload both a **Reference** and **Test** dataset in the sidebar to begin.")

elif data_type == "Image":
    if uploaded_files:
        images = []
        filenames = []
        for f in uploaded_files:
            bytes_data = f.read()
            nparr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
                filenames.append(f.name)
        if images:
            report = ImageDriftReport(images, filenames=filenames, quality_model_path=MODEL_PATH)
            report.generate_streamlit_report()
        else:
            st.error("Could not read any of the uploaded images.")
    else:
        st.info("Please upload one or more images in the sidebar to begin.")

elif data_type == "Video":
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        with st.sidebar:
            st.divider()
            sample_rate = st.slider(
                "Frame Sample Rate", 1, 30, 5, help="Process every Nth frame"
            )
            if analysis_method == "OnlineFLIQE":
                smoothing_window = st.slider("Smoothing Window", 10, 300, 150)
            else:
                smoothing_window = 150

        report = VideoDriftReport(
            tfile.name,
            quality_model_path=MODEL_PATH,
            smoothing_window=smoothing_window,
        )
        report.generate_streamlit_report(sample_rate, analysis_method)
        os.unlink(tfile.name)
    else:
        st.info("Please upload a video file in the sidebar to begin.")
