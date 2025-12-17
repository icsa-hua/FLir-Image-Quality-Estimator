import streamlit as st
import cv2
import numpy as np
from fliqe import OnlineFLIQE
import time
import os

st.set_page_config(
    page_title="FLIR Image Quality Estimator",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 FLIR Image Quality Estimator (FLIQE)")
st.markdown("Real-time video quality assessment using FLIQE")

# Configuration
VIDEO_PATH = "data/test_video.MP4"
MODEL_PATH = "models/encoder_with_binary_head.pth"
SMOOTHING_WINDOW = st.sidebar.slider("Smoothing Window", 1, 300, 150)

if not os.path.exists(VIDEO_PATH):
    st.error(f"Video file not found: {VIDEO_PATH}")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

# Initialize FLIQE
if 'fliqe' not in st.session_state:
    st.session_state.fliqe = OnlineFLIQE(
        quality_model_path=MODEL_PATH,
        smoothing_window=SMOOTHING_WINDOW
    )
    st.session_state.fliqe.create_session("demo_video")

# Update smoothing window if changed
if st.session_state.fliqe.smoothing_window != SMOOTHING_WINDOW:
    st.session_state.fliqe = OnlineFLIQE(
        quality_model_path=MODEL_PATH,
        smoothing_window=SMOOTHING_WINDOW
    )
    st.session_state.fliqe.create_session("demo_video")

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Video Player")
    video_placeholder = st.empty()
    frame_placeholder = st.empty()

with col2:
    st.subheader("Quality Metrics")
    smoothed_metric = st.empty()
    raw_metric = st.empty()
    frame_count_metric = st.empty()

# Control buttons
col_btn1, col_btn2, col_btn3 = st.columns(3)
with col_btn1:
    start_button = st.button("▶️ Start Processing", use_container_width=True)
with col_btn2:
    stop_button = st.button("⏹️ Stop", use_container_width=True)
with col_btn3:
    reset_button = st.button("🔄 Reset", use_container_width=True)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# Reset functionality
if reset_button:
    st.session_state.processing = False
    st.session_state.frame_count = 0
    st.session_state.fliqe.create_session("demo_video")
    st.rerun()

# Stop functionality
if stop_button:
    st.session_state.processing = False

# Video processing
if start_button:
    st.session_state.processing = True

if st.session_state.processing:
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        st.error("Cannot open video file")
        st.session_state.processing = False
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 0.033
        
        # Process video
        while st.session_state.processing:
            ret, frame = cap.read()
            
            if not ret:
                # Video ended, loop back
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                st.session_state.frame_count = 0
                continue
            
            # Estimate quality
            _ = st.session_state.fliqe.estimate_smoothed_quality(frame, session_id="demo_video")
            smoothed_score = st.session_state.fliqe.get_smoothed_quality("demo_video")
            raw_score = st.session_state.fliqe.get_raw_quality("demo_video")
            
            # Update metrics
            st.session_state.frame_count += 1
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update metrics
            smoothed_metric.metric(
                label="Smoothed Quality Score",
                value=f"{smoothed_score:.4f}" if smoothed_score is not None else "N/A",
                delta=None
            )
            raw_metric.metric(
                label="Raw Quality Score",
                value=f"{raw_score:.4f}" if raw_score is not None else "N/A",
                delta=None
            )
            frame_count_metric.metric(
                label="Frames Processed",
                value=st.session_state.frame_count
            )
            
            # Control playback speed
            time.sleep(frame_delay)
        
        cap.release()
else:
    # Show video file info when not processing
    with col1:
        st.video(VIDEO_PATH)
    
    with col2:
        st.info("Click '▶️ Start Processing' to begin quality estimation")
