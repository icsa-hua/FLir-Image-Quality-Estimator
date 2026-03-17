import numpy as np
import cv2
from drift_reports.base_drift_report import Report
from fliqe import FLIQE, OnlineFLIQE
import plotly.graph_objects as go
import pandas as pd


class VideoDriftReport(Report):
    """Video quality estimation report using OnlineFLIQE."""

    def __init__(self, video_path, quality_model_path="models/encoder_with_binary_head.pth", smoothing_window=150):
        self.video_path = video_path
        self.quality_model_path = quality_model_path
        self.smoothing_window = smoothing_window

    def _process_with_online_fliqe(self, sample_rate=5, progress_callback=None):
        """Process video frames using OnlineFLIQE (smoothed quality)."""
        fliqe = OnlineFLIQE(
            quality_model_path=self.quality_model_path,
            smoothing_window=self.smoothing_window,
        )
        fliqe.create_session("video_analysis")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        raw_scores = []
        smoothed_scores = []
        frame_indices = []
        sample_frames = []
        n_samples = 8
        sample_interval = max(1, total_frames // n_samples) if total_frames > 0 else 1

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                smoothed = fliqe.estimate_smoothed_quality(frame, session_id="video_analysis")
                raw = fliqe.get_raw_quality("video_analysis")
                raw_scores.append(raw)
                smoothed_scores.append(smoothed)
                frame_indices.append(frame_idx)

                if len(sample_frames) < n_samples and frame_idx % sample_interval < sample_rate:
                    sample_frames.append((frame.copy(), raw, smoothed, frame_idx))

            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_idx / total_frames)

        cap.release()

        return {
            "raw_scores": raw_scores,
            "smoothed_scores": smoothed_scores,
            "frame_indices": frame_indices,
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": len(raw_scores),
            "sample_frames": sample_frames,
        }

    def _process_with_fliqe(self, sample_rate=5, progress_callback=None):
        """Process video frames using FLIQE (per-frame quality, no smoothing)."""
        fliqe = FLIQE(quality_model_path=self.quality_model_path)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        raw_scores = []
        frame_indices = []
        sample_frames = []
        n_samples = 8
        sample_interval = max(1, total_frames // n_samples) if total_frames > 0 else 1

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                score = fliqe.estimate_image_quality(frame)
                raw_scores.append(score)
                frame_indices.append(frame_idx)

                if len(sample_frames) < n_samples and frame_idx % sample_interval < sample_rate:
                    sample_frames.append((frame.copy(), score, score, frame_idx))

            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_idx / total_frames)

        cap.release()

        return {
            "raw_scores": raw_scores,
            "smoothed_scores": raw_scores,
            "frame_indices": frame_indices,
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": len(raw_scores),
            "sample_frames": sample_frames,
        }

    def generate_streamlit_report(self, *args):
        import streamlit as st

        sample_rate = args[0] if len(args) > 0 else 5
        analysis_method = args[1] if len(args) > 1 else "OnlineFLIQE"

        title = "Video Quality Estimation Report"
        st.markdown(
            f"<h2 style=\"text-align: center; margin-bottom: 20px;\">{title} ({analysis_method})</h2>",
            unsafe_allow_html=True,
        )

        progress_bar = st.progress(0, text="Processing video frames...")
        def update_progress(pct):
            progress_bar.progress(min(pct, 1.0), text=f"Processing video frames... {pct:.0%}")

        if analysis_method == "OnlineFLIQE":
            results = self._process_with_online_fliqe(sample_rate=sample_rate, progress_callback=update_progress)
        else:
            results = self._process_with_fliqe(sample_rate=sample_rate, progress_callback=update_progress)
        progress_bar.empty()

        raw = results["raw_scores"]
        smoothed = results["smoothed_scores"]
        indices = results["frame_indices"]
        fps = results["fps"]
        total = results["total_frames"]
        processed = results["processed_frames"]
        samples = results["sample_frames"]

        if not raw:
            st.error("No frames could be processed from the video.")
            return

        # --- Summary Metrics ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Frames", total)
        col2.metric("Processed Frames", processed)
        col3.metric("Avg Raw Quality", f"{np.mean(raw):.4f}")
        if analysis_method == "OnlineFLIQE":
            col4.metric("Avg Smoothed Quality", f"{np.mean(smoothed):.4f}")
        else:
            col4.metric("Std Dev", f"{np.std(raw):.4f}")
        col5.metric("FPS", f"{fps:.1f}")

        st.divider()

        # --- Quality Timeline ---
        time_axis = [idx / fps if fps > 0 else idx for idx in indices]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=raw,
                mode="lines",
                name="Raw Quality",
                line=dict(width=1, color="lightblue"),
                opacity=0.7,
            )
        )
        if analysis_method == "OnlineFLIQE":
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=smoothed,
                    mode="lines",
                    name="Smoothed Quality",
                    line=dict(width=2, color="cornflowerblue"),
                )
            )
        fig.add_hline(
            y=np.mean(raw),
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Mean: {np.mean(raw):.4f}",
        )
        fig.update_layout(
            title="Quality Over Time",
            xaxis_title="Time (seconds)" if fps > 0 else "Frame Index",
            yaxis_title="Quality Score",
            height=400,
            margin=dict(t=40, b=0, l=0, r=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)

        # --- Distribution & Stats ---
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_hist = go.Figure()
            fig_hist.add_trace(
                go.Histogram(x=raw, nbinsx=30, name="Raw Scores", marker_color="cornflowerblue")
            )
            fig_hist.update_layout(
                title="Raw Quality Score Distribution",
                xaxis_title="Quality Score",
                yaxis_title="Count",
                height=300,
                margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_hist, config={"displayModeBar": False}, use_container_width=True)

        with chart_col2:
            stats_cols = {
                "Metric": ["Mean", "Std Dev", "Min", "Max", "Median", "Q1 (25%)", "Q3 (75%)"],
                "Raw Score": [
                    f"{np.mean(raw):.4f}",
                    f"{np.std(raw):.4f}",
                    f"{np.min(raw):.4f}",
                    f"{np.max(raw):.4f}",
                    f"{np.median(raw):.4f}",
                    f"{np.percentile(raw, 25):.4f}",
                    f"{np.percentile(raw, 75):.4f}",
                ],
            }
            if analysis_method == "OnlineFLIQE":
                stats_cols["Smoothed Score"] = [
                    f"{np.mean(smoothed):.4f}",
                    f"{np.std(smoothed):.4f}",
                    f"{np.min(smoothed):.4f}",
                    f"{np.max(smoothed):.4f}",
                    f"{np.median(smoothed):.4f}",
                    f"{np.percentile(smoothed, 25):.4f}",
                    f"{np.percentile(smoothed, 75):.4f}",
                ]
            st.dataframe(
                pd.DataFrame(stats_cols).set_index("Metric"),
                use_container_width=True,
            )

        st.divider()

        # --- Sample Frames ---
        if samples:
            st.subheader("Sample Frames")
            for row_start in range(0, len(samples), 4):
                row_samples = samples[row_start : row_start + 4]
                cols = st.columns(len(row_samples))
                for i, (frame, raw_s, smooth_s, fidx) in enumerate(row_samples):
                    with cols[i]:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if analysis_method == "OnlineFLIQE":
                            caption = f"Frame {fidx} | Raw: {raw_s:.3f} | Smooth: {smooth_s:.3f}"
                        else:
                            caption = f"Frame {fidx} | Score: {raw_s:.3f}"
                        st.image(img_rgb, caption=caption, use_container_width=True)
