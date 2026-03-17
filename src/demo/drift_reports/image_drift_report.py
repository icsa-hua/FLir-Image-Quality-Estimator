import numpy as np
import cv2
from drift_reports.base_drift_report import Report
from fliqe import FLIQE
import plotly.graph_objects as go
import pandas as pd


class ImageDriftReport(Report):
    """Image quality estimation report using FLIQE."""

    def __init__(self, images, filenames=None, quality_model_path="models/encoder_with_binary_head.pth"):
        self.images = images
        self.filenames = filenames or [f"Image {i + 1}" for i in range(len(images))]
        self.fliqe = FLIQE(quality_model_path=quality_model_path)

    def estimate_quality_scores(self):
        """Estimate quality scores for all images using FLIQE."""
        scores = []
        for img in self.images:
            score = self.fliqe.estimate_image_quality(img)
            scores.append(score)
        return scores

    def generate_streamlit_report(self, *args):
        import streamlit as st

        st.markdown(
            "<h2 style=\"text-align: center; margin-bottom: 20px;\">Image Quality Estimation Report (FLIQE)</h2>",
            unsafe_allow_html=True,
        )

        with st.spinner("Estimating image quality...", show_time=True):
            scores = self.estimate_quality_scores()

        # --- Summary Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Images Analyzed", len(scores))
        col2.metric("Average Quality", f"{np.mean(scores):.4f}")
        col3.metric("Min Quality", f"{np.min(scores):.4f}")
        col4.metric("Max Quality", f"{np.max(scores):.4f}")

        st.divider()

        # --- Charts ---
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=scores,
                    nbinsx=min(20, max(5, len(scores))),
                    name="Quality Scores",
                    marker_color="cornflowerblue",
                )
            )
            fig.update_layout(
                title="Quality Score Distribution",
                xaxis_title="Quality Score",
                yaxis_title="Count",
                height=350,
                margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)

        with chart_col2:
            fig2 = go.Figure()
            colors = [self._score_to_color(s) for s in scores]
            fig2.add_trace(
                go.Bar(
                    x=list(range(len(scores))),
                    y=scores,
                    marker_color=colors,
                    name="Quality Score",
                )
            )
            fig2.add_hline(
                y=np.mean(scores),
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Mean: {np.mean(scores):.4f}",
            )
            fig2.update_layout(
                title="Quality Score per Image",
                xaxis_title="Image Index",
                yaxis_title="Quality Score",
                height=350,
                margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig2, config={"displayModeBar": False}, use_container_width=True)

        st.divider()

        # --- Quality Statistics ---
        stats_data = {
            "Metric": ["Mean", "Std Dev", "Min", "Max", "Median"],
            "Value": [
                f"{np.mean(scores):.4f}",
                f"{np.std(scores):.4f}",
                f"{np.min(scores):.4f}",
                f"{np.max(scores):.4f}",
                f"{np.median(scores):.4f}",
            ],
        }
        st.dataframe(pd.DataFrame(stats_data).set_index("Metric"), use_container_width=True)

        st.divider()

        # --- Images with Scores ---
        st.subheader("Images with Quality Scores")
        cols_per_row = 4
        for i in range(0, len(self.images), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(self.images):
                    with col:
                        img_rgb = cv2.cvtColor(self.images[idx], cv2.COLOR_BGR2RGB)
                        label = f"{self.filenames[idx]} — Score: {scores[idx]:.4f}"
                        st.image(img_rgb, caption=label, use_container_width=True)

    @staticmethod
    def _score_to_color(score):
        score = max(0.0, min(1.0, score))
        red = int((1 - score) * 255)
        green = int(score * 255)
        return f"rgb({red},{green},0)"
