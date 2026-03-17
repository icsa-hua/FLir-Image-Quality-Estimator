from dataclasses import dataclass
import pandas as pd
import plotly.graph_objects as go
from drift_reports.base_drift_report import Report
from alibi_detect.cd import ChiSquareDrift, TabularDrift
import numpy as np


@dataclass
class DriftDetectionResult:
    """Holds the results of drift detection."""
    is_drift: bool
    threshold: float
    p_val: list
    distance: list
    feature_names: list
    method: str


class TabularDriftReport(Report):

    PD_NUMERICAL_KEYS = ["number"]
    PD_CATEGORICAL_KEYS = ["category", "object", "string"]
    PD_DATETIME_KEYS = ["datetime"]

    def __init__(self, refe_data, test_data, dt_columns=None, cat_columns=None):
        if not isinstance(refe_data, pd.DataFrame):
            self.df_ref = pd.read_csv(refe_data)
        else:
            self.df_ref = refe_data
        if not isinstance(test_data, pd.DataFrame):
            self.df_test = pd.read_csv(test_data)
        else:
            self.df_test = test_data
        self.df_ref = self.df_ref.convert_dtypes()
        self.df_test = self.df_test.convert_dtypes()
        self.schema = self.df_test.dtypes
        self.common_features = self.df_ref.columns.intersection(self.df_test.columns)
        self.common_features = [col for col in self.common_features if self.df_ref[col].dtype == self.df_test[col].dtype]
        self.dt_columns = dt_columns
        self.cat_columns = cat_columns
        if dt_columns:
            self.cast_dt_columns(dt_columns)
        if cat_columns:
            self.cast_cat_columns(cat_columns)

    def detect_drift(self, method="Statistical", threshold=0.05):
        if method == "Statistical":
            categorical_features = self.df_ref.select_dtypes(include=self.PD_CATEGORICAL_KEYS).columns
            categorical_indeces = [self.df_ref.columns.get_loc(col) for col in categorical_features]
            categories_per_feature = {col_idx: None for col_idx in categorical_indeces}
            X_ref = self.df_ref[self.common_features].to_numpy()
            X_test = self.df_test[self.common_features].to_numpy()
            drift_detector = TabularDrift(X_ref, p_val=.05, categories_per_feature=categories_per_feature)
            preds = drift_detector.predict(X_test)
            return preds["data"]

    def cast_dt_columns(self, dt_columns):
        for col in dt_columns:
            self.df_ref[col] = pd.to_datetime(self.df_ref[col])
            self.df_test[col] = pd.to_datetime(self.df_test[col])

    def cast_cat_columns(self, cat_columns):
        for col in cat_columns:
            self.df_ref[col] = self.df_ref[col].astype("category")
            self.df_test[col] = self.df_test[col].astype("category")

    def df_overall_comparison(self):
        ref_summary = self.summarize_dataframe(self.df_ref)
        test_summary = self.summarize_dataframe(self.df_test)
        return pd.DataFrame([ref_summary, test_summary], index=["Reference", "Test"]).T

    def df_feature_comparison(self, feature_name):
        ref_stats = self.df_ref[feature_name].describe()
        test_stats = self.df_test[feature_name].describe()
        return pd.DataFrame([ref_stats, test_stats], index=["Reference", "Test"]).T

    def plot_hist_distibutions(self, feature_name, is_cumulative=False, barmode="overlay"):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=self.df_ref[feature_name], name="Reference", opacity=0.5, cumulative_enabled=is_cumulative))
        fig.add_trace(go.Histogram(x=self.df_test[feature_name], name="Test", opacity=0.5, cumulative_enabled=is_cumulative))
        fig.update_layout(
            barmode=barmode,
            height=300,
            margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        return fig

    def plot_ecdf_distributions(self, feature_name):
        x_ref = np.sort(self.df_ref[feature_name])
        y_ref = np.arange(1, len(x_ref) + 1) / len(x_ref)
        x_test = np.sort(self.df_test[feature_name])
        y_test = np.arange(1, len(x_test) + 1) / len(x_test)
        all_x = np.sort(np.concatenate([x_ref, x_test]))
        y_ref_interp = np.interp(all_x, x_ref, y_ref)
        y_test_interp = np.interp(all_x, x_test, y_test)
        diffs = np.abs(y_ref_interp - y_test_interp)
        max_diff = np.max(diffs)
        max_diff_idx = np.argmax(diffs)
        max_diff_x = all_x[max_diff_idx]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_ref, y=y_ref, name="Reference", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=x_test, y=y_test, name="Test", line=dict(width=2)))
        fig.add_trace(
            go.Scatter(
                x=[max_diff_x, max_diff_x],
                y=[y_ref_interp[max_diff_idx], y_test_interp[max_diff_idx]],
                mode="lines",
                line=dict(color="red", width=1, dash="dash"),
                name=f"Max abs diff (D={max_diff:.3f})",
            )
        )
        fig.update_layout(
            height=310,
            margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        return fig

    def plot_box_distributions(self, feature_name):
        """Box plot comparison for numerical features."""
        fig = go.Figure()
        fig.add_trace(go.Box(y=self.df_ref[feature_name], name="Reference", marker_color="cornflowerblue"))
        fig.add_trace(go.Box(y=self.df_test[feature_name], name="Test", marker_color="lightsalmon"))
        fig.update_layout(
            height=300,
            margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        return fig

    def plot_missing_values(self):
        """Bar chart showing missing values per feature."""
        ref_missing = self.df_ref[self.common_features].isna().sum()
        test_missing = self.df_test[self.common_features].isna().sum()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ref_missing.index, y=ref_missing.values, name="Reference", marker_color="cornflowerblue"))
        fig.add_trace(go.Bar(x=test_missing.index, y=test_missing.values, name="Test", marker_color="lightsalmon"))
        fig.update_layout(
            barmode="group",
            title="Missing Values per Feature",
            height=300,
            margin=dict(t=40, b=0, l=0, r=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        return fig

    def generate_streamlit_report(self, *args):
        drift_method = args[0] if args else "Statistical"
        import streamlit as st

        # --- Run drift detection up front ---
        with st.spinner("Running drift detection...", show_time=True):
            result = self.detect_drift(method=drift_method)

        drifted = [
            self.common_features[i]
            for i in range(len(self.common_features))
            if result["p_val"][i] < result["threshold"]
        ]
        n_drifted = len(drifted)
        n_features = len(self.common_features)

        # --- Header with drift status badge ---
        if result["is_drift"]:
            badge_color, badge_text = "#d32f2f", "DRIFT DETECTED"
        else:
            badge_color, badge_text = "#388e3c", "NO DRIFT"

        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
                <h2 style="margin:0;">Tabular Drift Report</h2>
                <span style="background:{badge_color}; color:white; padding:4px 14px;
                             border-radius:12px; font-size:0.85rem; font-weight:600;">
                    {badge_text}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Metric cards row ---
        ref_summary = self.summarize_dataframe(self.df_ref)
        test_summary = self.summarize_dataframe(self.df_test)
        ref_missing = self.df_ref[self.common_features].isna().sum().sum()
        test_missing = self.df_test[self.common_features].isna().sum().sum()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Reference rows", f"{ref_summary['Number of observations']:,}")
        m2.metric("Test rows", f"{test_summary['Number of observations']:,}")
        m3.metric("Common features", n_features)
        m4.metric("Features drifted", f"{n_drifted} / {n_features}")
        m5.metric("Missing cells", f"{ref_missing + test_missing:,}")

        # --- Drift summary bar ---
        if n_drifted > 0:
            st.warning(f"Drift in: {', '.join(drifted)}")
        else:
            st.success("All features are stable — no drift detected.")

        # --- Missing values chart (only if there are missing values) ---
        has_missing = ref_missing > 0 or test_missing > 0
        if has_missing:
            with st.expander("Missing values breakdown"):
                st.plotly_chart(
                    self.plot_missing_values(),
                    config={"displayModeBar": False},
                    use_container_width=True,
                )

        st.divider()

        # --- Per-Feature Analysis ---
        for idx, df_col in enumerate(self.common_features):
            feature_p_val = result["p_val"][idx]
            feature_distance = result["distance"][idx]
            feature_is_drift = "Yes" if feature_p_val < result["threshold"] else "No"
            st_col1, st_col2, st_col3 = st.columns([0.2, 0.4, 0.4])
            with st_col1:
                st.write(f"##### {df_col}")
                feature_info = pd.DataFrame(
                    [
                        ["Data type:", self.get_simplified_type(self.df_test, df_col)],
                        ["Distance:", feature_distance],
                        ["p-value:", feature_p_val],
                        ["Drift:", feature_is_drift],
                    ]
                )
                styled_df = (
                    feature_info.style.hide(axis="index")
                    .hide(axis="columns")
                    .map(self.apply_status_style)
                    .set_properties(**{"border": "none", "font-size": "11pt"})
                    .set_table_styles(
                        [dict(selector="td", props=[("padding-right", "40px"), ("padding-bottom", "5px")])]
                    )
                    .to_html()
                )
                st.markdown(styled_df, unsafe_allow_html=True)

            with st_col2:
                st.dataframe(self.df_feature_comparison(df_col))

            with st_col3:
                if self.get_simplified_type(self.df_test, df_col) == "Numerical":
                    tab_hist, tab_ecdf, tab_box = st.tabs(["Histogram", "ECDF", "Box Plot"])
                    with tab_hist:
                        st.plotly_chart(
                            self.plot_hist_distibutions(df_col),
                            config={"displayModeBar": False},
                            use_container_width=True,
                        )
                    with tab_ecdf:
                        st.plotly_chart(
                            self.plot_ecdf_distributions(df_col),
                            config={"displayModeBar": False},
                            use_container_width=True,
                        )
                    with tab_box:
                        st.plotly_chart(
                            self.plot_box_distributions(df_col),
                            config={"displayModeBar": False},
                            use_container_width=True,
                        )
                elif self.get_simplified_type(self.df_test, df_col) == "Categorical":
                    st.plotly_chart(
                        self.plot_hist_distibutions(df_col, barmode="group"),
                        config={"displayModeBar": False},
                        use_container_width=True,
                    )

            st.divider()

    @staticmethod
    def summarize_dataframe(df: pd.DataFrame):
        total_cells = df.shape[0] * df.shape[1]
        missing = df.isna().sum().sum()
        missing_pct = f" ({missing / total_cells * 100:.1f}%)" if total_cells > 0 else ""
        return {
            "Number of observations": df.shape[0],
            "Number of features": df.shape[1],
            "Missing cells": f"{missing}{missing_pct}",
            "Duplicate rows": df.duplicated().sum(),
            "Constant features": len([col for col in df.columns if df[col].nunique() == 1]),
            "Numerical features": len(df.select_dtypes(include=["number"]).columns),
            "Categorical features": len(df.select_dtypes(include=["category", "object", "string"]).columns),
        }

    @staticmethod
    def get_simplified_type(df: pd.DataFrame, df_col: str):
        if df_col in df.select_dtypes(include=TabularDriftReport.PD_NUMERICAL_KEYS).columns:
            return "Numerical"
        elif df_col in df.select_dtypes(include=TabularDriftReport.PD_CATEGORICAL_KEYS).columns:
            return "Categorical"
        elif df_col in df.select_dtypes(include=TabularDriftReport.PD_DATETIME_KEYS).columns:
            return "Datetime"
        else:
            return None

    @staticmethod
    def apply_status_style(val):
        if val == "No":
            return "color: green"
        elif val == "Yes":
            return "color: red"
