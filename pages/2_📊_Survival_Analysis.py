from __future__ import annotations
from importlib import import_module
from typing import Callable, cast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import chi2_contingency
from pandas.io.formats.style import Styler
from enum import Enum


def to_percentage(df: pd.DataFrame) -> Styler:
    return df.apply(lambda col: col / col.sum()).mul(100).round(1).style.format("{}%")

def highlight_small_p(styler: Styler) -> Styler:
    return styler.highlight_between("p", left=0, right=0.0001)

def format_small_values(styler: Styler) -> Styler:
    """
    Format small values to '< 0.0001'
    """
    cols = [
        col
        for col, dtype in zip(styler.data.columns, styler.data.dtypes)
        if dtype == "float"
    ]
    for col in cols:
        subset = pd.IndexSlice[styler.data[col].between(0, 0.0001, inclusive="left"), col]
        styler = styler.format("< 0.0001", subset)
    return styler

class StreamlitEnum(str, Enum):
    def __eq__(self, other) -> bool:
        """Enable to check equality by value."""
        if not isinstance(other, StreamlitEnum):
            return NotImplemented
        return self.value == other.value

    @classmethod
    def to_list(cls) -> list[str]:
        return [e.value for e in cls]

class Gender(StreamlitEnum):
    ALL = "all"
    MALE = "male"
    FEMALE = "female"

def import_funcs_from_statistical_analysis() -> tuple[
    Callable, Callable, Callable, Callable
]:
    page = import_module(".1_📈_Statistical_Analysis", "pages")
    return (
        page.read_design_matrix,
        page.filter_gender,
        page.filter_age,
        page.crop_event,
    )

def make_km_section(
    design_matrix: pd.DataFrame,
) -> tuple[KaplanMeierFitter, KaplanMeierFitter]:
    design_matrix.attrs["predictor_col"] = "with_psychosis"
    predictor_col = design_matrix.attrs["predictor_col"]
    case = design_matrix.query(predictor_col)
    control = design_matrix.query(f"~{predictor_col}")

    case_fitter = KaplanMeierFitter()
    control_fitter = KaplanMeierFitter()
    event_col = design_matrix.attrs["event_col"]
    case_fitter.fit(case["T"], case[event_col], label="case")
    control_fitter.fit(control["T"], control[event_col], label="control")

    return case_fitter, control_fitter

def show_km_section(
    case_fitter: KaplanMeierFitter, control_fitter: KaplanMeierFitter
) -> None:
    fig, ax = plt.subplots()
    ax = case_fitter.plot_survival_function(ax=ax)
    ax = control_fitter.plot_survival_function(ax=ax)
    ax.set_xlabel("days")
    ax.set_ylabel("survival rate")
    ax.grid()
    st.markdown("")
    st.header("Kaplan-Meier Estimator")
    st.pyplot(fig)
    st.markdown("---")
    
def make_deconstructing_logrank_section(
    design_matrix: pd.DataFrame,
) -> pd.DataFrame:
    design_matrix.attrs["predictor_col"] = "with_psychosis"
    predictor_col = design_matrix.attrs["predictor_col"]
    case = design_matrix.query(predictor_col)
    control = design_matrix.query(f"~{predictor_col}")
    event_col = design_matrix.attrs["event_col"]
    case_have_event = case.query(event_col)
    control_have_event = control.query(event_col)

    def calc_cumulative_histogram(group: pd.DataFrame):
        max_duration = int(design_matrix["T"].max())
        bins = range(max_duration + 2)
        hist, _ = np.histogram(group["T"], bins)
        return np.cumsum(hist)

    nhappen_cum = pd.DataFrame(
        {
            "case": calc_cumulative_histogram(case_have_event),
            "control": calc_cumulative_histogram(control_have_event),
        }
    )
    total = len(case), len(control)
    nriskset_cum = total - nhappen_cum

    chi2values = []
    pvalues = []
    for observed in zip(nhappen_cum.values, nriskset_cum.values):
        try:
            chi2, p, _, _ = chi2_contingency(observed)
            chi2, p = cast(float, chi2), cast(float, p)
        except ValueError:
            chi2, p = 0, 0
        finally:
            chi2values.append(chi2)
            pvalues.append(p)

    chi2_df = pd.DataFrame(
        {
            "case_nhappen": nhappen_cum["case"],
            "control_nhappen": nhappen_cum["control"],
            "case_riskset": nriskset_cum["case"],
            "control_riskset": nriskset_cum["control"],
            "chi2": chi2values,
            "p": pvalues,
        }
    )

    assert chi2_df["case_nhappen"].iloc[-1] == len(case_have_event)
    assert chi2_df["control_nhappen"].iloc[-1] == len(control_have_event)

    assert chi2_df["case_nhappen"].idxmax() == case_have_event["T"].max()
    assert chi2_df["control_nhappen"].idxmax() == control_have_event["T"].max()

    return chi2_df

def show_deconstructing_logrank_section(chi2_df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    axs[0].plot(chi2_df["chi2"])
    axs[1].plot(chi2_df["p"])
    axs[1].axhline(
        0.05, color="red", label="$ p = 0.05 $", linewidth=0.8, linestyle="--"
    )
    axs[1].set_xlabel("days")
    axs[0].grid()
    axs[1].grid()
    axs[1].legend()
    axs[0].set_ylabel("chi2")
    axs[1].set_ylabel("p")

    # add percentage suffix
    nhappen = chi2_df[["case_nhappen", "control_nhappen"]]  # shorthand
    nhappen = (
        nhappen.astype(str)
        + "("
        + nhappen.div(nhappen.iloc[-1]).mul(100).round().astype(int).astype(str)
        + "%)"
    )
    chi2_df[["case_nhappen", "control_nhappen"]] = nhappen  # writeback

    st.header("Deconstructing Logrank Test")
    st.pyplot(fig)
    st.dataframe(chi2_df.style.pipe(format_small_values).pipe(highlight_small_p))
    st.markdown("---")

def make_cox_section(design_matrix: pd.DataFrame) -> tuple[CoxPHFitter, CoxPHFitter]:
    design_matrix = design_matrix.select_dtypes(exclude="datetime")
    if design_matrix.attrs["event_col"] != "E":
        design_matrix.drop(columns="E", inplace=True)

    # CoxPHFitter fails to converge when the values in gender column are all the same.
    if design_matrix.attrs["gender_selected"] != Gender.ALL:
        design_matrix = design_matrix.drop("gender", axis=1)

    kwargs = {
        "duration_col": "T",
        "event_col": design_matrix.attrs["event_col"],
        # adjust step_size if delta contains nan value
        "fit_options": {"step_size": 0.25},
    }

    covariates = [
        "hypertension",
        "heart_type_disease",
        "neurological_type_disease",
        "diabetes",
        "hyperlipidemia",
    ]
    design_matrix.attrs["with_covariate_cols"] = [f"with_{c}" for c in covariates]
    design_matrix.attrs["covariate_times_cols"] = [f"{c}_times" for c in covariates]

    with_covariate_fitter = CoxPHFitter()
    covariate_times_cols = design_matrix.attrs["covariate_times_cols"]
    with_covariate_fitter.fit(
        design_matrix.drop(columns=covariate_times_cols), **kwargs
    )

    covariate_times_fitter = CoxPHFitter()
    with_covariate_cols = design_matrix.attrs["with_covariate_cols"]
    covariate_times_fitter.fit(
        design_matrix.drop(columns=with_covariate_cols), **kwargs
    )

    return with_covariate_fitter, covariate_times_fitter

format_dict = {'exp(coef)': '{:.4f}', 'exp(coef) lower 95%': '{:.4f}', 'exp(coef) upper 95%': '{:.4f}', 'p': '{:.4f}'}

def show_cox_section(
    with_covariate_fitter: CoxPHFitter, *covariate_times_fitters: CoxPHFitter
) -> None:
    interested_cols = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]

    st.dataframe(
        with_covariate_fitter.summary[interested_cols].style.pipe(format_small_values).pipe(highlight_small_p),
    )

    for i, covariate_fitter in enumerate(covariate_times_fitters):
        st.dataframe(
            covariate_fitter.summary[interested_cols]
            .style.pipe(format_small_values)
            .pipe(highlight_small_p),
        )

    st.markdown("---")

def make_data_view_section(
    design_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    design_matrix.attrs["predictor_col"] = "with_psychosis"
    predictor_col = design_matrix.attrs["predictor_col"]
    case = design_matrix.query(predictor_col)
    control = design_matrix.query(f"~{predictor_col}")
    return case, control

def show_data_view_section(case_df: pd.DataFrame, control_df: pd.DataFrame) -> None:
    st.header("Data View")
    st.subheader("Case")
    if st.checkbox("show", key="case"):
        st.dataframe(case_df, height=600)
    st.subheader("Control")
    if st.checkbox("show", key="control"):
        st.dataframe(control_df, height=600)

if __name__ == "__main__":
    (
        read_design_matrix,
        filter_gender,
        filter_age,
        crop_event,
    ) = import_funcs_from_statistical_analysis()
    design_matrix = (
        read_design_matrix().pipe(filter_gender).pipe(filter_age).pipe(crop_event)
    )

    data_copy = design_matrix.copy()
    # Cut to columns of specified range
    design_matrix_df = data_copy.loc[:, 'gender':'hyperlipidemia_times']

    # Manually set a hidden anchor point
    st.markdown('<div id="section-1"></div>', unsafe_allow_html=True)
    show_km_section(*make_km_section(design_matrix))
    show_deconstructing_logrank_section(
        make_deconstructing_logrank_section(design_matrix)
    )

    st.header("Proportional Hazard Model")
    show_cox_section(*make_cox_section(design_matrix_df))
    show_data_view_section(*make_data_view_section(design_matrix))

    # Add a "Back to Top" button fixed in the lower right corner
    st.markdown("""
        <style>
        #go-top-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #007BFF;
            color: white;
            border: none;
            padding: 10px 15px;
            cursor: pointer;
            border-radius: 5px;
            font-size: 14px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            z-index: 9999;
        }
        </style>
        <a href="#section-1">
            <button id="go-top-button">Back to Top</button>
        </a>
    """, unsafe_allow_html=True)
