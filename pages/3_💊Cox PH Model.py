from __future__ import annotations
from importlib import import_module
from typing import Callable
import pandas as pd
import streamlit as st
from lifelines import CoxPHFitter
from pandas.io.formats.style import Styler
from enum import Enum
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


#讓網頁變寬
st.set_page_config(layout="wide")

def to_percentage(df: pd.DataFrame) -> Styler:
    return df.apply(lambda col: col / col.sum()).mul(100).round(1).style.format("{}%")

def highlight_small_p(styler: Styler) -> Styler:
    return styler.highlight_between("p", left=0, right=0.0001)

def highlight_last_col_small(styler: Styler) -> Styler:
    """
    Highlight cells in the last column of the DataFrame
    where 0 <= value < 0.0001.
    """
    last_col = styler.data.columns[-1]
    return styler.highlight_between(last_col, left=0, right=0.0001)

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

def make_cox_section(design_matrix: pd.DataFrame) -> tuple[CoxPHFitter, CoxPHFitter]:
    design_matrix = design_matrix.select_dtypes(exclude="datetime")
    if design_matrix.attrs["event_col"] != "E":
        design_matrix.drop(columns="E", inplace=True)

    # CoxPHFitter fails to converge when the values in gender column are all the same.
    # if design_matrix.attrs["gender_selected"] != Gender.ALL:
    #     design_matrix = design_matrix.drop("gender", axis=1)

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

def make_type2_anova(df: pd.DataFrame) -> pd.DataFrame:
    bool_cols = [
        "with_psychosis",
        "with_hypertension",
        "with_heart_type_disease",
        "with_neurological_type_disease",
        "with_diabetes",
        "with_hyperlipidemia",
    ]
    time_cols = [
        "with_psychosis",
        "hypertension_times",
        "heart_type_disease_times",
        "neurological_type_disease_times",
        "diabetes_times",
        "hyperlipidemia_times",
    ]

    def run_anova(cols: list[str]) -> pd.DataFrame:
        dfc = df.copy()
        dfc["E"] = dfc["E"].astype(int)
        for c in cols + ["gender", "age"]:
            dfc[c] = dfc[c].astype(float)
        dfc = dfc.dropna(subset=["E", "gender", "age"] + cols)

        formula = f"E ~ gender + age + {' + '.join(cols)}"
        lm = ols(formula, data=dfc).fit()
        # 整張表都回來，index = ['gender','age', ... cols]
        return sm.stats.anova_lm(lm, typ=2).drop(columns=["df"])

    # 拿回 full ANOVA
    anova_b = run_anova(bool_cols)
    anova_t = run_anova(time_cols)

    # 重命名欄位
    anova_b.columns = [f"{c}_bool" for c in anova_b.columns]
    anova_t.columns = [f"{c}_time" for c in anova_t.columns]

    # 外 join 保留所有 gender, age, with_* 以及 *_times
    return anova_b.join(anova_t, how="outer")


def make_cox_section_V2(design_matrix: pd.DataFrame) -> tuple[CoxPHFitter, ...]:
    # Initial processing
    design_matrix = design_matrix.select_dtypes(exclude="datetime")
    if design_matrix.attrs["event_col"] != "E":
        design_matrix.drop(columns="E", inplace=True)

    # CoxPHFitter fails to converge when the values in gender column are all the same.
    # if design_matrix.attrs["gender_selected"] != Gender.ALL:
    #     design_matrix = design_matrix.drop("gender", axis=1)

    kwargs = {
        "duration_col": "T",
        "event_col": design_matrix.attrs["event_col"],
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

    # Define operating rules for each type of data
    aspirin_variants = {
        "copy": ["aspirin_count", "aspirin_mean", "aspirin_max", "aspirin_min", "aspirin_median", "aspirin_hours_diff_mean", "aspirin_hours_diff_max", "aspirin_hours_diff_min", "aspirin_hours_diff_median"],
        "count": ["aspirin", "aspirin_mean", "aspirin_max", "aspirin_min", "aspirin_median", "aspirin_hours_diff_mean", "aspirin_hours_diff_max", "aspirin_hours_diff_min", "aspirin_hours_diff_median"],
        "mean": ["aspirin", "aspirin_count", "aspirin_max", "aspirin_min", "aspirin_median", "aspirin_hours_diff_mean", "aspirin_hours_diff_max", "aspirin_hours_diff_min", "aspirin_hours_diff_median"],
        "max": ["aspirin", "aspirin_count", "aspirin_mean", "aspirin_min", "aspirin_median", "aspirin_hours_diff_mean", "aspirin_hours_diff_max", "aspirin_hours_diff_min", "aspirin_hours_diff_median"],
        "min": ["aspirin", "aspirin_count", "aspirin_mean", "aspirin_max", "aspirin_median", "aspirin_hours_diff_mean", "aspirin_hours_diff_max", "aspirin_hours_diff_min", "aspirin_hours_diff_median"],
        "median": ["aspirin", "aspirin_count", "aspirin_mean", "aspirin_max", "aspirin_min", "aspirin_hours_diff_mean", "aspirin_hours_diff_max", "aspirin_hours_diff_min", "aspirin_hours_diff_median"],
        "hours_diff_mean": ["aspirin", "aspirin_count", "aspirin_mean", "aspirin_max", "aspirin_min", "aspirin_median", "aspirin_hours_diff_max", "aspirin_hours_diff_min", "aspirin_hours_diff_median"],
        "hours_diff_max": ["aspirin", "aspirin_count", "aspirin_mean", "aspirin_max", "aspirin_min", "aspirin_median", "aspirin_hours_diff_mean", "aspirin_hours_diff_min", "aspirin_hours_diff_median"],
        "hours_diff_min": ["aspirin", "aspirin_count", "aspirin_mean", "aspirin_max", "aspirin_min", "aspirin_median", "aspirin_hours_diff_mean", "aspirin_hours_diff_max", "aspirin_hours_diff_median"],
        "hours_diff_median": ["aspirin", "aspirin_count", "aspirin_mean", "aspirin_max", "aspirin_min", "aspirin_median", "aspirin_hours_diff_mean", "aspirin_hours_diff_max", "aspirin_hours_diff_min"],
    }

    fitters = []

    for variant, drop_cols in aspirin_variants.items():
        aspirin_data = design_matrix.copy().drop(columns=drop_cols)
        fitter = CoxPHFitter()
        
        # Select the field type to discard
        if variant == "copy":
            drop_cols = aspirin_data.attrs["covariate_times_cols"]
        else:
            drop_cols = aspirin_data.attrs["with_covariate_cols"]
        
        fitter.fit(aspirin_data.drop(columns=drop_cols), **kwargs)
        fitters.append(fitter)

    return tuple(fitters)



def make_cox_section_V3(design_matrix: pd.DataFrame) -> tuple[CoxPHFitter, ...]:
    # Initial processing
    design_matrix = design_matrix.select_dtypes(exclude="datetime")
    if design_matrix.attrs["event_col"] != "E":
        design_matrix.drop(columns="E", inplace=True)

    kwargs = {
        "duration_col": "T",
        "event_col": design_matrix.attrs["event_col"],
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

    # Define operating rules for each type of data
    warfarin_variants = {
        "copy": ["warfarin_count", "warfarin_mean", "warfarin_max", "warfarin_min", "warfarin_median", "warfarin_hours_diff_mean", "warfarin_hours_diff_max", "warfarin_hours_diff_min", "warfarin_hours_diff_median"],
        "count": ["warfarin", "warfarin_mean", "warfarin_max", "warfarin_min", "warfarin_median", "warfarin_hours_diff_mean", "warfarin_hours_diff_max", "warfarin_hours_diff_min", "warfarin_hours_diff_median"],
        "mean": ["warfarin", "warfarin_count", "warfarin_max", "warfarin_min", "warfarin_median", "warfarin_hours_diff_mean", "warfarin_hours_diff_max", "warfarin_hours_diff_min", "warfarin_hours_diff_median"],
        "max": ["warfarin", "warfarin_count", "warfarin_mean", "warfarin_min", "warfarin_median", "warfarin_hours_diff_mean", "warfarin_hours_diff_max", "warfarin_hours_diff_min", "warfarin_hours_diff_median"],
        "min": ["warfarin", "warfarin_count", "warfarin_mean", "warfarin_max", "warfarin_median", "warfarin_hours_diff_mean", "warfarin_hours_diff_max", "warfarin_hours_diff_min", "warfarin_hours_diff_median"],
        "median": ["warfarin", "warfarin_count", "warfarin_mean", "warfarin_max", "warfarin_min", "warfarin_hours_diff_mean", "warfarin_hours_diff_max", "warfarin_hours_diff_min", "warfarin_hours_diff_median"],
        "hours_diff_mean": ["warfarin", "warfarin_count", "warfarin_mean", "warfarin_max", "warfarin_min", "warfarin_median", "warfarin_hours_diff_max", "warfarin_hours_diff_min", "warfarin_hours_diff_median"],
        "hours_diff_max": ["warfarin", "warfarin_count", "warfarin_mean", "warfarin_max", "warfarin_min", "warfarin_median", "warfarin_hours_diff_mean", "warfarin_hours_diff_min", "warfarin_hours_diff_median"],
        "hours_diff_min": ["warfarin", "warfarin_count", "warfarin_mean", "warfarin_max", "warfarin_min", "warfarin_median", "warfarin_hours_diff_mean", "warfarin_hours_diff_max", "warfarin_hours_diff_median"],
        "hours_diff_median": ["warfarin", "warfarin_count", "warfarin_mean", "warfarin_max", "warfarin_min", "warfarin_median", "warfarin_hours_diff_mean", "warfarin_hours_diff_max", "warfarin_hours_diff_min"],
    }

    fitters = []

    for variant, drop_cols in warfarin_variants.items():
        warfarin_data = design_matrix.copy().drop(columns=drop_cols)
        fitter = CoxPHFitter()
        
        # Select the field type to discard
        if variant == "copy":
            drop_cols = warfarin_data.attrs["covariate_times_cols"]
        else:
            drop_cols = warfarin_data.attrs["with_covariate_cols"]
        
        fitter.fit(warfarin_data.drop(columns=drop_cols), **kwargs)
        fitters.append(fitter)

    return tuple(fitters)

def make_cox_section_V4(design_matrix: pd.DataFrame) -> tuple[CoxPHFitter, ...]:
    # Initial processing
    design_matrix = design_matrix.select_dtypes(exclude="datetime")
    if design_matrix.attrs["event_col"] != "E":
        design_matrix.drop(columns="E", inplace=True)

    kwargs = {
        "duration_col": "T",
        "event_col": design_matrix.attrs["event_col"],
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

    # Define operating rules for each type of data
    clopidogrel_variants = {
        "copy": ["clopidogrel_count", "clopidogrel_mean", "clopidogrel_max", "clopidogrel_min", "clopidogrel_median", "clopidogrel_hours_diff_mean", "clopidogrel_hours_diff_max", "clopidogrel_hours_diff_min", "clopidogrel_hours_diff_median"],
        "count": ["clopidogrel", "clopidogrel_mean", "clopidogrel_max", "clopidogrel_min", "clopidogrel_median", "clopidogrel_hours_diff_mean", "clopidogrel_hours_diff_max", "clopidogrel_hours_diff_min", "clopidogrel_hours_diff_median"],
        "mean": ["clopidogrel", "clopidogrel_count", "clopidogrel_max", "clopidogrel_min", "clopidogrel_median", "clopidogrel_hours_diff_mean", "clopidogrel_hours_diff_max", "clopidogrel_hours_diff_min", "clopidogrel_hours_diff_median"],
        "max": ["clopidogrel", "clopidogrel_count", "clopidogrel_mean", "clopidogrel_min", "clopidogrel_median", "clopidogrel_hours_diff_mean", "clopidogrel_hours_diff_max", "clopidogrel_hours_diff_min", "clopidogrel_hours_diff_median"],
        "min": ["clopidogrel", "clopidogrel_count", "clopidogrel_mean", "clopidogrel_max", "clopidogrel_median", "clopidogrel_hours_diff_mean", "clopidogrel_hours_diff_max", "clopidogrel_hours_diff_min", "clopidogrel_hours_diff_median"],
        "median": ["clopidogrel", "clopidogrel_count", "clopidogrel_mean", "clopidogrel_max", "clopidogrel_min", "clopidogrel_hours_diff_mean", "clopidogrel_hours_diff_max", "clopidogrel_hours_diff_min", "clopidogrel_hours_diff_median"],
        "hours_diff_mean": ["clopidogrel", "clopidogrel_count", "clopidogrel_mean", "clopidogrel_max", "clopidogrel_min", "clopidogrel_median", "clopidogrel_hours_diff_max", "clopidogrel_hours_diff_min", "clopidogrel_hours_diff_median"],
        "hours_diff_max": ["clopidogrel", "clopidogrel_count", "clopidogrel_mean", "clopidogrel_max", "clopidogrel_min", "clopidogrel_median", "clopidogrel_hours_diff_mean", "clopidogrel_hours_diff_min", "clopidogrel_hours_diff_median"],
        "hours_diff_min": ["clopidogrel", "clopidogrel_count", "clopidogrel_mean", "clopidogrel_max", "clopidogrel_min", "clopidogrel_median", "clopidogrel_hours_diff_mean", "clopidogrel_hours_diff_max", "clopidogrel_hours_diff_median"],
        "hours_diff_median": ["clopidogrel", "clopidogrel_count", "clopidogrel_mean", "clopidogrel_max", "clopidogrel_min", "clopidogrel_median", "clopidogrel_hours_diff_mean", "clopidogrel_hours_diff_max", "clopidogrel_hours_diff_min"],
    }

    fitters = []

    for variant, drop_cols in clopidogrel_variants.items():
        clopidogrel_data = design_matrix.copy().drop(columns=drop_cols)
        fitter = CoxPHFitter()
        
        # Select the field type to discard
        if variant == "copy":
            drop_cols = clopidogrel_data.attrs["covariate_times_cols"]
        else:
            drop_cols = clopidogrel_data.attrs["with_covariate_cols"]
        
        fitter.fit(clopidogrel_data.drop(columns=drop_cols), **kwargs)
        fitters.append(fitter)

    return tuple(fitters)

def make_cox_section_V5(design_matrix: pd.DataFrame) -> tuple[CoxPHFitter, ...]:
    # Initial processing
    design_matrix = design_matrix.select_dtypes(exclude="datetime")
    if design_matrix.attrs["event_col"] != "E":
        design_matrix.drop(columns="E", inplace=True)

    kwargs = {
        "duration_col": "T",
        "event_col": design_matrix.attrs["event_col"],
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

    # Define operating rules for each type of data
    apixaban_variants = {
        "copy": ["apixaban_count", "apixaban_mean", "apixaban_max", "apixaban_min", "apixaban_median", "apixaban_hours_diff_mean", "apixaban_hours_diff_max", "apixaban_hours_diff_min", "apixaban_hours_diff_median"],
        "count": ["apixaban", "apixaban_mean", "apixaban_max", "apixaban_min", "apixaban_median", "apixaban_hours_diff_mean", "apixaban_hours_diff_max", "apixaban_hours_diff_min", "apixaban_hours_diff_median"],
        "mean": ["apixaban", "apixaban_count", "apixaban_max", "apixaban_min", "apixaban_median", "apixaban_hours_diff_mean", "apixaban_hours_diff_max", "apixaban_hours_diff_min", "apixaban_hours_diff_median"],
        "max": ["apixaban", "apixaban_count", "apixaban_mean", "apixaban_min", "apixaban_median", "apixaban_hours_diff_mean", "apixaban_hours_diff_max", "apixaban_hours_diff_min", "apixaban_hours_diff_median"],
        "min": ["apixaban", "apixaban_count", "apixaban_mean", "apixaban_max", "apixaban_median", "apixaban_hours_diff_mean", "apixaban_hours_diff_max", "apixaban_hours_diff_min", "apixaban_hours_diff_median"],
        "median": ["apixaban", "apixaban_count", "apixaban_mean", "apixaban_max", "apixaban_min", "apixaban_hours_diff_mean", "apixaban_hours_diff_max", "apixaban_hours_diff_min", "apixaban_hours_diff_median"],
        "hours_diff_mean": ["apixaban", "apixaban_count", "apixaban_mean", "apixaban_max", "apixaban_min", "apixaban_median", "apixaban_hours_diff_max", "apixaban_hours_diff_min", "apixaban_hours_diff_median"],
        "hours_diff_max": ["apixaban", "apixaban_count", "apixaban_mean", "apixaban_max", "apixaban_min", "apixaban_median", "apixaban_hours_diff_mean", "apixaban_hours_diff_min", "apixaban_hours_diff_median"],
        "hours_diff_min": ["apixaban", "apixaban_count", "apixaban_mean", "apixaban_max", "apixaban_min", "apixaban_median", "apixaban_hours_diff_mean", "apixaban_hours_diff_max", "apixaban_hours_diff_median"],
        "hours_diff_median": ["apixaban", "apixaban_count", "apixaban_mean", "apixaban_max", "apixaban_min", "apixaban_median", "apixaban_hours_diff_mean", "apixaban_hours_diff_max", "apixaban_hours_diff_min"],
    }

    fitters = []

    for variant, drop_cols in apixaban_variants.items():
        apixaban_data = design_matrix.copy().drop(columns=drop_cols)
        fitter = CoxPHFitter()
        
        # Select the field type to discard
        if variant == "copy":
            drop_cols = apixaban_data.attrs["covariate_times_cols"]
        else:
            drop_cols = apixaban_data.attrs["with_covariate_cols"]
        
        fitter.fit(apixaban_data.drop(columns=drop_cols), **kwargs)
        fitters.append(fitter)

    return tuple(fitters)

def make_cox_section_V6(design_matrix: pd.DataFrame) -> tuple[CoxPHFitter, ...]:
    # Initial processing
    design_matrix = design_matrix.select_dtypes(exclude="datetime")
    if design_matrix.attrs["event_col"] != "E":
        design_matrix.drop(columns="E", inplace=True)

    kwargs = {
        "duration_col": "T",
        "event_col": design_matrix.attrs["event_col"],
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

    # Define operating rules for each type of data
    rivaroxaban_variants = {
        "copy": ["rivaroxaban_count", "rivaroxaban_mean", "rivaroxaban_max", "rivaroxaban_min", "rivaroxaban_median", "rivaroxaban_hours_diff_mean", "rivaroxaban_hours_diff_max", "rivaroxaban_hours_diff_min", "rivaroxaban_hours_diff_median"],
        "count": ["rivaroxaban", "rivaroxaban_mean", "rivaroxaban_max", "rivaroxaban_min", "rivaroxaban_median", "rivaroxaban_hours_diff_mean", "rivaroxaban_hours_diff_max", "rivaroxaban_hours_diff_min", "rivaroxaban_hours_diff_median"],
        "mean": ["rivaroxaban", "rivaroxaban_count", "rivaroxaban_max", "rivaroxaban_min", "rivaroxaban_median", "rivaroxaban_hours_diff_mean", "rivaroxaban_hours_diff_max", "rivaroxaban_hours_diff_min", "rivaroxaban_hours_diff_median"],
        "max": ["rivaroxaban", "rivaroxaban_count", "rivaroxaban_mean", "rivaroxaban_min", "rivaroxaban_median", "rivaroxaban_hours_diff_mean", "rivaroxaban_hours_diff_max", "rivaroxaban_hours_diff_min", "rivaroxaban_hours_diff_median"],
        "min": ["rivaroxaban", "rivaroxaban_count", "rivaroxaban_mean", "rivaroxaban_max", "rivaroxaban_median", "rivaroxaban_hours_diff_mean", "rivaroxaban_hours_diff_max", "rivaroxaban_hours_diff_min", "rivaroxaban_hours_diff_median"],
        "median": ["rivaroxaban", "rivaroxaban_count", "rivaroxaban_mean", "rivaroxaban_max", "rivaroxaban_min", "rivaroxaban_hours_diff_mean", "rivaroxaban_hours_diff_max", "rivaroxaban_hours_diff_min", "rivaroxaban_hours_diff_median"],
        "hours_diff_mean": ["rivaroxaban", "rivaroxaban_count", "rivaroxaban_mean", "rivaroxaban_max", "rivaroxaban_min", "rivaroxaban_median", "rivaroxaban_hours_diff_max", "rivaroxaban_hours_diff_min", "rivaroxaban_hours_diff_median"],
        "hours_diff_max": ["rivaroxaban", "rivaroxaban_count", "rivaroxaban_mean", "rivaroxaban_max", "rivaroxaban_min", "rivaroxaban_median", "rivaroxaban_hours_diff_mean", "rivaroxaban_hours_diff_min", "rivaroxaban_hours_diff_median"],
        "hours_diff_min": ["rivaroxaban", "rivaroxaban_count", "rivaroxaban_mean", "rivaroxaban_max", "rivaroxaban_min", "rivaroxaban_median", "rivaroxaban_hours_diff_mean", "rivaroxaban_hours_diff_max", "rivaroxaban_hours_diff_median"],
        "hours_diff_median": ["rivaroxaban", "rivaroxaban_count", "rivaroxaban_mean", "rivaroxaban_max", "rivaroxaban_min", "rivaroxaban_median", "rivaroxaban_hours_diff_mean", "rivaroxaban_hours_diff_max", "rivaroxaban_hours_diff_min"],
    }

    fitters = []

    for variant, drop_cols in rivaroxaban_variants.items():
        rivaroxaban_data = design_matrix.copy().drop(columns=drop_cols)
        fitter = CoxPHFitter()
        
        # Select the field type to discard
        if variant == "copy":
            drop_cols = rivaroxaban_data.attrs["covariate_times_cols"]
        else:
            drop_cols = rivaroxaban_data.attrs["with_covariate_cols"]
        
        fitter.fit(rivaroxaban_data.drop(columns=drop_cols), **kwargs)
        fitters.append(fitter)

    return tuple(fitters)


def make_cox_section_V7(design_matrix: pd.DataFrame) -> tuple[CoxPHFitter, ...]:
    # Initial processing
    design_matrix = design_matrix.select_dtypes(exclude="datetime")
    if design_matrix.attrs["event_col"] != "E":
        design_matrix.drop(columns="E", inplace=True)

    kwargs = {
        "duration_col": "T",
        "event_col": design_matrix.attrs["event_col"],
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

    # Define operating rules for each type of data
    dabigatran_etexilate_variants = {
        "copy": ["dabigatran etexilate_count", "dabigatran etexilate_mean", "dabigatran etexilate_max", "dabigatran etexilate_min", "dabigatran etexilate_median", "dabigatran etexilate_hours_diff_mean", "dabigatran etexilate_hours_diff_max", "dabigatran etexilate_hours_diff_min", "dabigatran etexilate_hours_diff_median"],
        "count": ["dabigatran etexilate", "dabigatran etexilate_mean", "dabigatran etexilate_max", "dabigatran etexilate_min", "dabigatran etexilate_median", "dabigatran etexilate_hours_diff_mean", "dabigatran etexilate_hours_diff_max", "dabigatran etexilate_hours_diff_min", "dabigatran etexilate_hours_diff_median"],
        "mean": ["dabigatran etexilate", "dabigatran etexilate_count", "dabigatran etexilate_max", "dabigatran etexilate_min", "dabigatran etexilate_median", "dabigatran etexilate_hours_diff_mean", "dabigatran etexilate_hours_diff_max", "dabigatran etexilate_hours_diff_min", "dabigatran etexilate_hours_diff_median"],
        "max": ["dabigatran etexilate", "dabigatran etexilate_count", "dabigatran etexilate_mean", "dabigatran etexilate_min", "dabigatran etexilate_median", "dabigatran etexilate_hours_diff_mean", "dabigatran etexilate_hours_diff_max", "dabigatran etexilate_hours_diff_min", "dabigatran etexilate_hours_diff_median"],
        "min": ["dabigatran etexilate", "dabigatran etexilate_count", "dabigatran etexilate_mean", "dabigatran etexilate_max", "dabigatran etexilate_median", "dabigatran etexilate_hours_diff_mean", "dabigatran etexilate_hours_diff_max", "dabigatran etexilate_hours_diff_min", "dabigatran etexilate_hours_diff_median"],
        "median": ["dabigatran etexilate", "dabigatran etexilate_count", "dabigatran etexilate_mean", "dabigatran etexilate_max", "dabigatran etexilate_min", "dabigatran etexilate_hours_diff_mean", "dabigatran etexilate_hours_diff_max", "dabigatran etexilate_hours_diff_min", "dabigatran etexilate_hours_diff_median"],
        "hours_diff_mean": ["dabigatran etexilate", "dabigatran etexilate_count", "dabigatran etexilate_mean", "dabigatran etexilate_max", "dabigatran etexilate_min", "dabigatran etexilate_median", "dabigatran etexilate_hours_diff_max", "dabigatran etexilate_hours_diff_min", "dabigatran etexilate_hours_diff_median"],
        "hours_diff_max": ["dabigatran etexilate", "dabigatran etexilate_count", "dabigatran etexilate_mean", "dabigatran etexilate_max", "dabigatran etexilate_min", "dabigatran etexilate_median", "dabigatran etexilate_hours_diff_mean", "dabigatran etexilate_hours_diff_min", "dabigatran etexilate_hours_diff_median"],
        "hours_diff_min": ["dabigatran etexilate", "dabigatran etexilate_count", "dabigatran etexilate_mean", "dabigatran etexilate_max", "dabigatran etexilate_min", "dabigatran etexilate_median", "dabigatran etexilate_hours_diff_mean", "dabigatran etexilate_hours_diff_max", "dabigatran etexilate_hours_diff_median"],
        "hours_diff_median": ["dabigatran etexilate", "dabigatran etexilate_count", "dabigatran etexilate_mean", "dabigatran etexilate_max", "dabigatran etexilate_min", "dabigatran etexilate_median", "dabigatran etexilate_hours_diff_mean", "dabigatran etexilate_hours_diff_max", "dabigatran etexilate_hours_diff_min"],
    }

    fitters = []

    for variant, drop_cols in dabigatran_etexilate_variants.items():
        dabigatran_etexilate_variants = design_matrix.copy().drop(columns=drop_cols)
        fitter = CoxPHFitter()
        
        # Select the field type to discard
        if variant == "copy":
            drop_cols = dabigatran_etexilate_variants.attrs["covariate_times_cols"]
        else:
            drop_cols = dabigatran_etexilate_variants.attrs["with_covariate_cols"]
        
        fitter.fit(dabigatran_etexilate_variants.drop(columns=drop_cols), **kwargs)
        fitters.append(fitter)

    return tuple(fitters)

def make_cox_section_V8(design_matrix: pd.DataFrame) -> tuple[CoxPHFitter, ...]:
    # Initial processing
    design_matrix = design_matrix.select_dtypes(exclude="datetime")
    if design_matrix.attrs["event_col"] != "E":
        design_matrix.drop(columns="E", inplace=True)

    kwargs = {
        "duration_col": "T",
        "event_col": design_matrix.attrs["event_col"],
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

    # Define operating rules for each type of data
    cilostazol_variants = {
        "copy": ["cilostazol_count", "cilostazol_mean", "cilostazol_max", "cilostazol_min", "cilostazol_median", "cilostazol_hours_diff_mean", "cilostazol_hours_diff_max", "cilostazol_hours_diff_min", "cilostazol_hours_diff_median"],
        "count": ["cilostazol", "cilostazol_mean", "cilostazol_max", "cilostazol_min", "cilostazol_median", "cilostazol_hours_diff_mean", "cilostazol_hours_diff_max", "cilostazol_hours_diff_min", "cilostazol_hours_diff_median"],
        "mean": ["cilostazol", "cilostazol_count", "cilostazol_max", "cilostazol_min", "cilostazol_median", "cilostazol_hours_diff_mean", "cilostazol_hours_diff_max", "cilostazol_hours_diff_min", "cilostazol_hours_diff_median"],
        "max": ["cilostazol", "cilostazol_count", "cilostazol_mean", "cilostazol_min", "cilostazol_median", "cilostazol_hours_diff_mean", "cilostazol_hours_diff_max", "cilostazol_hours_diff_min", "cilostazol_hours_diff_median"],
        "min": ["cilostazol", "cilostazol_count", "cilostazol_mean", "cilostazol_max", "cilostazol_median", "cilostazol_hours_diff_mean", "cilostazol_hours_diff_max", "cilostazol_hours_diff_min", "cilostazol_hours_diff_median"],
        "median": ["cilostazol", "cilostazol_count", "cilostazol_mean", "cilostazol_max", "cilostazol_min", "cilostazol_hours_diff_mean", "cilostazol_hours_diff_max", "cilostazol_hours_diff_min", "cilostazol_hours_diff_median"],
        "hours_diff_mean": ["cilostazol", "cilostazol_count", "cilostazol_mean", "cilostazol_max", "cilostazol_min", "cilostazol_median", "cilostazol_hours_diff_max", "cilostazol_hours_diff_min", "cilostazol_hours_diff_median"],
        "hours_diff_max": ["cilostazol", "cilostazol_count", "cilostazol_mean", "cilostazol_max", "cilostazol_min", "cilostazol_median", "cilostazol_hours_diff_mean", "cilostazol_hours_diff_min", "cilostazol_hours_diff_median"],
        "hours_diff_min": ["cilostazol", "cilostazol_count", "cilostazol_mean", "cilostazol_max", "cilostazol_min", "cilostazol_median", "cilostazol_hours_diff_mean", "cilostazol_hours_diff_max", "cilostazol_hours_diff_median"],
        "hours_diff_median": ["cilostazol", "cilostazol_count", "cilostazol_mean", "cilostazol_max", "cilostazol_min", "cilostazol_median", "cilostazol_hours_diff_mean", "cilostazol_hours_diff_max", "cilostazol_hours_diff_min"],
    }

    fitters = []

    for variant, drop_cols in cilostazol_variants.items():
        cilostazol_data = design_matrix.copy().drop(columns=drop_cols)
        fitter = CoxPHFitter()
        
        # Select the field type to discard
        if variant == "copy":
            drop_cols = cilostazol_data.attrs["covariate_times_cols"]
        else:
            drop_cols = cilostazol_data.attrs["with_covariate_cols"]
        
        fitter.fit(cilostazol_data.drop(columns=drop_cols), **kwargs)
        fitters.append(fitter)

    return tuple(fitters)

def make_cox_section_V9(design_matrix: pd.DataFrame) -> tuple[CoxPHFitter, ...]:
    # Initial processing
    design_matrix = design_matrix.select_dtypes(exclude="datetime")
    if design_matrix.attrs["event_col"] != "E":
        design_matrix.drop(columns="E", inplace=True)

    kwargs = {
        "duration_col": "T",
        "event_col": design_matrix.attrs["event_col"],
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

    # Define operating rules for each type of data
    enoxaparin_variants = {
        "copy": ["enoxaparin_count", "enoxaparin_mean", "enoxaparin_max", "enoxaparin_min", "enoxaparin_median", "enoxaparin_hours_diff_mean", "enoxaparin_hours_diff_max", "enoxaparin_hours_diff_min", "enoxaparin_hours_diff_median"],
        "count": ["enoxaparin", "enoxaparin_mean", "enoxaparin_max", "enoxaparin_min", "enoxaparin_median", "enoxaparin_hours_diff_mean", "enoxaparin_hours_diff_max", "enoxaparin_hours_diff_min", "enoxaparin_hours_diff_median"],
        "mean": ["enoxaparin", "enoxaparin_count", "enoxaparin_max", "enoxaparin_min", "enoxaparin_median", "enoxaparin_hours_diff_mean", "enoxaparin_hours_diff_max", "enoxaparin_hours_diff_min", "enoxaparin_hours_diff_median"],
        "max": ["enoxaparin", "enoxaparin_count", "enoxaparin_mean", "enoxaparin_min", "enoxaparin_median", "enoxaparin_hours_diff_mean", "enoxaparin_hours_diff_max", "enoxaparin_hours_diff_min", "enoxaparin_hours_diff_median"],
        "min": ["enoxaparin", "enoxaparin_count", "enoxaparin_mean", "enoxaparin_max", "enoxaparin_median", "enoxaparin_hours_diff_mean", "enoxaparin_hours_diff_max", "enoxaparin_hours_diff_min", "enoxaparin_hours_diff_median"],
        "median": ["enoxaparin", "enoxaparin_count", "enoxaparin_mean", "enoxaparin_max", "enoxaparin_min", "enoxaparin_hours_diff_mean", "enoxaparin_hours_diff_max", "enoxaparin_hours_diff_min", "enoxaparin_hours_diff_median"],
        "hours_diff_mean": ["enoxaparin", "enoxaparin_count", "enoxaparin_mean", "enoxaparin_max", "enoxaparin_min", "enoxaparin_median", "enoxaparin_hours_diff_max", "enoxaparin_hours_diff_min", "enoxaparin_hours_diff_median"],
        "hours_diff_max": ["enoxaparin", "enoxaparin_count", "enoxaparin_mean", "enoxaparin_max", "enoxaparin_min", "enoxaparin_median", "enoxaparin_hours_diff_mean", "enoxaparin_hours_diff_min", "enoxaparin_hours_diff_median"],
        "hours_diff_min": ["enoxaparin", "enoxaparin_count", "enoxaparin_mean", "enoxaparin_max", "enoxaparin_min", "enoxaparin_median", "enoxaparin_hours_diff_mean", "enoxaparin_hours_diff_max", "enoxaparin_hours_diff_median"],
        "hours_diff_median": ["enoxaparin", "enoxaparin_count", "enoxaparin_mean", "enoxaparin_max", "enoxaparin_min", "enoxaparin_median", "enoxaparin_hours_diff_mean", "enoxaparin_hours_diff_max", "enoxaparin_hours_diff_min"],
    }

    fitters = []

    for variant, drop_cols in enoxaparin_variants.items():
        enoxaparin_data = design_matrix.copy().drop(columns=drop_cols)
        fitter = CoxPHFitter()
        
        # Select the field type to discard
        if variant == "copy":
            drop_cols = enoxaparin_data.attrs["covariate_times_cols"]
        else:
            drop_cols = enoxaparin_data.attrs["with_covariate_cols"]
        
        fitter.fit(enoxaparin_data.drop(columns=drop_cols), **kwargs)
        fitters.append(fitter)

    return tuple(fitters)

format_dict = {'exp(coef)': '{:.4f}', 'exp(coef) lower 95%': '{:.4f}', 'exp(coef) upper 95%': '{:.4f}', 'p': '{:.4f}'}

def show_cox_and_anova(
    with_cov_fitter: CoxPHFitter,
    covariate_times_fitter: CoxPHFitter,
    df: pd.DataFrame,
) -> None:
    """
    1) CoxPH(binary) + Bool ANOVA
    2) CoxPH(times)  + Time  ANOVA
    """
    cox_cols = [
        "exp(coef)",
        "exp(coef) lower 95%",
        "exp(coef) upper 95%",
        "p",
    ]
    anova_all = make_type2_anova(df)

    # 第一張：Binary Covariates + Bool ANOVA
    bool_cols = [c for c in anova_all.columns if c.endswith("_bool")]
    
    merged_bool = (
        with_cov_fitter.summary.loc[:, cox_cols]
        .join(anova_all[bool_cols], how="left")
    )
    st.dataframe(
        merged_bool
          .style
          .pipe(format_small_values)
          .pipe(highlight_small_p)
          .pipe(highlight_last_col_small)
    )
    

    # 第二張：Covariate Times + Time ANOVA
    time_cols = [c for c in anova_all.columns if c.endswith("_time")]
    
    merged_time = (
        covariate_times_fitter.summary.loc[:, cox_cols]
        .join(anova_all[time_cols], how="left")
    )
    st.dataframe(
        merged_time
          .style
          .pipe(format_small_values)
          .pipe(highlight_small_p)
          .pipe(highlight_last_col_small)
    )
    st.markdown("---")

def show_all_aspirin_results(aspirin_df: pd.DataFrame) -> None:
    fitters  = make_cox_section_V2(aspirin_df)
    variants = [
        "binary","count","mean","max","min","median",
        "hours_diff_mean","hours_diff_max","hours_diff_min","hours_diff_median"
    ]

    # 共用變數列表
    with_cols     = ["with_psychosis","with_hypertension","with_heart_type_disease",
                     "with_neurological_type_disease","with_diabetes","with_hyperlipidemia"]
    times_cols    = ["hypertension_times","heart_type_disease_times","neurological_type_disease_times",
                     "diabetes_times","hyperlipidemia_times"]
    base_terms    = ["gender", "age"]

    cox_cols = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]

    for i, variant in enumerate(variants):
        cph = fitters[i]

        if variant == "binary":
            asp_col = "aspirin"
            # binary: gender, age, all with_*, aspirin
            anova_terms = base_terms + with_cols + [asp_col]
        else:
            asp_col = "aspirin_count" if variant=="count" else f"aspirin_{variant}"
            # time variants: gender, age, with_psychosis, all *_times, aspirin_xxx
            anova_terms = base_terms + ["with_psychosis"] + times_cols + [asp_col]

        # 1) CoxPHFitter 摘要: 取出所有 anova_terms 的行
        cox_summary = cph.summary.loc[anova_terms, cox_cols]

        # 2) Type II ANOVA
        dfc = aspirin_df.copy()
        dfc["E"]      = dfc["E"].astype(int)
        dfc["gender"] = dfc["gender"].astype(float)
        dfc["age"]    = dfc["age"].astype(float)

        formula = "E ~ " + " + ".join(anova_terms)
        lm = ols(formula, data=dfc.dropna(subset=anova_terms+["E"])).fit()
        anova_df = anova_lm(lm, typ=2).drop(columns=["df"])
        if "Residual" in anova_df.index:
            anova_df = anova_df.drop(index="Residual")

        # 3) 合併並顯示
        merged = pd.concat([cox_summary, anova_df], axis=1)
        #st.write(f"## Table {i+1}: Variant = {variant} ({asp_col})")
        st.dataframe(
            merged
            .style
            .pipe(format_small_values)
            .pipe(highlight_small_p)
            .pipe(highlight_last_col_small)
        )

def show_all_warfarin_results(warfarin_df: pd.DataFrame) -> None:
    fitters  = make_cox_section_V3(warfarin_df)
    variants = [
        "binary", "count", "mean", "max", "min", "median",
        "hours_diff_mean", "hours_diff_max", "hours_diff_min", "hours_diff_median"
    ]

    # 共用變數列表
    with_cols  = ["with_psychosis", "with_hypertension", "with_heart_type_disease",
                  "with_neurological_type_disease", "with_diabetes", "with_hyperlipidemia"]
    times_cols = ["hypertension_times", "heart_type_disease_times", "neurological_type_disease_times",
                  "diabetes_times", "hyperlipidemia_times"]
    base_terms = ["gender", "age"]

    cox_cols = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]

    for i, variant in enumerate(variants):
        cph = fitters[i]

        if variant == "binary":
            war_col = "warfarin"
            # binary: gender, age, all with_*, warfarin
            anova_terms = base_terms + with_cols + [war_col]
        else:
            war_col = "warfarin_count" if variant == "count" else f"warfarin_{variant}"
            # time variants: gender, age, with_psychosis, all *_times, warfarin_xxx
            anova_terms = base_terms + ["with_psychosis"] + times_cols + [war_col]

        # 1) CoxPHFitter 摘要: 取出所有 anova_terms 的行
        cox_summary = cph.summary.loc[anova_terms, cox_cols]

        # 2) Type II ANOVA
        dfc = warfarin_df.copy()
        dfc["E"]      = dfc["E"].astype(int)
        dfc["gender"] = dfc["gender"].astype(float)
        dfc["age"]    = dfc["age"].astype(float)

        formula = "E ~ " + " + ".join(anova_terms)
        lm = ols(formula, data=dfc.dropna(subset=anova_terms + ["E"])).fit()
        anova_df = anova_lm(lm, typ=2).drop(columns=["df"])
        if "Residual" in anova_df.index:
            anova_df = anova_df.drop(index="Residual")

        # 3) 合併並顯示
        merged = pd.concat([cox_summary, anova_df], axis=1)
        # st.write(f"## Table {i+1}: Variant = {variant} ({war_col})")
        st.dataframe(
            merged
            .style
            .pipe(format_small_values)
            .pipe(highlight_small_p)
            .pipe(highlight_last_col_small)
        )

def show_all_clopidogrel_results(clopidogrel_df: pd.DataFrame) -> None:
    fitters  = make_cox_section_V4(clopidogrel_df)
    variants = [
        "binary","count","mean","max","min","median",
        "hours_diff_mean","hours_diff_max","hours_diff_min","hours_diff_median"
    ]

    # 共用變數列表
    with_cols     = ["with_psychosis","with_hypertension","with_heart_type_disease",
                     "with_neurological_type_disease","with_diabetes","with_hyperlipidemia"]
    times_cols    = ["hypertension_times","heart_type_disease_times","neurological_type_disease_times",
                     "diabetes_times","hyperlipidemia_times"]
    base_terms    = ["gender", "age"]

    cox_cols = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]

    for i, variant in enumerate(variants):
        cph = fitters[i]

        if variant == "binary":
            drug_col = "clopidogrel"
            # binary: gender, age, all with_*, clopidogrel
            anova_terms = base_terms + with_cols + [drug_col]
        else:
            drug_col = "clopidogrel_count" if variant == "count" else f"clopidogrel_{variant}"
            # time variants: gender, age, with_psychosis, all *_times, clopidogrel_xxx
            anova_terms = base_terms + ["with_psychosis"] + times_cols + [drug_col]

        # 1) CoxPHFitter 摘要: 取出所有 anova_terms 的行
        cox_summary = cph.summary.loc[anova_terms, cox_cols]

        # 2) Type II ANOVA
        dfc = clopidogrel_df.copy()
        dfc["E"]      = dfc["E"].astype(int)
        dfc["gender"] = dfc["gender"].astype(float)
        dfc["age"]    = dfc["age"].astype(float)

        formula = "E ~ " + " + ".join(anova_terms)
        lm = ols(formula, data=dfc.dropna(subset=anova_terms + ["E"])).fit()
        anova_df = anova_lm(lm, typ=2).drop(columns=["df"])
        if "Residual" in anova_df.index:
            anova_df = anova_df.drop(index="Residual")

        # 3) 合併並顯示
        merged = pd.concat([cox_summary, anova_df], axis=1)
        st.dataframe(
            merged
            .style
            .pipe(format_small_values)
            .pipe(highlight_small_p)
            .pipe(highlight_last_col_small)
        )

def show_all_apixaban_results(apixaban_df: pd.DataFrame) -> None:
    fitters  = make_cox_section_V5(apixaban_df)
    variants = [
        "binary", "count", "mean", "max", "min", "median",
        "hours_diff_mean", "hours_diff_max", "hours_diff_min", "hours_diff_median"
    ]

    # 共用變數列表
    with_cols  = [
        "with_psychosis", "with_hypertension", "with_heart_type_disease",
        "with_neurological_type_disease", "with_diabetes", "with_hyperlipidemia"
    ]
    times_cols = [
        "hypertension_times", "heart_type_disease_times", "neurological_type_disease_times",
        "diabetes_times", "hyperlipidemia_times"
    ]
    base_terms = ["gender", "age"]

    cox_cols = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]

    for i, variant in enumerate(variants):
        cph = fitters[i]

        if variant == "binary":
            drug_col = "apixaban"
            # binary: gender, age, all with_*, apixaban
            anova_terms = base_terms + with_cols + [drug_col]
        else:
            drug_col = "apixaban_count" if variant == "count" else f"apixaban_{variant}"
            # time variants: gender, age, with_psychosis, all *_times, apixaban_xxx
            anova_terms = base_terms + ["with_psychosis"] + times_cols + [drug_col]

        # 1) CoxPHFitter 摘要: 取出所有 anova_terms 的行
        cox_summary = cph.summary.loc[anova_terms, cox_cols]

        # 2) Type II ANOVA
        dfc = apixaban_df.copy()
        dfc["E"]      = dfc["E"].astype(int)
        dfc["gender"] = dfc["gender"].astype(float)
        dfc["age"]    = dfc["age"].astype(float)

        formula = "E ~ " + " + ".join(anova_terms)
        lm = ols(formula, data=dfc.dropna(subset=anova_terms + ["E"])).fit()
        anova_df = anova_lm(lm, typ=2).drop(columns=["df"])
        if "Residual" in anova_df.index:
            anova_df = anova_df.drop(index="Residual")

        # 3) 合併並顯示
        merged = pd.concat([cox_summary, anova_df], axis=1)
        st.dataframe(
            merged
            .style
            .pipe(format_small_values)
            .pipe(highlight_small_p)
            .pipe(highlight_last_col_small)
        )

def show_all_rivaroxaban_results(rivaroxaban_df: pd.DataFrame) -> None:
    fitters  = make_cox_section_V6(rivaroxaban_df)
    variants = [
        "binary", "count", "mean", "max", "min", "median",
        "hours_diff_mean", "hours_diff_max", "hours_diff_min", "hours_diff_median"
    ]

    # 共用變數列表
    with_cols  = [
        "with_psychosis", "with_hypertension", "with_heart_type_disease",
        "with_neurological_type_disease", "with_diabetes", "with_hyperlipidemia"
    ]
    times_cols = [
        "hypertension_times", "heart_type_disease_times", "neurological_type_disease_times",
        "diabetes_times", "hyperlipidemia_times"
    ]
    base_terms = ["gender", "age"]

    cox_cols = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]

    for i, variant in enumerate(variants):
        cph = fitters[i]

        if variant == "binary":
            drug_col = "rivaroxaban"
            # binary: gender, age, all with_*, rivaroxaban
            anova_terms = base_terms + with_cols + [drug_col]
        else:
            drug_col = "rivaroxaban_count" if variant == "count" else f"rivaroxaban_{variant}"
            # time variants: gender, age, with_psychosis, all *_times, rivaroxaban_xxx
            anova_terms = base_terms + ["with_psychosis"] + times_cols + [drug_col]

        # 1) CoxPHFitter 摘要: 取出所有 anova_terms 的行
        cox_summary = cph.summary.loc[anova_terms, cox_cols]

        # 2) Type II ANOVA
        dfc = rivaroxaban_df.copy()
        dfc["E"]      = dfc["E"].astype(int)
        dfc["gender"] = dfc["gender"].astype(float)
        dfc["age"]    = dfc["age"].astype(float)

        formula = "E ~ " + " + ".join(anova_terms)
        lm = ols(formula, data=dfc.dropna(subset=anova_terms + ["E"])).fit()
        anova_df = anova_lm(lm, typ=2).drop(columns=["df"])
        if "Residual" in anova_df.index:
            anova_df = anova_df.drop(index="Residual")

        # 3) 合併並顯示
        merged = pd.concat([cox_summary, anova_df], axis=1)
        st.dataframe(
            merged
            .style
            .pipe(format_small_values)
            .pipe(highlight_small_p)
            .pipe(highlight_last_col_small)
        )


def show_all_dabigatran_etexilate_results(dabigatran_etexilate_df: pd.DataFrame) -> None:
    fitters  = make_cox_section_V7(dabigatran_etexilate_df)
    variants = [
        "binary", "count", "mean", "max", "min", "median",
        "hours_diff_mean", "hours_diff_max", "hours_diff_min", "hours_diff_median"
    ]

    with_cols  = [
        "with_psychosis", "with_hypertension", "with_heart_type_disease",
        "with_neurological_type_disease", "with_diabetes", "with_hyperlipidemia"
    ]
    times_cols = [
        "hypertension_times", "heart_type_disease_times", "neurological_type_disease_times",
        "diabetes_times", "hyperlipidemia_times"
    ]
    base_terms = ["gender", "age"]
    cox_cols   = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]

    for i, variant in enumerate(variants):
        cph = fitters[i]

        # —— 1) 準備「原始」與「安全」欄位名稱 —— #
        if variant == "binary":
            orig_drug_col = "dabigatran etexilate"
        else:
            suffix        = "count" if variant == "count" else variant
            orig_drug_col = f"dabigatran etexilate_{suffix}"

        safe_drug_col = orig_drug_col.replace(" ", "_")

        # Cox 要抓的 index
        anova_terms_cox = (
            base_terms +
            (with_cols if variant=="binary" else ["with_psychosis"] + times_cols) +
            [orig_drug_col]
        )

        # —— 2) 取 CoxPH 結果 —— #
        cox_summary = cph.summary.loc[anova_terms_cox, cox_cols]

        # —— 3) 準備 ANOVA DataFrame —— #
        dfc = dabigatran_etexilate_df.copy()
        # 把 DataFrame 裡的「原始」欄位改名成「安全」欄位
        dfc = dfc.rename(columns={orig_drug_col: safe_drug_col})
        dfc["E"]      = dfc["E"].astype(int)
        dfc["gender"] = dfc["gender"].astype(float)
        dfc["age"]    = dfc["age"].astype(float)

        anova_terms_ols = (
            base_terms +
            (with_cols if variant=="binary" else ["with_psychosis"] + times_cols) +
            [safe_drug_col]
        )
        formula = "E ~ " + " + ".join(anova_terms_ols)
        lm      = ols(formula, data=dfc.dropna(subset=anova_terms_ols + ["E"])).fit()
        anova_df = anova_lm(lm, typ=2).drop(columns=["df"])

        # 刪掉 Residual
        if "Residual" in anova_df.index:
            anova_df = anova_df.drop(index="Residual")

        # —— 4) 把 ANOVA 的 index rename 回「原始」欄位名稱 —— #
        anova_df = anova_df.rename(index={safe_drug_col: orig_drug_col})

        # —— 5) 合併並顯示 —— #
        merged = pd.concat([cox_summary, anova_df], axis=1)
        st.dataframe(
            merged
            .style
            .pipe(format_small_values)
            .pipe(highlight_small_p)
            .pipe(highlight_last_col_small)
        )


def show_all_cilostazol_results(cilostazol_df: pd.DataFrame) -> None:
    fitters  = make_cox_section_V8(cilostazol_df)
    variants = [
        "binary", "count", "mean", "max", "min", "median",
        "hours_diff_mean", "hours_diff_max", "hours_diff_min", "hours_diff_median"
    ]

    # 共用變數列表
    with_cols  = [
        "with_psychosis", "with_hypertension", "with_heart_type_disease",
        "with_neurological_type_disease", "with_diabetes", "with_hyperlipidemia"
    ]
    times_cols = [
        "hypertension_times", "heart_type_disease_times", "neurological_type_disease_times",
        "diabetes_times", "hyperlipidemia_times"
    ]
    base_terms = ["gender", "age"]

    cox_cols = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]

    for i, variant in enumerate(variants):
        cph = fitters[i]

        if variant == "binary":
            drug_col = "cilostazol"
            # binary: gender, age, all with_*, cilostazol
            anova_terms = base_terms + with_cols + [drug_col]
        else:
            drug_col = "cilostazol_count" if variant == "count" else f"cilostazol_{variant}"
            # time variants: gender, age, with_psychosis, all *_times, cilostazol_xxx
            anova_terms = base_terms + ["with_psychosis"] + times_cols + [drug_col]

        # 1) CoxPHFitter 摘要: 取出所有 anova_terms 的行
        cox_summary = cph.summary.loc[anova_terms, cox_cols]

        # 2) Type II ANOVA
        dfc = cilostazol_df.copy()
        dfc["E"]      = dfc["E"].astype(int)
        dfc["gender"] = dfc["gender"].astype(float)
        dfc["age"]    = dfc["age"].astype(float)

        formula = "E ~ " + " + ".join(anova_terms)
        lm = ols(formula, data=dfc.dropna(subset=anova_terms + ["E"])).fit()
        anova_df = anova_lm(lm, typ=2).drop(columns=["df"])
        if "Residual" in anova_df.index:
            anova_df = anova_df.drop(index="Residual")

        # 3) 合併並顯示
        merged = pd.concat([cox_summary, anova_df], axis=1)
        st.dataframe(
            merged
            .style
            .pipe(format_small_values)
            .pipe(highlight_small_p)
            .pipe(highlight_last_col_small)
        )

def show_all_enoxaparin_results(enoxaparin_df: pd.DataFrame) -> None:
    fitters  = make_cox_section_V9(enoxaparin_df)
    variants = [
        "binary", "count", "mean", "max", "min", "median",
        "hours_diff_mean", "hours_diff_max", "hours_diff_min", "hours_diff_median"
    ]

    # 共用變數列表
    with_cols  = [
        "with_psychosis", "with_hypertension", "with_heart_type_disease",
        "with_neurological_type_disease", "with_diabetes", "with_hyperlipidemia"
    ]
    times_cols = [
        "hypertension_times", "heart_type_disease_times", "neurological_type_disease_times",
        "diabetes_times", "hyperlipidemia_times"
    ]
    base_terms = ["gender", "age"]

    cox_cols = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]

    for i, variant in enumerate(variants):
        cph = fitters[i]

        if variant == "binary":
            drug_col = "enoxaparin"
            # binary: gender, age, all with_*, enoxaparin
            anova_terms = base_terms + with_cols + [drug_col]
        else:
            drug_col = "enoxaparin_count" if variant == "count" else f"enoxaparin_{variant}"
            # time variants: gender, age, with_psychosis, all *_times, enoxaparin_xxx
            anova_terms = base_terms + ["with_psychosis"] + times_cols + [drug_col]

        # 1) CoxPHFitter 摘要: 取出所有 anova_terms 的行
        cox_summary = cph.summary.loc[anova_terms, cox_cols]

        # 2) Type II ANOVA
        dfc = enoxaparin_df.copy()
        dfc["E"]      = dfc["E"].astype(int)
        dfc["gender"] = dfc["gender"].astype(float)
        dfc["age"]    = dfc["age"].astype(float)

        formula = "E ~ " + " + ".join(anova_terms)
        lm = ols(formula, data=dfc.dropna(subset=anova_terms + ["E"])).fit()
        anova_df = anova_lm(lm, typ=2).drop(columns=["df"])
        if "Residual" in anova_df.index:
            anova_df = anova_df.drop(index="Residual")

        # 3) 合併並顯示
        merged = pd.concat([cox_summary, anova_df], axis=1)
        st.dataframe(
            merged
            .style
            .pipe(format_small_values)
            .pipe(highlight_small_p)
            .pipe(highlight_last_col_small)
        )


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
        #read_design_matrix().pipe(filter_age).pipe(crop_event)
    )
    
    data_copy = design_matrix.copy()
    # Cut to columns of specified range
    design_matrix_df = data_copy.loc[:, 'gender':'hyperlipidemia_times']

    aspirin_df = data_copy.loc[:, 'gender':'aspirin_hours_diff_median']
    aspirin_df.drop(columns=['aspirin_hours_diff_count'], inplace=True)

    warfarin_df = data_copy.loc[:, 'warfarin':'warfarin_hours_diff_median']
    warfarin_df.drop(columns=['warfarin_hours_diff_count'], inplace=True)

    clopidogrel_df = data_copy.loc[:, 'clopidogrel':'clopidogrel_hours_diff_median']
    clopidogrel_df.drop(columns=['clopidogrel_hours_diff_count'], inplace=True)

    apixaban_df = data_copy.loc[:, 'apixaban':'apixaban_hours_diff_median']
    apixaban_df.drop(columns=['apixaban_hours_diff_count'], inplace=True)

    rivaroxaban_df = data_copy.loc[:, 'rivaroxaban':'rivaroxaban_hours_diff_median']
    rivaroxaban_df.drop(columns=['rivaroxaban_hours_diff_count'], inplace=True)

    dabigatran_etexilate_df = data_copy.loc[:, 'dabigatran etexilate':'dabigatran etexilate_hours_diff_median']
    dabigatran_etexilate_df.drop(columns=['dabigatran etexilate_hours_diff_count'], inplace=True)

    cilostazol_df = data_copy.loc[:, 'cilostazol':'cilostazol_hours_diff_median']
    cilostazol_df.drop(columns=['cilostazol_hours_diff_count'], inplace=True)

    enoxaparin_df = data_copy.loc[:, 'enoxaparin':'enoxaparin_hours_diff_median']
    enoxaparin_df.drop(columns=['enoxaparin_hours_diff_count'], inplace=True)

    # Concatenate the results of two column ranges
    warfarin_df = pd.concat([design_matrix_df, warfarin_df], axis=1)
    clopidogrel_df = pd.concat([design_matrix_df, clopidogrel_df], axis=1)
    apixaban_df = pd.concat([design_matrix_df, apixaban_df], axis=1)
    rivaroxaban_df = pd.concat([design_matrix_df, rivaroxaban_df], axis=1)
    dabigatran_etexilate_df = pd.concat([design_matrix_df, dabigatran_etexilate_df], axis=1)
    cilostazol_df = pd.concat([design_matrix_df, cilostazol_df], axis=1)
    enoxaparin_df = pd.concat([design_matrix_df, enoxaparin_df], axis=1)
    
    # Manually set a hidden anchor point
    st.markdown('<div id="section-1"></div>', unsafe_allow_html=True)

    st.markdown("# Proportional Hazard Model & ANOVA", unsafe_allow_html=False)

    st.header("Disease")
    with_cov_fitter, cov_times_fitter = make_cox_section(design_matrix_df)
    show_cox_and_anova(with_cov_fitter, cov_times_fitter, design_matrix_df)
    #show_cox_section(*make_cox_section(design_matrix_df))

    # Use standard Streamlit headers
    st.header("Drug")

    # Create tab page
    tabs = st.tabs([
        "1. Aspirin",
        "2. Warfarin",
        "3. Clopidogrel",
        "4. Apixaban",
        "5. Rivaroxaban",
        "6. Dabigatran etexilate",
        "7. Cilostazol",
        "8. Enoxaparin"
    ])
    
    # Aspirin
    with tabs[0]:
        st.header("Aspirin")
        #show_cox_section(*make_cox_section_V2(aspirin_df))
        show_all_aspirin_results(aspirin_df)
    # Warfarin
    with tabs[1]:
        st.header("Warfarin")
        #show_cox_section(*make_cox_section_V3(warfarin_df))
        show_all_warfarin_results(warfarin_df)
    # Clopidogrel
    with tabs[2]:
        st.header("Clopidogrel")
        #show_cox_section(*make_cox_section_V4(clopidogrel_df))
        show_all_clopidogrel_results(clopidogrel_df)
    # Apixaban
    with tabs[3]:
        st.header("Apixaban")
        #show_cox_section(*make_cox_section_V5(apixaban_df))
        show_all_apixaban_results(apixaban_df)
    # Rivaroxaban
    with tabs[4]:
        st.header("Rivaroxaban")
        #show_cox_section(*make_cox_section_V6(rivaroxaban_df))
        show_all_rivaroxaban_results(rivaroxaban_df)
    # Dabigatran etexilate
    with tabs[5]:
        st.header("Dabigatran etexilate")
        #show_cox_section(*make_cox_section_V7(dabigatran_etexilate_df))
        show_all_dabigatran_etexilate_results(dabigatran_etexilate_df)
    # Cilostazol
    with tabs[6]:
        st.header("Cilostazol")
        #show_cox_section(*make_cox_section_V8(cilostazol_df))
        show_all_cilostazol_results(cilostazol_df)
    # Enoxaparin
    with tabs[7]:
        st.header("Enoxaparin")
        #show_cox_section(*make_cox_section_V9(enoxaparin_df))
        show_all_enoxaparin_results(enoxaparin_df)
    show_data_view_section(*make_data_view_section(design_matrix))

    # anchor button
    # Add a "Back to Top" button fixed in the lower right corner
    st.markdown("""
    <style>
    #go-top-button {
        position: fixed;
        bottom: 60px; /*Adjust this to change the distance of the button from the bottom*/
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