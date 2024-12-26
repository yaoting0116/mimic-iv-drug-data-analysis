from __future__ import annotations
from importlib import import_module
from typing import Callable
import pandas as pd
import streamlit as st
from lifelines import CoxPHFitter
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

def make_cox_section_V2(design_matrix: pd.DataFrame) -> tuple[CoxPHFitter, ...]:
    # Initial processing
    design_matrix = design_matrix.select_dtypes(exclude="datetime")
    if design_matrix.attrs["event_col"] != "E":
        design_matrix.drop(columns="E", inplace=True)
        
    # CoxPHFitter fails to converge when the values in gender column are all the same.
    if design_matrix.attrs["gender_selected"] != Gender.ALL:
        design_matrix = design_matrix.drop("gender", axis=1)
    
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

    # CoxPHFitter fails to converge when the values in gender column are all the same.
    if design_matrix.attrs["gender_selected"] != Gender.ALL:
        design_matrix = design_matrix.drop("gender", axis=1)
    
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

    # CoxPHFitter fails to converge when the values in gender column are all the same.
    if design_matrix.attrs["gender_selected"] != Gender.ALL:
        design_matrix = design_matrix.drop("gender", axis=1)
    
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

    # CoxPHFitter fails to converge when the values in gender column are all the same.
    if design_matrix.attrs["gender_selected"] != Gender.ALL:
        design_matrix = design_matrix.drop("gender", axis=1)
    
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

    # CoxPHFitter fails to converge when the values in gender column are all the same.
    if design_matrix.attrs["gender_selected"] != Gender.ALL:
        design_matrix = design_matrix.drop("gender", axis=1)
    
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

    # CoxPHFitter fails to converge when the values in gender column are all the same.
    if design_matrix.attrs["gender_selected"] != Gender.ALL:
        design_matrix = design_matrix.drop("gender", axis=1)
    
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

    # CoxPHFitter fails to converge when the values in gender column are all the same.
    if design_matrix.attrs["gender_selected"] != Gender.ALL:
        design_matrix = design_matrix.drop("gender", axis=1)
    
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

    # CoxPHFitter fails to converge when the values in gender column are all the same.
    if design_matrix.attrs["gender_selected"] != Gender.ALL:
        design_matrix = design_matrix.drop("gender", axis=1)
    
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
    # Use standard Streamlit headers
    st.markdown("# Drug Proportional Hazard Model", unsafe_allow_html=False)

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
        show_cox_section(*make_cox_section_V2(aspirin_df))

    # Warfarin
    with tabs[1]:
        st.header("Warfarin")
        show_cox_section(*make_cox_section_V3(warfarin_df))

    # Clopidogrel
    with tabs[2]:
        st.header("Clopidogrel")
        show_cox_section(*make_cox_section_V4(clopidogrel_df))

    # Apixaban
    with tabs[3]:
        st.header("Apixaban")
        show_cox_section(*make_cox_section_V5(apixaban_df))

    # Rivaroxaban
    with tabs[4]:
        st.header("Rivaroxaban")
        show_cox_section(*make_cox_section_V6(rivaroxaban_df))

    # Dabigatran etexilate
    with tabs[5]:
        st.header("Dabigatran etexilate")
        show_cox_section(*make_cox_section_V7(dabigatran_etexilate_df))

    # Cilostazol
    with tabs[6]:
        st.header("Cilostazol")
        show_cox_section(*make_cox_section_V8(cilostazol_df))
    
    # Enoxaparin
    with tabs[7]:
        st.header("Enoxaparin")
        show_cox_section(*make_cox_section_V9(enoxaparin_df))

    show_data_view_section(*make_data_view_section(design_matrix))

    # anchor button
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
