from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import streamlit as st
import warnings
from pandas.io.formats.style import Styler
from itertools import groupby
from enum import Enum
from more_itertools import chunked
from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    ks_2samp,
    mannwhitneyu,
    ttest_ind,
)


def highlight_small_p(styler: Styler) -> Styler:
    return styler.highlight_between("p", left=0, right=0.0001) and styler.highlight_between("p", left=0.9900, right=1)

def to_percentage(df: pd.DataFrame) -> Styler:
    return df.apply(lambda col: col / col.sum()).mul(100).round(1).style.format("{}%")

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
        subset = pd.IndexSlice[styler.data[col].between(0.9900, 1, inclusive="right"), col]
        styler = styler.format("> 0.9900", subset)
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

def read_design_matrix() -> pd.DataFrame:
    topic_to_pickle_fname = {
        "Psychosis & Ischemic Stroke": "Ischemic_Stroke_join_drug_V5",
        "Psychosis & Ischemic Stroke Control Diagnosis Top 3 Filter": "Ischemic_Stroke_join_drug_V4_T_1day",
        "Psychosis & Hemorrhagic Stroke": "Hemorrhagic_Stroke_join_drug_V5",
        "Psychosis & Hemorrhagic Stroke Control Diagnosis Top 3 Filter": "Hemorrhagic_Stroke_join_drug_V4_T_1day",
    }
    if "topic_index" not in st.session_state:
        st.session_state["topic_index"] = 0

    def on_change() -> None:
        topic_selected = st.session_state["topic-radio"]
        st.session_state["topic_index"] = list(topic_to_pickle_fname).index(
            topic_selected
        )

    topic_selected: str = st.sidebar.radio(
        "topic",
        options=topic_to_pickle_fname,
        index=st.session_state["topic_index"],
        key="topic-radio",  # to identify widgets on session state
        on_change=on_change,
    )
    design_matrix = pd.read_pickle(f"data/{topic_to_pickle_fname[topic_selected]}.pkl",compression='gzip') # Your own folder path

    # for slider usage
    design_matrix.attrs["max_duration"] = int(design_matrix["T"].max())

    return design_matrix


# def filter_gender(design_matrix: pd.DataFrame) -> pd.DataFrame:
#     if "gender_index" not in st.session_state:
#         st.session_state["gender_index"] = 0

#     def on_change() -> None:
#         gender_selected = st.session_state["gender-radio"]
#         st.session_state["gender_index"] = Gender.to_list().index(gender_selected)

#     gender_selected: Gender = st.sidebar.radio(
#         "gender",
#         options=Gender.to_list(),
#         index=st.session_state["gender_index"],
#         key="gender-radio",  # to identify widgets on session state
#         on_change=on_change,
#     )
#     design_matrix.attrs["gender_selected"] = gender_selected
#     match gender_selected:
#         case Gender.ALL:
#             return design_matrix
#         case Gender.MALE:
#             return design_matrix.query("gender == 1")
#         case Gender.FEMALE:
#             return design_matrix.query("gender == 0")
#         case _:
#             raise AttributeError

def filter_age(design_matrix: pd.DataFrame, min_gap: int = 25) -> pd.DataFrame:
    if "age_threshold" not in st.session_state:
        st.session_state["age_threshold"] = (18, 110)

    def on_change() -> None:
        lower, upper = st.session_state["age-slider"]
        # Make sure the slider has at least min_gap gap
        if upper - lower < min_gap:
            st.error(f"The slider range must be at least {min_gap} years.")
            # Restore to previous legal scope
            st.session_state["age-slider"] = st.session_state["age_threshold"]
        else:
            st.session_state["age_threshold"] = st.session_state["age-slider"]

    age_threshold = st.sidebar.slider(
        "age",
        min_value=18,
        max_value=110,
        value=st.session_state["age_threshold"],
        key="age-slider",
        on_change=on_change,
    )
    
    # Filter data within a set range
    filtered_data = design_matrix.query(f"{age_threshold[0]} <= age <= {age_threshold[1]}")
    
    return filtered_data

def crop_event(design_matrix: pd.DataFrame, min_gap: int = 90) -> pd.DataFrame:
    if "duration_threshold" not in st.session_state:
        # Initialize the valid range as (1, max_duration)
        st.session_state["duration_threshold"] = (1, design_matrix.attrs["max_duration"])

    def on_change() -> None:
        lower, upper = 1, st.session_state["duration-slider"]
        max_duration = design_matrix.attrs["max_duration"]
        # Ensure the range is valid and has at least min_gap
        if upper - lower < min_gap or not (1 <= upper <= max_duration):
            st.error(f"The upper limit must be at least {min_gap} and within the range 1 to 4684.")
            # Revert to the previous valid range
            st.session_state["duration-slider"] = st.session_state["duration_threshold"][1]
        else:
            # Update the valid range
            st.session_state["duration_threshold"] = (1, upper)

    max_duration = design_matrix.attrs["max_duration"]
    upper = st.sidebar.slider(
        "duration",
        min_value=1,
        max_value=max_duration,
        value=st.session_state["duration_threshold"][1],
        step=90,
        key="duration-slider",
        on_change=on_change,
    )

    lower, upper = 1, upper  # Lower limit is fixed at 1, only the upper limit is adjustable by the user
    # Set the event column name
    if lower == 1 and upper == max_duration:
        event_col = "E"
    elif lower == 1 and upper > 1:
        event_col = f"E{upper}"
    else:
        event_col = f"E{lower}-{upper}"
    design_matrix.attrs["event_col"] = event_col

    # Process the event column
    if event_col != "E":
        design_matrix[event_col] = False
        design_matrix.loc[
            (design_matrix["E"] == True)
            & (lower <= design_matrix["T"])
            & (design_matrix["T"] <= upper),
            event_col,
        ] = True
        # Find the index of column "E" and insert the new column after it
        e_col_index = design_matrix.columns.get_loc("E") + 1
        design_matrix.insert(e_col_index, event_col, design_matrix.pop(event_col))
    return design_matrix





@dataclass
class StatSubSection:
    """stat_section is composed of multiple StatSubSections."""

    subheader: str
    crosstab: pd.DataFrame
    stat_result: pd.DataFrame

def make_catgorical_stat_results_3(filtered_values_T,filtered_values_F):
    
    t_res = ttest_ind(filtered_values_T, filtered_values_F)
    ks_res = ks_2samp(filtered_values_T, filtered_values_F)
    u_res = mannwhitneyu(filtered_values_T, filtered_values_F)
    
    return pd.DataFrame(
        [
            [t_res.statistic, t_res.pvalue],
            [u_res.statistic, u_res.pvalue],
            [ks_res.statistic, ks_res.pvalue],
        ],
        index=["t test", "U test", "KS test"],
        columns=["stat", "p"],
    )

def t_u_ks_statistics_3(drug_names,drug_statistics):
    
    # Do statistics on column_C when column_P is True and column_A is True
    condition_true_true = (design_matrix['with_psychosis'] & design_matrix[drug_names])
    # Do statistics on column_C when column_P is False and column_A is True
    condition_false_true = (~design_matrix['with_psychosis'] & design_matrix[drug_names])

    # Filter DataFrame based on conditions
    filtered_true_true = design_matrix[condition_true_true]
    filtered_false_true = design_matrix[condition_false_true]

    control= filtered_false_true[f'{drug_names}_{drug_statistics}'].tolist()
    case= filtered_true_true[f'{drug_names}_{drug_statistics}'].tolist()

    return case,control

def make_stat_section(
    design_matrix: pd.DataFrame,
) -> list[StatSubSection]:
    subsections: list[StatSubSection] = []
    
    design_matrix.attrs["predictor_col"] = "with_psychosis"
    predictor_col = design_matrix.attrs["predictor_col"]
    case = design_matrix.query(predictor_col)
    control = design_matrix.query(f"~{predictor_col}")

    # age
    crosstab = (
        design_matrix.groupby(predictor_col)["age"]
        .agg(("mean", "std"))
        .transpose()
        .rename(columns={True: "case", False: "control"})
    )

    t_res = ttest_ind(case["age"], control["age"])
    ks_res = ks_2samp(case["age"], control["age"])
    u_res = mannwhitneyu(case["age"], control["age"])
    stat_results = pd.DataFrame(
        [
            [t_res.statistic, t_res.pvalue],
            [u_res.statistic, u_res.pvalue],
            [ks_res.statistic, ks_res.pvalue],
        ],
        index=["t test", "U test", "KS test"],
        columns=["stat", "p"],
    )
    subsection = StatSubSection("Age", crosstab, stat_results)
    subsections.append(subsection)

    def make_catgorical_stat_results(crosstab: pd.DataFrame) -> pd.DataFrame:
        chi2_res = chi2_contingency(crosstab, correction=False)
        fe_res = fisher_exact(crosstab)
        return pd.DataFrame(
            [
                [chi2_res[0], chi2_res[1]],
                [fe_res[0], fe_res[1]],
            ],
            index=["chi2 test", "Fisher exact test"],
            columns=["stat", "p"],
        )

    # gender
    # if design_matrix.attrs["gender_selected"] == Gender.ALL:
    #     crosstab = pd.crosstab(
    #         design_matrix["gender"], design_matrix[predictor_col]
    #     ).rename(
    #         index={1: "male", 0: "female"}, columns={True: "case", False: "control"}
    #     )
    #     stat_result = make_catgorical_stat_results(crosstab)
    #     subsection = StatSubSection("Gender", crosstab, stat_result)
    #     subsections.append(subsection)
    crosstab = pd.crosstab(
        design_matrix["gender"], design_matrix[predictor_col]
    ).rename(
        index={1: "male", 0: "female"}, columns={True: "case", False: "control"}
    )
    stat_result = make_catgorical_stat_results(crosstab)
    subsection = StatSubSection("Gender", crosstab, stat_result)
    subsections.append(subsection)

    # event
    event_col = design_matrix.attrs["event_col"]
    crosstab = pd.crosstab(
        design_matrix[event_col], design_matrix[predictor_col]
    ).rename(
        index={True: "true", False: "false"},
        columns={True: "case", False: "control"},
    )
    stat_result = make_catgorical_stat_results(crosstab)
    subsection = StatSubSection("Event", crosstab, stat_result)
    subsections.append(subsection)

    # covariate
    covariates = [
    "hypertension",
    "heart_type_disease",
    "neurological_type_disease",
    "diabetes",
    "hyperlipidemia",
    ]

    drug_cols = [
    "aspirin",
    "warfarin",
    "clopidogrel",
    "apixaban",
    "rivaroxaban",
    "dabigatran etexilate",
    "cilostazol",
    "enoxaparin",
    ]
    drug_statistics_cols = []
    drug_statistics = ['median','count','max','mean','min',
                       'hours_diff_mean','hours_diff_max','hours_diff_min','hours_diff_median']

    design_matrix.attrs["with_covariate_cols"] = [f"with_{c}" for c in covariates]
    design_matrix.attrs["covariate_times_cols"] = [f"{c}_times" for c in covariates]
    cols = design_matrix.attrs["with_covariate_cols"]

    design_matrix.attrs["drug_name_cols"] = drug_cols
    drug_cols = design_matrix.attrs["drug_name_cols"]

    for drug in drug_cols:
        for stat in drug_statistics:
            drug_statistics_cols.append(f"{drug}_{stat}")
    
    design_matrix.attrs["drug_statistics_cols"] = drug_statistics_cols

    for col in cols:
        subheader = col.replace("_", " ").title()
        crosstab = pd.crosstab(design_matrix[col], design_matrix[predictor_col]).rename(
            index={True: "with", False: "without"},
            columns={True: "case", False: "control"},
        )
        stat_result = make_catgorical_stat_results(crosstab)
        subsection = StatSubSection(subheader, crosstab, stat_result)
        subsections.append(subsection)
    
    for cs in drug_cols:
        subheader = cs.title()
        
        # Calculate the basic crosstab for this drug
        crosstab = pd.crosstab(design_matrix[cs], design_matrix[predictor_col]).rename(
            index={True: "true", False: "false"},
            columns={True: "case", False: "control"},
        )
        stat_result = make_catgorical_stat_results(crosstab)
        subsection = StatSubSection(subheader, crosstab, stat_result)
        subsections.append(subsection)

        # Filter data containing this drug
        filtered_design_matrix = design_matrix[design_matrix[cs]]

        # Process all statistical results for this drug
        for stat in drug_statistics:
            drug_stat_col = f"{cs}_{stat}"
            
            if drug_stat_col in drug_statistics_cols:
                subheader = drug_stat_col.replace("_", " ").title()

                # Calculate the grouped data for this statistic
                crosstab = (
                    filtered_design_matrix.groupby(predictor_col)[drug_stat_col]
                    .agg(("mean", "std"))
                    .transpose()
                    .rename(
                        index={"mean": "Mean", "std": "Std"},
                        columns={True: "case", False: "control"}
                    )
                )

                # Calculate statistical results (e.g., t-test, U-test)
                case, control = t_u_ks_statistics_3(cs, stat)
                stat_result = make_catgorical_stat_results_3(case, control)

                # Add to report
                subsection = StatSubSection(subheader, crosstab, stat_result)
                subsections.append(subsection)

    return subsections

def show_stat_section(subsections: list[StatSubSection]) -> None:
    format_dict = {'stat': '{:.4f}', 'p': '{:.4f}'}
    drug_keywords = {"Aspirin", "Warfarin", "Clopidogrel", "Apixaban", "Rivaroxaban",
                     "Dabigatran Etexilate", "Cilostazol", "Enoxaparin"}
    exclude_headers = ["Gender", "Event", "With Hypertension", "With Heart Type Disease",
                       "With Neurological Type Disease", "With Diabetes", "With Hyperlipidemia",
                       "Aspirin", "Warfarin", "Clopidogrel", "Apixaban", "Rivaroxaban",
                       "Dabigatran Etexilate", "Cilostazol", "Enoxaparin"]
    st.markdown("")
    st.header("Independence Tests")

    # Determine whether it is a drug-related title
    def is_drug_related(subheader: str) -> bool:
        return any(keyword in subheader for keyword in drug_keywords)

    # Grouping drugs and non-drugs
    grouped = groupby(subsections, lambda s: is_drug_related(s.subheader))

    for is_drug, group in grouped:
        group = list(group)  # Convert group to list for multiple iterations
        if is_drug:
            st.markdown("---")
            st.header("Drug Independence Tests")

        # Arrange table in pairs
        for two_subsections in chunked(group, 2):
            cols = st.columns(2)  # Fixed two columns
            for stcol, subsection in zip(cols, two_subsections):
                stcol.subheader(subsection.subheader)
                
                # Determine whether a percentage needs to be calculated
                if subsection.subheader in exclude_headers:
                    crosstab_with_percentage = subsection.crosstab.copy()
                    total = crosstab_with_percentage.sum(axis=0, skipna=True)
                    crosstab_with_percentage = crosstab_with_percentage.applymap(
                        lambda x: f"{int(x):,}"  # Format non-floating point numbers as three-digit commas
                    )
                    percentage = (subsection.crosstab.div(total, axis=1) * 100).round(1).astype(str) + '%'
                    crosstab_with_percentage = crosstab_with_percentage + ' (' + percentage + ')'
                    stcol.table(crosstab_with_percentage)
                else:
                    # Format original crosstab
                    formatted_crosstab = subsection.crosstab.applymap(
                        lambda x: f"{x:,.4f}" if isinstance(x, float) else f"{int(x):,}"
                    )
                    stcol.table(formatted_crosstab)
                
                # Show statistical results table
                stcol.table(
                    subsection.stat_result.style.format(format_dict).pipe(format_small_values).pipe(
                        highlight_small_p
                    )
                )
            # If there is one table left, put it on the left side of a column alone.
            if len(two_subsections) < 2:
                cols[1].empty()  # Leave the right column blank
    st.markdown("---")

if __name__ == "__main__":
    # Manually set a hidden anchor point
    st.markdown('<div id="section-1"></div>', unsafe_allow_html=True)

    # Ignore RuntimeWarning. During a comprehensive inspection, this line must be removed first and then checked.
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    design_matrix = (
        #read_design_matrix().pipe(filter_gender).pipe(filter_age).pipe(crop_event)
        read_design_matrix().pipe(filter_age).pipe(crop_event)
    )
    show_stat_section(make_stat_section(design_matrix))

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
