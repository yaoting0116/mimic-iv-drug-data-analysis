import streamlit as st

redirect_url = "https://mimic-iv-disease-medication-sql-ting-uwu.streamlit.app/"

st.markdown(
    f"""
    <iframe src="{redirect_url}" width="100%" height="800px"></iframe>
    """,
    unsafe_allow_html=True
)
