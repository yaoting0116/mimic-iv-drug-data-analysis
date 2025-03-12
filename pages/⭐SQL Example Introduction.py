import streamlit as st

redirect_url = "https://mimic-iv-disease-medication-sql-ting-uwu.streamlit.app/"

st.markdown(
    f"""
    <script type="text/javascript">
        window.location.href = "{redirect_url}";
    </script>
    """,
    unsafe_allow_html=True
)
