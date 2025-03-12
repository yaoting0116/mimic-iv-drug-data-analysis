import streamlit as st

redirect_url = "https://mimic-iv-disease-medication-sql-ting-uwu.streamlit.app/"

st.markdown(
    f"""
    <script type="text/javascript">
        setTimeout(function() {{
            window.location.href = "{redirect_url}";
        }}, 10);
    </script>
    """,
    unsafe_allow_html=True
)

# Stop execution to prevent further Streamlit processing
st.stop()
