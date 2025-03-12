import streamlit as st

# Specify the URL you want to redirect to
redirect_url = "https://mimic-iv-disease-medication-sql-ting-uwu.streamlit.app/"

# Use an HTML meta tag to perform the redirect
st.markdown(
    f"""
    <script>
      window.location.href = "{redirect_url}";
    </script>
    """,
    unsafe_allow_html=True
)

st.write("Redirecting to the new website...")
