import streamlit as st
import pandas as pd
import os
import plotly.express as px
from ydata_profiling import ProfileReport
from datetime import datetime

# -------------------- Streamlit UI Setup --------------------
st.set_page_config(page_title="AIBE 3.14 EDA", layout="wide", page_icon="📊")

st.markdown("""
    <div style='text-align:center'>
        <img src='https://img.icons8.com/color/96/000000/artificial-intelligence.png'/>
        <h1 style='color:#4E8EE0;'>AIBE 3.14 – Automated EDA Dashboard</h1>
        <p>Upload a dataset, explore it, and generate full EDA reports 🎯</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------- Upload CSV File --------------------
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()

    filename = uploaded_file.name.rsplit('.', 1)[0]
    st.success(f"✅ Successfully loaded `{uploaded_file.name}`")

    # -------------------- Show Preview --------------------
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # -------------------- Column Info --------------------
    st.subheader("📦 Column Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔢 Numerical Columns**")
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        st.write(num_cols)
    with col2:
        st.markdown("**🔠 Categorical Columns**")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        st.write(cat_cols)

    # -------------------- Quick Charts --------------------
    st.subheader("📊 Visual Insights")

    if cat_cols:
        cat_col = st.selectbox("Choose a categorical column", cat_cols, key="cat")
        fig1 = px.bar(df[cat_col].value_counts().head(20).reset_index(),
                      x='index', y=cat_col,
                      labels={'index': cat_col, cat_col: 'Count'},
                      title=f"Top 20 Most Frequent Values in '{cat_col}'")
        st.plotly_chart(fig1, use_container_width=True)

    if num_cols:
        num_col = st.selectbox("Choose a numerical column", num_cols, key="num")
        fig2 = px.histogram(df, x=num_col, nbins=30, title=f"Distribution of '{num_col}'")
        st.plotly_chart(fig2, use_container_width=True)

    # -------------------- Generate Profiling Report --------------------
    st.subheader("🧠 Full EDA Report")

    if st.button("💾 Generate & Save HTML Report"):
        profile = ProfileReport(df, title=f"{filename} - EDA Report", explorative=True)

        output_dir = f"reports/{filename}"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"{filename}_report_{timestamp}.html")
        profile.to_file(report_path)

        st.success(f"✅ Report saved to `{report_path}`")

        # Display the HTML report in-app
        with open(report_path, 'r', encoding='utf-8') as f:
            html_report = f.read()
            st.components.v1.html(html_report, height=800, scrolling=True)


