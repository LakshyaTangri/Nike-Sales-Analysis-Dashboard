import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Nike Sales Dashboard",
    layout="wide",
    page_icon="📊"
)

st.markdown("""
<style>
body {
    background-color: #F5F7FA;
}
.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Nike Sales Performance Dashboard")

# Load data
df = pd.read_csv("../data/nike_sales.csv", parse_dates=["Invoice Date"])
df["Year"] = df["Invoice Date"].dt.year
df["Month"] = df["Invoice Date"].dt.month

# KPIs
col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"${df['Total Sales'].sum():,.0f}")
col2.metric("Units Sold", f"{df['Units Sold'].sum():,}")
col3.metric("Avg Price", f"${df['Price per Unit'].mean():.2f}")

st.divider()

# Sales Trend
st.subheader("📈 Monthly Sales Trend")

monthly = df.groupby(df["Invoice Date"].dt.to_period("M"))["Total Sales"].sum()
monthly.index = monthly.index.to_timestamp()

fig, ax = plt.subplots()
ax.plot(monthly.index, monthly.values)
ax.set_ylabel("Revenue")
st.pyplot(fig)

# Regional Breakdown
st.subheader("🌎 Sales by Region")

region_sales = df.groupby("Region")["Total Sales"].sum()

fig2, ax2 = plt.subplots()
region_sales.plot(kind="bar", ax=ax2)
st.pyplot(fig2)

# Product Performance
st.subheader("👟 Top Products")

top_products = (
    df.groupby("Product")["Total Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.dataframe(top_products.reset_index())
