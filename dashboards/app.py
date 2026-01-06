import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from tensorflow.keras.models import load_model
import numpy as np

st.set_page_config(
    page_title="Nike Sales Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for Nike dark theme
st.markdown("""
<style>
    /* App background - Nike black */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }

    /* Main content container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        color: #ffffff;
    }

    /* Sidebar - Dark gray */
    [data-testid="stSidebar"] {
        background-color: #111111;
        color: #ffffff;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
    }

    [data-testid="stMetricLabel"] {
        color: #cccccc;
        font-weight: 500;
        font-size: 14px;
    }

    [data-testid="stMetricDelta"] {
        color: #10b981;
    }

    /* Metric containers */
    [data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 1.5rem 1rem;
    }

    /* Divider */
    hr {
        border-color: #333333;
        margin: 2rem 0;
    }

    /* Info boxes */
    .stAlert {
        background-color: #1a1a1a;
        color: #ffffff;
        border-left: 4px solid #3b82f6;
    }

    /* Slider */
    .stSlider {
        color: #ffffff;
    }

    /* Select boxes and inputs */
    .stSelectbox label, .stDateInput label {
        color: #ffffff !important;
    }

    /* Download button */
    .stDownloadButton button {
        background-color: #ffffff;
        color: #000000;
        font-weight: 600;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 4px;
    }

    .stDownloadButton button:hover {
        background-color: #f5f5f5;
        border: none;
    }

    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #333333;
    }

    /* Text color fix for all elements */
    p, label, span {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "data" / "nike_sales.csv"

    data = pd.read_csv(data_path)

    # Force datetime conversion
    data["Invoice Date"] = pd.to_datetime(
        data["Invoice Date"],
        errors="coerce"
    )

    # Drop rows with invalid dates
    data = data.dropna(subset=["Invoice Date"])

    data["Year"] = data["Invoice Date"].dt.year
    data["Month"] = data["Invoice Date"].dt.month
    data["Month-Year"] = data["Invoice Date"].dt.to_period("M").astype(str)

    return data


df = load_data()

@st.cache_resource
def load_forecast_model():
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / "models" / "nn_sales_forecast.h5"
    return load_model(model_path)

# Sidebar filters
with st.sidebar:
    st.header("🎯 Filters")

    # Date range filter
    min_date = df["Invoice Date"].min().date()
    max_date = df["Invoice Date"].max().date()

    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Region filter
    regions = ["All"] + sorted(df["Region"].unique().tolist())
    selected_region = st.selectbox("Region", regions)

    # Product filter
    products = ["All"] + sorted(df["Product"].unique().tolist())
    selected_product = st.selectbox("Product", products)

    st.divider()
    st.info("💡 Use filters to drill down into specific segments")

# Apply filters
filtered_df = df.copy()

if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["Invoice Date"].dt.date >= date_range[0]) &
        (filtered_df["Invoice Date"].dt.date <= date_range[1])
        ]

if selected_region != "All":
    filtered_df = filtered_df[filtered_df["Region"] == selected_region]

if selected_product != "All":
    filtered_df = filtered_df[filtered_df["Product"] == selected_product]

def prepare_forecast_data(data):
    monthly = (
        data.groupby(["Year", "Month"])
        .agg({
            "Units Sold": "sum",
            "Price per Unit": "mean",
            "Total Sales": "sum"
        })
        .reset_index()
        .sort_values(["Year", "Month"])
    )

    # Scale features manually (same logic as training)
    monthly["Units Sold"] = monthly["Units Sold"] / monthly["Units Sold"].max()
    monthly["Price per Unit"] = monthly["Price per Unit"] / monthly["Price per Unit"].max()
    monthly["Month"] = monthly["Month"] / 12
    monthly["Quarter"] = ((monthly["Month"] * 12 - 1) // 3 + 1) / 4

    X = monthly[["Units Sold", "Price per Unit", "Month", "Quarter"]]
    y = monthly["Total Sales"]

    return X, y, monthly


# Main dashboard
st.title("📊 Nike Sales Performance Dashboard")
st.markdown("### Real-time insights into sales performance and trends")

# KPI Metrics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

total_revenue = filtered_df["Total Sales"].sum()
total_units = filtered_df["Units Sold"].sum()
avg_price = filtered_df["Price per Unit"].mean()
total_orders = len(filtered_df)

with col1:
    st.metric(
        "Total Revenue",
        f"${total_revenue:,.0f}",
        delta=f"{(total_revenue / df['Total Sales'].sum() * 100):.1f}% of total"
    )

with col2:
    st.metric(
        "Units Sold",
        f"{total_units:,}",
        delta=f"{total_units / len(filtered_df):.1f} avg/order"
    )

with col3:
    st.metric(
        "Avg Price per Unit",
        f"${avg_price:.2f}",
        delta=f"${avg_price - df['Price per Unit'].mean():.2f}"
    )

with col4:
    st.metric(
        "Total Orders",
        f"{total_orders:,}",
        delta=f"{(total_orders / len(df) * 100):.1f}% of total"
    )

st.markdown("---")

# Row 1: Sales Trend and Regional Distribution
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Monthly Sales Trend")

    monthly_sales = filtered_df.groupby("Month-Year").agg({
        "Total Sales": "sum",
        "Units Sold": "sum"
    }).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=monthly_sales["Month-Year"],
            y=monthly_sales["Total Sales"],
            name="Revenue",
            line=dict(color="#00ff00", width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_sales["Month-Year"],
            y=monthly_sales["Units Sold"],
            name="Units Sold",
            line=dict(color="#ffffff", width=2, dash="dot")
        ),
        secondary_y=True
    )

    fig.update_layout(
        height=400,
        hovermode="x unified",
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#ffffff")
        ),
        font=dict(color="#ffffff")
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#333333',
        color="#ffffff"
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#333333',
        secondary_y=False,
        color="#ffffff"
    )
    fig.update_yaxes(
        showgrid=False,
        secondary_y=True,
        color="#ffffff"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🌎 Regional Distribution")

    region_sales = filtered_df.groupby("Region")["Total Sales"].sum().reset_index()
    region_sales = region_sales.sort_values("Total Sales", ascending=False)

    fig2 = px.pie(
        region_sales,
        values="Total Sales",
        names="Region",
        hole=0.4,
        color_discrete_sequence=['#00ff00', '#ffffff', '#888888', '#444444', '#cccccc']
    )

    fig2.update_layout(
        height=400,
        showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#000000",
        font=dict(color="#ffffff"),
        legend=dict(font=dict(color="#ffffff"))
    )

    fig2.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont=dict(color="#000000", size=12, family="Arial Black"),
        hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}'
    )

    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Row 2: Top Products and Sales Method
col1, col2 = st.columns(2)

with col1:
    st.subheader("👟 Top 10 Products by Revenue")

    top_products = (
        filtered_df.groupby("Product")["Total Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig3 = px.bar(
        top_products,
        x="Total Sales",
        y="Product",
        orientation="h",
        color="Total Sales",
        color_continuous_scale=[[0, "#444444"], [0.5, "#888888"], [1, "#00ff00"]]
    )

    fig3.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Revenue ($)",
        yaxis_title="",
        font=dict(color="#ffffff")
    )

    fig3.update_traces(
        hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>'
    )

    fig3.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#333333',
        color="#ffffff"
    )
    fig3.update_yaxes(
        showgrid=False,
        color="#ffffff"
    )

    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("💳 Sales by Method")

    if "Sales Method" in filtered_df.columns:
        sales_method = filtered_df.groupby("Sales Method").agg({
            "Total Sales": "sum",
            "Units Sold": "sum"
        }).reset_index()

        fig4 = go.Figure()

        fig4.add_trace(go.Bar(
            x=sales_method["Sales Method"],
            y=sales_method["Total Sales"],
            name="Revenue",
            marker_color="#00ff00",
            text=sales_method["Total Sales"],
            texttemplate='$%{text:,.0f}',
            textposition='outside',
            textfont=dict(color="#ffffff")
        ))

        fig4.update_layout(
            height=400,
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="",
            yaxis_title="Revenue ($)",
            showlegend=False,
            font=dict(color="#ffffff")
        )

        fig4.update_xaxes(
            showgrid=False,
            color="#ffffff"
        )
        fig4.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#333333',
            color="#ffffff"
        )

        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Sales Method data not available in dataset")

st.markdown("---")

# Detailed Data Table
st.subheader("📋 Detailed Sales Data")

# Display options
show_rows = st.slider("Number of rows to display", 10, 100, 20)

display_df = filtered_df[[
    "Invoice Date", "Region", "Product", "Sales Method",
    "Units Sold", "Price per Unit", "Total Sales"
]].sort_values("Invoice Date", ascending=False).head(show_rows)

display_df["Invoice Date"] = display_df["Invoice Date"].dt.strftime("%Y-%m-%d")
display_df["Price per Unit"] = display_df["Price per Unit"].apply(lambda x: f"${x:.2f}")
display_df["Total Sales"] = display_df["Total Sales"].apply(lambda x: f"${x:,.2f}")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=400
)

# Download button
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name="nike_sales_filtered.csv",
    mime="text/csv"
)

st.markdown("---")
st.subheader("🔮 Predictive Sales Forecast (Neural Network)")

model = load_forecast_model()
X_pred, y_actual, monthly_df = prepare_forecast_data(filtered_df)

# Forecast horizon
forecast_months = st.slider(
    "Select forecast horizon (months)",
    min_value=1,
    max_value=12,
    value=6
)

# Model predictions (historical)
historical_preds = model.predict(X_pred).flatten()

# Future forecasting (naive forward projection)
last_row = X_pred.iloc[-1].values
future_preds = []

for i in range(forecast_months):
    pred = model.predict(last_row.reshape(1, -1))[0][0]
    future_preds.append(pred)

# Create future date index
last_year = monthly_df.iloc[-1]["Year"]
last_month = monthly_df.iloc[-1]["Month"]

future_dates = pd.date_range(
    start=f"{int(last_year)}-{int(last_month)}-01",
    periods=forecast_months + 1,
    freq="M"
)[1:]

# Plot
fig_forecast = go.Figure()

fig_forecast.add_trace(go.Scatter(
    x=pd.to_datetime(monthly_df["Year"].astype(str) + "-" + monthly_df["Month"].astype(str)),
    y=y_actual,
    name="Actual Sales",
    line=dict(color="#ffffff", width=2)
))

fig_forecast.add_trace(go.Scatter(
    x=pd.to_datetime(monthly_df["Year"].astype(str) + "-" + monthly_df["Month"].astype(str)),
    y=historical_preds,
    name="Predicted Sales",
    line=dict(color="#00ff00", width=3, dash="dot")
))

fig_forecast.add_trace(go.Scatter(
    x=future_dates,
    y=future_preds,
    name="Forecast",
    line=dict(color="#00ff00", width=3),
    mode="lines+markers"
))

fig_forecast.update_layout(
    height=450,
    plot_bgcolor="#000000",
    paper_bgcolor="#000000",
    font=dict(color="#ffffff"),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.1)
)

fig_forecast.update_xaxes(
    showgrid=True,
    gridcolor="#333333",
    color="#ffffff"
)

fig_forecast.update_yaxes(
    showgrid=True,
    gridcolor="#333333",
    color="#ffffff",
    title="Revenue ($)"
)

st.plotly_chart(fig_forecast, use_container_width=True)
