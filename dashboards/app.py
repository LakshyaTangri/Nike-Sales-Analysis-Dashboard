import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
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
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #111111;
        color: #ffffff;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700;
    }
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
    [data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 1.5rem 1rem;
    }
    hr {
        border-color: #333333;
        margin: 2rem 0;
    }
    .stAlert {
        background-color: #1a1a1a;
        color: #ffffff;
        border-left: 4px solid #3b82f6;
    }
    .stSelectbox label, .stDateInput label, .stSlider label {
        color: #ffffff !important;
    }
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
    }
    p, label, span {
        color: #ffffff;
    }
    .forecast-metric {
        background-color: #1a1a1a;
        border: 2px solid #00ff00;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "data" / "nike_sales.csv"

    data = pd.read_csv(data_path)
    data["Invoice Date"] = pd.to_datetime(data["Invoice Date"], errors="coerce")
    data = data.dropna(subset=["Invoice Date"])

    data["Year"] = data["Invoice Date"].dt.year
    data["Month"] = data["Invoice Date"].dt.month
    data["Quarter"] = data["Invoice Date"].dt.quarter
    data["Month-Year"] = data["Invoice Date"].dt.to_period("M").astype(str)

    return data


@st.cache_resource
def load_forecast_model():
    """Load the trained neural network model"""
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Try the fixed model first, fall back to original
    model_paths = [
        BASE_DIR / "models" / "nn_sales_forecast_fixed.keras",
        BASE_DIR / "models" / "nn_sales_forecast.h5"
    ]

    for model_path in model_paths:
        if model_path.exists():
            try:
                return load_model(model_path)
            except Exception as e:
                st.sidebar.warning(f"Could not load {model_path.name}: {str(e)}")
                continue

    return None


def prepare_model_features(data):
    """Prepare features exactly as done during training"""
    # Basic cleaning
    data = data[data["Units Sold"] > 0]
    data = data[data["Total Sales"] > 0]

    # Monthly aggregation
    monthly_df = (
        data.groupby(["Year", "Month", "Quarter"])
        .agg({
            "Units Sold": "sum",
            "Total Sales": "sum",
            "Price per Unit": "mean"
        })
        .reset_index()
    )

    # Prepare features
    feature_cols = ["Units Sold", "Price per Unit", "Month", "Quarter"]
    X = monthly_df[feature_cols].values
    y = monthly_df["Total Sales"].values

    # Scale features using StandardScaler (same as training)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Scale target
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    return X_scaled, y, monthly_df, scaler_y


def generate_future_features(last_features, months_ahead, scaler_X):
    """Generate features for future predictions"""
    future_features = []
    current = last_features.copy()

    for i in range(months_ahead):
        # Increment month (cyclical)
        month_idx = 2  # Month is the 3rd feature
        current[month_idx] = (current[month_idx] * 12 + 1) % 12 / 12

        # Update quarter based on month
        quarter_idx = 3
        month_val = int(current[month_idx] * 12)
        current[quarter_idx] = ((month_val - 1) // 3 + 1) / 4

        future_features.append(current.copy())

    return np.array(future_features)


# Load data
df = load_data()

# Sidebar filters
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a6/Logo_NIKE.svg", width=100)
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

    # Sales Method filter
    if "Sales Method" in df.columns:
        methods = ["All"] + sorted(df["Sales Method"].unique().tolist())
        selected_method = st.selectbox("Sales Method", methods)
    else:
        selected_method = "All"

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

if selected_method != "All" and "Sales Method" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Sales Method"] == selected_method]

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

# Calculate growth rates
current_month = filtered_df[filtered_df["Invoice Date"] == filtered_df["Invoice Date"].max()]
prev_month = filtered_df[filtered_df["Invoice Date"] == filtered_df["Invoice Date"].max() - pd.DateOffset(months=1)]

if len(prev_month) > 0:
    revenue_growth = ((current_month["Total Sales"].sum() - prev_month["Total Sales"].sum()) /
                      prev_month["Total Sales"].sum() * 100)
else:
    revenue_growth = 0

with col1:
    st.metric(
        "Total Revenue",
        f"${total_revenue:,.0f}",
        delta=f"{revenue_growth:+.1f}% MoM"
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
        delta=f"${avg_price - df['Price per Unit'].mean():+.2f}"
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

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333333', color="#ffffff")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333', secondary_y=False, color="#ffffff",
                     title="Revenue ($)")
    fig.update_yaxes(showgrid=False, secondary_y=True, color="#ffffff", title="Units")

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

    fig3.update_traces(hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>')
    fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333333', color="#ffffff")
    fig3.update_yaxes(showgrid=False, color="#ffffff")

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

        fig4.update_xaxes(showgrid=False, color="#ffffff")
        fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333', color="#ffffff")

        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Sales Method data not available in dataset")

st.markdown("---")

# AI-Powered Sales Forecast Section
st.subheader("🔮 AI-Powered Sales Forecast")
st.markdown("*Neural network predictions based on historical patterns*")

model = load_forecast_model()

if model is not None:
    try:
        # Prepare data for forecasting
        X_scaled, y_actual, monthly_df, scaler_y = prepare_model_features(filtered_df)

        # Make predictions on historical data
        historical_preds_scaled = model.predict(X_scaled, verbose=0)
        historical_preds = scaler_y.inverse_transform(historical_preds_scaled).ravel()

        # Calculate model accuracy metrics
        mae = mean_absolute_error(y_actual, historical_preds)
        r2 = r2_score(y_actual, historical_preds)
        mape = np.mean(np.abs((y_actual - historical_preds) / y_actual)) * 100

        # Display model performance
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Accuracy (R²)", f"{r2:.2%}")
        with col2:
            st.metric("Mean Absolute Error", f"${mae:,.0f}")
        with col3:
            st.metric("Avg Error", f"{mape:.1f}%")

        st.markdown("---")

        # Forecast controls
        col1, col2 = st.columns([3, 1])

        with col1:
            forecast_months = st.slider(
                "Forecast Horizon (months)",
                min_value=1,
                max_value=12,
                value=6,
                help="Select how many months ahead to forecast"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            show_confidence = st.checkbox("Show confidence interval", value=True)

        # Generate future predictions
        last_features = X_scaled[-1]
        future_features = generate_future_features(last_features, forecast_months, None)

        future_preds_scaled = model.predict(future_features, verbose=0)
        future_preds = scaler_y.inverse_transform(future_preds_scaled).ravel()

        # Create future date range
        last_date = pd.to_datetime(
            f"{int(monthly_df.iloc[-1]['Year'])}-{int(monthly_df.iloc[-1]['Month'])}-01"
        )
        future_dates = pd.date_range(start=last_date, periods=forecast_months + 1, freq='MS')[1:]

        # Historical dates
        historical_dates = pd.to_datetime(
            monthly_df["Year"].astype(str) + "-" + monthly_df["Month"].astype(str) + "-01"
        )

        # Create forecast visualization
        fig_forecast = go.Figure()

        # Actual historical sales
        fig_forecast.add_trace(go.Scatter(
            x=historical_dates,
            y=y_actual,
            name="Actual Sales",
            line=dict(color="#ffffff", width=3),
            mode="lines+markers",
            marker=dict(size=8)
        ))

        # Predicted historical sales
        fig_forecast.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_preds,
            name="Model Prediction",
            line=dict(color="#00ff00", width=2, dash="dot"),
            mode="lines"
        ))

        # Future forecast
        fig_forecast.add_trace(go.Scatter(
            x=future_dates,
            y=future_preds,
            name="Forecast",
            line=dict(color="#00ff00", width=4),
            mode="lines+markers",
            marker=dict(size=10, symbol="star")
        ))

        # Add confidence interval if requested
        if show_confidence:
            # Simple confidence interval based on historical error
            std_error = np.std(y_actual - historical_preds)
            upper_bound = future_preds + 1.96 * std_error
            lower_bound = future_preds - 1.96 * std_error

            fig_forecast.add_trace(go.Scatter(
                x=future_dates,
                y=upper_bound,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ))

            fig_forecast.add_trace(go.Scatter(
                x=future_dates,
                y=lower_bound,
                mode="lines",
                line=dict(width=0),
                fillcolor="rgba(0, 255, 0, 0.1)",
                fill="tonexty",
                name="95% Confidence",
                hoverinfo="skip"
            ))

        fig_forecast.update_layout(
            height=500,
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
            font=dict(color="#ffffff"),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_title="Date",
            yaxis_title="Revenue ($)"
        )

        fig_forecast.update_xaxes(
            showgrid=True,
            gridcolor="#333333",
            color="#ffffff"
        )

        fig_forecast.update_yaxes(
            showgrid=True,
            gridcolor="#333333",
            color="#ffffff"
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        # Forecast summary table
        st.markdown("#### 📊 Forecast Summary")

        forecast_df = pd.DataFrame({
            "Month": future_dates.strftime("%B %Y"),
            "Predicted Sales": [f"${x:,.0f}" for x in future_preds],
            "Growth vs Last Month": [""] + [
                f"{((future_preds[i] - future_preds[i - 1]) / future_preds[i - 1] * 100):+.1f}%"
                for i in range(1, len(future_preds))]
        })

        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        # Key insights
        st.markdown("#### 💡 Key Insights")

        total_forecast = sum(future_preds)
        avg_forecast = np.mean(future_preds)
        trend = "increasing" if future_preds[-1] > future_preds[0] else "decreasing"
        trend_pct = abs((future_preds[-1] - future_preds[0]) / future_preds[0] * 100)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="forecast-metric">
                <h4 style="margin:0; color:#00ff00;">Total Forecast Revenue</h4>
                <h2 style="margin:0;">${total_forecast:,.0f}</h2>
                <p style="margin:0; color:#888;">Next {forecast_months} months</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="forecast-metric">
                <h4 style="margin:0; color:#00ff00;">Avg Monthly Revenue</h4>
                <h2 style="margin:0;">${avg_forecast:,.0f}</h2>
                <p style="margin:0; color:#888;">Projected average</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="forecast-metric">
                <h4 style="margin:0; color:#00ff00;">Trend</h4>
                <h2 style="margin:0;">{trend_pct:.1f}% {trend}</h2>
                <p style="margin:0; color:#888;">Over forecast period</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")
        st.info("Please ensure the model was trained with the correct data format.")
else:
    st.warning("⚠️ Forecast model not found. Please train the model first by running `sales_forecasting_nn_fixed.py`")
    st.info(
        "The model file should be located at `models/nn_sales_forecast_fixed.keras` or `models/nn_sales_forecast.h5`")

st.markdown("---")

# Detailed Data Table
st.subheader("📋 Detailed Sales Data")

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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem 0;">
    <p>Nike Sales Dashboard | Using Data analytics & Neural Networks for predicting sales</p>
    <p style="font-size: 0.8rem;">Data-driven insights for strategic decision making</p>
</div>
""", unsafe_allow_html=True)