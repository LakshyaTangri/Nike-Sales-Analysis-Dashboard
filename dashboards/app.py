import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Nike Sales Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
    }

    /* Card-like containers */
    .element-container {
        background-color: white;
        border-radius: 8px;
    }

    /* Header styling */
    h1 {
        color: #1f2937;
        font-weight: 700;
        padding-bottom: 1rem;
    }

    h2, h3 {
        color: #374151;
        font-weight: 600;
        padding-top: 1rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("../data/nike_sales.csv", parse_dates=["Invoice Date"])
    df["Year"] = df["Invoice Date"].dt.year
    df["Month"] = df["Invoice Date"].dt.month
    df["Month-Year"] = df["Invoice Date"].dt.to_period("M").astype(str)
    return df


df = load_data()

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
            line=dict(color="#3b82f6", width=3),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_sales["Month-Year"],
            y=monthly_sales["Units Sold"],
            name="Units Sold",
            line=dict(color="#10b981", width=2, dash="dot")
        ),
        secondary_y=True
    )

    fig.update_layout(
        height=400,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6', secondary_y=False)
    fig.update_yaxes(showgrid=False, secondary_y=True)

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
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig2.update_layout(
        height=400,
        showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white"
    )

    fig2.update_traces(
        textposition='inside',
        textinfo='percent+label',
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
        color_continuous_scale="Blues"
    )

    fig3.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Revenue ($)",
        yaxis_title=""
    )

    fig3.update_traces(
        hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>'
    )

    fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
    fig3.update_yaxes(showgrid=False)

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
            marker_color="#3b82f6",
            text=sales_method["Total Sales"],
            texttemplate='$%{text:,.0f}',
            textposition='outside'
        ))

        fig4.update_layout(
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="",
            yaxis_title="Revenue ($)",
            showlegend=False
        )

        fig4.update_xaxes(showgrid=False)
        fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')

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