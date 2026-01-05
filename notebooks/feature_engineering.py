df = pd.read_csv("../data/nike_sales.csv", parse_dates=["Invoice Date"])

df = df[df["Units Sold"] > 0]
df = df[df["Total Sales"] > 0]

# Time-Based Features
df["Year"] = df["Invoice Date"].dt.year
df["Month"] = df["Invoice Date"].dt.month
df["Quarter"] = df["Invoice Date"].dt.quarter
df["Day"] = df["Invoice Date"].dt.day

# Aggregation for Forecasting (Monthly)

monthly_df = (
    df.groupby(["Year", "Month"])
    .agg({
        "Units Sold": "sum",
        "Total Sales": "sum",
        "Price per Unit": "mean"
    })
    .reset_index()
)
monthly_df.head()

#Scaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

features = ["Units Sold", "Price per Unit", "Month", "Quarter"]
monthly_df[features] = scaler.fit_transform(monthly_df[features])

#slpit
X = monthly_df[features]
y = monthly_df["Total Sales"]

split = int(len(monthly_df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
