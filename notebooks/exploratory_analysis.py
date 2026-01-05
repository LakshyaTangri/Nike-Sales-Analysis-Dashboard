import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10,6)

df = pd.read_csv("../data/nike_sales.csv", parse_dates=["Invoice Date"])
df.head()

df.info()
df.describe()

df.isnull().sum()

#Feature Extraction
df["Year"] = df["Invoice Date"].dt.year
df["Month"] = df["Invoice Date"].dt.month
df["Quarter"] = df["Invoice Date"].dt.quarter

#Revenue Trend Over Time
monthly_sales = (
    df.groupby(df["Invoice Date"].dt.to_period("M"))
    ["Total Sales"]
    .sum()
    .reset_index()
)

monthly_sales["Invoice Date"] = monthly_sales["Invoice Date"].dt.to_timestamp()

plt.plot(monthly_sales["Invoice Date"], monthly_sales["Total Sales"])
plt.title("Monthly Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()

# Regional Performance
region_sales = df.groupby("Region")["Total Sales"].sum().sort_values()

region_sales.plot(kind="barh", title="Sales by Region")
plt.show()

#Product Performance
top_products = (
    df.groupby("Product")["Total Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_products.plot(kind="bar", title="Top 10 Products by Revenue")
plt.xticks(rotation=45)
plt.show()

#Sales Method Analysis
sns.boxplot(data=df, x="Sales Method", y="Total Sales")
plt.title("Sales Distribution by Method")
plt.show()
