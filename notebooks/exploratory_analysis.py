"""
Nike Sales Exploratory Data Analysis
Author: Your Name
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 6)


def run_eda():
    # Load data
    df = pd.read_csv("../data/nike_sales.csv", parse_dates=["Invoice Date"])

    print("Dataset Info:")
    print(df.info())
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Feature Extraction
    df["Year"] = df["Invoice Date"].dt.year
    df["Month"] = df["Invoice Date"].dt.month
    df["Quarter"] = df["Invoice Date"].dt.quarter

    # Revenue Trend Over Time
    monthly_sales = (
        df.groupby(df["Invoice Date"].dt.to_period("M"))["Total Sales"]
        .sum()
        .reset_index()
    )
    monthly_sales["Invoice Date"] = monthly_sales["Invoice Date"].dt.to_timestamp()

    plt.plot(monthly_sales["Invoice Date"], monthly_sales["Total Sales"])
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()

    # Regional Performance
    region_sales = df.groupby("Region")["Total Sales"].sum().sort_values()
    region_sales.plot(kind="barh", title="Sales by Region")
    plt.tight_layout()
    plt.show()

    # Product Performance
    top_products = (
        df.groupby("Product")["Total Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    top_products.plot(kind="bar", title="Top 10 Products by Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Sales Method Analysis
    sns.boxplot(data=df, x="Sales Method", y="Total Sales")
    plt.title("Sales Distribution by Method")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_eda()
