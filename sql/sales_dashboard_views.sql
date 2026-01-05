
--Total Sales

CREATE VIEW v_sales_kpis AS
SELECT
    SUM(total_sales) AS total_revenue,
    SUM(units_sold) AS total_units_sold,
    AVG(price_per_unit) AS avg_price
FROM nike_sales;

--Sales by Region
CREATE VIEW v_sales_by_region AS
SELECT
    region,
    SUM(total_sales) AS revenue,
    SUM(units_sold) AS units
FROM nike_sales
GROUP BY region;

-- Product Performance
CREATE VIEW v_product_performance AS
SELECT
    product,
    SUM(units_sold) AS units_sold,
    SUM(total_sales) AS revenue
FROM nike_sales
GROUP BY product
ORDER BY revenue DESC;

--Monthly Sales Trend
CREATE VIEW v_monthly_sales AS
SELECT
    year,
    month,
    SUM(total_sales) AS revenue
FROM nike_sales
GROUP BY year, month
ORDER BY year, month;
