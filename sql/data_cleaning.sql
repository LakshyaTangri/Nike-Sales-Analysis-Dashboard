-- Add derived time fields
ALTER TABLE nike_sales
ADD COLUMN year INT,
ADD COLUMN month INT,
ADD COLUMN quarter INT;

UPDATE nike_sales
SET
    year = EXTRACT(YEAR FROM invoice_date),
    month = EXTRACT(MONTH FROM invoice_date),
    quarter = EXTRACT(QUARTER FROM invoice_date);

-- Handle missing or negative values
DELETE FROM nike_sales
WHERE units_sold <= 0 OR total_sales <= 0;