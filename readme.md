
# <img src="https://upload.wikimedia.org/wikipedia/commons/a/a6/Logo_NIKE.svg" alt="Nike Logo" width="100"/> NIKE PREDICTIVE SALES ANALYSIS DASHBOARD


Welcome to the Nike Sales Analysis Dashboard repository! This comprehensive project leverages advanced machine learning techniques, neural network forecasting, and interactive data visualization to deliver actionable insights into Nike's sales performance, trends, and future projections.

---

## 🎯 OBJECTIVE

Our primary aim is to transform raw sales data into strategic insights that drive data-driven decision-making across the organization. Through sophisticated exploratory data analysis (EDA), feature engineering, and AI-powered predictive modeling, we provide stakeholders with:

- **Comprehensive Sales Insights:** Deep understanding of revenue patterns, product performance, and regional dynamics
- **Predictive Intelligence:** Neural network-powered forecasts to anticipate future sales trends
- **Interactive Visualization:** Real-time dashboard for exploring sales metrics and KPIs
- **Strategic Recommendations:** Data-backed insights for inventory management, marketing strategies, and resource allocation

---

## 📁 PROJECT STRUCTURE

```
Nike-Sales-Analysis-Dashboard/
├── data/
│   └── nike_sales.csv                    # Raw sales data (2020-2021)
├── notebooks/
│   ├── exploratory_analysis.py           # Comprehensive EDA with visualizations
│   ├── feature_engineering.py            # Data preprocessing and feature creation
│   ├── sales_forecasting_nn.py           # Neural network model (original)
│   └── sales_forecasting_nn_fixed.py     # Optimized neural network model
├── models/
│   ├── nn_sales_forecast.h5              # Trained TensorFlow model (legacy)
│   └── nn_sales_forecast_fixed.keras     # Optimized model with proper scaling
├── app.py                                # Interactive Streamlit dashboard
├── requirements.txt                       # Python dependencies
└── README.md                             # Project documentation
```

---

## 📊 DATA

**Dataset Information:** 
- **Time Period:** January 2020 - December 2021 (24 months)
- **Total Records:** 9,356 individual sales transactions
- **Geographic Coverage:** Multiple regions including West, Midwest, South, Southeast, and Northeast
- **Product Range:** Diverse Nike product portfolio including footwear and apparel

**Key Features:**
- `Invoice Date`: Transaction date
- `Region`: Geographic sales region
- `Product`: Specific Nike product name
- `Sales Method`: Sales channel (In-store, Online, Outlet)
- `Units Sold`: Quantity of products sold
- `Price per Unit`: Unit selling price
- `Total Sales`: Total transaction revenue

**Data Quality:**
- Zero missing values after preprocessing
- All transactions validated for positive sales and units
- Dates properly formatted and validated
- Outliers retained for authentic market representation

---

## 🔧 TECHNOLOGY STACK

### **Data Processing & Analysis**
- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and aggregation
- **NumPy**: Numerical computing and array operations

### **Machine Learning & AI**
- **TensorFlow 2.x / Keras**: Deep learning framework for neural networks
- **Scikit-learn**: Feature scaling, preprocessing, and metrics evaluation
  - StandardScaler for feature normalization
  - Train-test splitting
  - Performance metrics (MAE, RMSE, R²)

### **Visualization**
- **Matplotlib**: Static plot generation for EDA
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive charts for dashboard
- **Streamlit**: Web-based interactive dashboard framework

### **Model Architecture**
- Sequential Neural Network with:
  - Dense layers (32→16→8→1 neurons)
  - Dropout regularization (20%)
  - Adam optimizer with adaptive learning rate
  - Mean Squared Error (MSE) loss function
  - Early stopping for optimal training

---

## 🧹 DATA CLEANING & PREPROCESSING

### **Phase 1: Initial Data Validation**
1. **Date Parsing:** Converted string dates to proper datetime objects
2. **Data Type Validation:** Ensured numeric fields are properly typed
3. **Missing Value Check:** Verified dataset integrity (zero nulls found)

### **Phase 2: Data Cleaning**
```python
# Removed invalid records
- Filtered out transactions with Units Sold ≤ 0
- Filtered out transactions with Total Sales ≤ 0
- Result: 9,356 valid transactions retained
```

### **Phase 3: Feature Engineering**
```python
# Time-based features
- Year extraction (2020, 2021)
- Month extraction (1-12)
- Quarter calculation (Q1-Q4)
- Month-Year period for time series analysis

# Aggregation features
- Monthly total units sold
- Monthly total sales revenue
- Monthly average price per unit
```

### **Phase 4: Feature Scaling**
- **Approach:** StandardScaler (zero mean, unit variance)
- **Features Scaled:** Units Sold, Price per Unit, Month, Quarter
- **Target Scaling:** Total Sales (separate scaler for inverse transformation)
- **Reason:** Neural networks require normalized inputs for optimal convergence

---

## 🔍 EXPLORATORY DATA ANALYSIS

The exploratory analysis phase uncovered critical insights through four key visualizations:

### **1. Monthly Revenue Trend Analysis**
- **Observation:** Seasonal patterns with peaks in specific months
- **Insight:** Holiday seasons and product launches drive significant spikes
- **Business Value:** Informs inventory planning and marketing campaign timing

### **2. Regional Performance Distribution**
- **Observation:** Revenue distribution varies significantly across regions
- **Insight:** Geographic expansion opportunities and market penetration strategies

### **3. Top 10 Products by Revenue**
- **Observation:** Pareto principle applies - top products drive majority of revenue
- **Insight:** Focus marketing and inventory on high-performing SKUs
- **Business Value:** Optimize product mix and shelf space allocation

### **4. Sales Method Effectiveness**
- **Channels Analyzed:** In-store, Online, Outlet
- **Observation:** Channel-specific performance patterns
- **Insight:** Multi-channel strategy optimization opportunities

---

## 🤖 MACHINE LEARNING MODEL

### **Model Architecture: Deep Neural Network**

```
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 32)             │           160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 32)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 16)             │           528 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 16)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 8)              │           136 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 1)              │             9 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total params: 833 (3.25 KB)
Trainable params: 833 (3.25 KB)
Non-trainable params: 0 (0.00 B)
```

### **Model Training Configuration**

- **Input Features (4):**
  - Units Sold (normalized)
  - Average Price per Unit (normalized)
  - Month (cyclical feature)
  - Quarter (categorical encoded)

- **Target Variable:**
  - Total Sales (monthly aggregated revenue)

- **Training Parameters:**
  - Train/Test Split: 80/20 (19 training samples, 5 test samples)
  - Epochs: Up to 200 with early stopping
  - Batch Size: 8
  - Validation Split: 20% of training data
  - Early Stopping Patience: 20 epochs
  - Optimizer: Adam (learning_rate=0.01)
  - Loss Function: Mean Squared Error (MSE)

### **Model Performance Metrics**

```
Model Performance (Test Set):
├── R² Score: 0.XX (Model explains XX% of variance)
├── Mean Absolute Error (MAE): $X,XXX
├── Root Mean Squared Error (RMSE): $X,XXX
└── Mean Absolute Percentage Error (MAPE): XX.X%
```

### **Key Model Features**

1. **Dropout Regularization (20%):** Prevents overfitting on small dataset
2. **ReLU Activation:** Non-linear transformations for complex pattern learning
3. **Separate Scalers:** Independent scaling for features and target
4. **Early Stopping:** Automatic training termination at optimal point

---

## 📈 PROBLEM STATEMENT & ANALYTICAL QUESTIONS

This project addresses critical business questions through data-driven analysis:

### **Revenue & Performance Analysis**
1. **What are the monthly revenue trends?** 
   - Time series analysis reveals seasonal patterns and growth trajectories
   
2. **Which products drive the most revenue?**
   - Product-level analysis identifies high-value SKUs for strategic focus

3. **How do different regions perform?**
   - Geographic analysis uncovers market opportunities and optimization areas

### **Sales Channel Effectiveness**
4. **Which sales methods are most effective?**
   - Channel comparison reveals customer preferences and conversion rates

5. **What is the optimal channel mix?**
   - Multi-channel analysis informs resource allocation decisions

### **Predictive Intelligence**
6. **What are the expected sales for the next 6-12 months?**
   - Neural network forecasts enable proactive planning

7. **What are the confidence intervals for predictions?**
   - Statistical bounds quantify forecast uncertainty

8. **How do seasonal patterns affect future sales?**
   - Time-based features capture cyclical trends

### **Strategic Insights**
9. **Which products should receive increased marketing investment?**
   - Performance metrics guide budget allocation

10. **What inventory levels should be maintained?**
    - Demand forecasts optimize stock management

---
## 🚀 INSTALLATION & SETUP

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Windows/Mac/Linux operating system

### **Step 1: Clone Repository**
```bash
git clone https://github.com/LakshyaTangri/Nike-Sales-Analysis-Dashboard.git
cd Nike-Sales-Analysis-Dashboard
```

### **Step 2: Create Virtual Environment (Recommended)**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Required Packages:**
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tensorflow>=2.6.0
streamlit>=1.10.0
plotly>=5.0.0
```

### **Step 4: Install Microsoft Visual C++ Redistributable (Windows Only)**

If you encounter TensorFlow DLL errors:
1. Download: [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Run installer
3. Restart computer if prompted

---

## 💻 USAGE GUIDE

### **Option 1: Run Complete Analysis Pipeline**

**1. Exploratory Data Analysis**
```bash
cd notebooks
python exploratory_analysis.py
```
**Output:**
- 4 visualization plots (saved and displayed)
- Statistical summaries in console
- Data quality report

**2. Train Neural Network Model**
```bash
python sales_forecasting_nn_fixed.py
```
**Output:**
- Model training progress (85-200 epochs)
- Performance metrics (R², MAE, RMSE)
- 4 evaluation plots
- Saved model: `models/nn_sales_forecast_fixed.keras`
- Sample predictions table

### **Option 2: Interactive Dashboard**

**Launch Streamlit Dashboard**
```bash
streamlit run app.py
```

**Dashboard Features:**
1. **Filters Sidebar:**
   - Date range selection
   - Region filter
   - Product filter
   - Sales method filter

2. **KPI Metrics:**
   - Total Revenue with MoM growth
   - Units Sold with averages
   - Average Price per Unit
   - Total Orders count

3. **Interactive Visualizations:**
   - Monthly sales trend (dual-axis)
   - Regional distribution (pie chart)
   - Top 10 products (horizontal bar)
   - Sales method comparison (bar chart)

4. **AI-Powered Forecast:**
   - Adjustable forecast horizon (1-12 months)
   - Confidence intervals (95%)
   - Model accuracy metrics
   - Forecast summary table
   - Key insights panel

5. **Data Export:**
   - Download filtered data as CSV
   - Detailed transaction table

---

## 🔬 MODEL TRAINING DETAILS

### **Training Process Flow**

```
1. Data Loading & Validation
   ↓
2. Data Cleaning (remove invalid records)
   ↓
3. Feature Engineering (time-based features)
   ↓
4. Monthly Aggregation (24 months → 24 samples)
   ↓
5. Feature Scaling (StandardScaler)
   ↓
6. Train/Test Split (80/20)
   ↓
7. Model Architecture Definition
   ↓
8. Model Compilation (Adam optimizer)
   ↓
9. Training with Early Stopping
   ↓
10. Model Evaluation (metrics calculation)
    ↓
11. Visualization Generation (4 plots)
    ↓
12. Model Saving (.keras format)
```

### **Visualization Outputs**

The model training generates four comprehensive plots:

1. **Training History Plot**
   - Shows loss convergence over epochs
   - Validation loss for overfitting detection
   - Helps identify optimal training duration

2. **Actual vs Predicted Plot**
   - Time series comparison
   - Visual assessment of model accuracy
   - Identifies systematic errors

3. **Prediction Accuracy Scatter**
   - Perfect prediction line reference
   - Scatter of predictions vs actuals
   - R² score visualization

4. **Residuals Plot**
   - Error distribution analysis
   - Identifies bias in predictions
   - Validates model assumptions

---

## ⚠️ LIMITATIONS & CONSIDERATIONS

### **Data Limitations**

1. **Limited Historical Data**
   - Only 24 months of data (2020-2021)
   - Small sample size for deep learning (24 monthly aggregates)
   - May not capture long-term cyclical patterns

2. **COVID-19 Impact**
   - Data period includes pandemic era
   - Unusual market conditions may affect patterns
   - Future predictions assume similar conditions

3. **Missing External Factors**
   - No marketing spend data
   - No competitor activity information
   - No macroeconomic indicators
   - No inventory constraints data

### **Model Limitations**

1. **Small Training Dataset**
   - Only 19 training samples after 80/20 split
   - Risk of overfitting despite dropout regularization
   - Limited generalization capability

2. **Feature Set Constraints**
   - Only 4 input features
   - Simple time-based features
   - No product-level or customer-level attributes

3. **Forecast Uncertainty**
   - Confidence intervals are statistical estimates
   - Actual future performance may vary
   - External shocks not accounted for

4. **Aggregation Effects**
   - Monthly aggregation smooths daily variations
   - Loss of granular patterns
   - Cannot predict daily sales

### **Technical Limitations**

1. **Computational Requirements**
   - TensorFlow requires significant memory
   - Windows users need Visual C++ Redistributable
   - GPU acceleration not utilized (CPU-only training)

2. **Scalability Concerns**
   - Current architecture not optimized for large datasets
   - Real-time predictions not implemented
   - Single-threaded processing

---

## 🎯 BUSINESS RECOMMENDATIONS

Based on the comprehensive analysis, we recommend:

### **Immediate Actions (0-3 months)**

1. **Inventory Optimization**
   - Increase stock for top 10 revenue-generating products
   - Reduce inventory for underperforming SKUs
   - Align stock levels with forecast predictions

2. **Marketing Focus**
   - Concentrate marketing spend on high-performing regions
   - Launch targeted campaigns during predicted peak months
   - Emphasize top-selling product categories

3. **Channel Strategy**
   - Optimize resource allocation based on channel performance
   - Enhance underperforming channel experiences
   - Integrate multi-channel customer journeys

### **Medium-Term Initiatives (3-6 months)**

1. **Regional Expansion**
   - Investigate growth opportunities in underserved regions
   - Replicate successful strategies from top-performing areas
   - Localize product offerings based on regional preferences

2. **Product Portfolio Management**
   - Phase out low-performing products
   - Introduce new products in high-demand categories
   - Test price optimization strategies

3. **Demand Forecasting Integration**
   - Integrate ML forecasts into planning systems
   - Establish monthly forecast review process
   - Track forecast accuracy and refine models

### **Long-Term Strategy (6-12 months)**

1. **Advanced Analytics Implementation**
   - Collect additional data (marketing spend, customer demographics)
   - Develop customer segmentation models
   - Build price elasticity models

2. **Technology Infrastructure**
   - Deploy real-time dashboard for executives
   - Automate reporting and alerting
   - Implement A/B testing framework

3. **Continuous Improvement**
   - Retrain models quarterly with new data
   - Experiment with advanced architectures (LSTM, Prophet)
   - Expand feature set with external data sources

---

## 🔮 FUTURE ENHANCEMENTS

### **Planned Features**

- [ ] **Real-time Data Pipeline:** Automated data ingestion from sales systems
- [ ] **Customer Segmentation:** RFM analysis and clustering
- [ ] **Price Optimization:** Dynamic pricing recommendations
- [ ] **Product Recommendation Engine:** Cross-sell and upsell suggestions
- [ ] **Anomaly Detection:** Automated alerts for unusual sales patterns
- [ ] **Mobile Dashboard:** Responsive design for mobile devices
- [ ] **API Development:** RESTful API for model predictions
- [ ] **A/B Testing Framework:** Experiment tracking and analysis

### **Model Improvements**

- [ ] **LSTM Networks:** Better handling of time series patterns
- [ ] **Ensemble Methods:** Combine multiple models for improved accuracy
- [ ] **Transfer Learning:** Leverage pre-trained models
- [ ] **Hyperparameter Tuning:** Automated optimization with Optuna
- [ ] **Feature Engineering:** Advanced time series features (lag, rolling statistics)
- [ ] **External Data Integration:** Weather, holidays, economic indicators

---


## 📝 LICENSE

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 AUTHORS & ACKNOWLEDGMENTS

**Project Author:** Lakshya Tangri
- GitHub: [@LakshyaTangri](https://github.com/LakshyaTangri)
- Project Link: [Nike Sales Analysis Dashboard](https://github.com/LakshyaTangri/Nike-Sales-Analysis-Dashboard)

**Acknowledgments:**
- Nike for inspiration in sales analytics
- TensorFlow and Keras teams for deep learning frameworks
- Streamlit for interactive dashboard capabilities
- The open-source community for invaluable tools and libraries
- https://www.kaggle.com/datasets/krishnavamsis/nike-sales for database

---

## 📧 CONTACT & SUPPORT

For questions, suggestions, or collaboration opportunities:

- **Create an Issue:** [GitHub Issues](https://github.com/LakshyaTangri/Nike-Sales-Analysis-Dashboard/issues)
- **Email:** info@lakshyatangri.com
- **LinkedIn:** linkedin.com/in/lakshyatangri

---

## 📚 REFERENCES & RESOURCES

### **Technical Documentation**
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Guide](https://plotly.com/python/)

### **Research Papers & Articles**
- Time Series Forecasting with Neural Networks
- Sales Prediction using Machine Learning
- Business Intelligence Dashboard Design Principles

### **Datasets & Tools**
- Python Data Analysis Library (Pandas)
- NumPy for Numerical Computing
- Matplotlib & Seaborn for Visualization

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.10+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Built with ❤️ for Data-Driven Decision Making**

</div>
