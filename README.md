# ML-Mastery
A collection of Python &amp; Machine Learning projects as part of my journey towards research &amp; industry excellence.


#1 🏡 Airbnb NYC Exploratory Data Analysis  

## 📌 Project Overview  
This project performs **Exploratory Data Analysis (EDA)** on Airbnb listings in New York City to understand pricing patterns and the factors influencing them.  

## 🛠️ Technologies Used  
- **Python** (Pandas, NumPy)  
- **Data Visualization** (Matplotlib, Seaborn)  

## 🔍 Key Insights  
- **Price Trends:** Prices vary significantly across boroughs; **Manhattan** listings are the most expensive.  
- **Outliers:** Some listings have unrealistic prices ($5,000+), affecting analysis.  
- **Correlations:** Room type, neighborhood, and availability strongly influence pricing.  

## 📂 Project Structure  
📁 ML-Mastery
│── 📁 Notebooks
│ ├── Airbnb_EDA.ipynb (Main analysis & visualizations)
│── 📁 Datasets
│ ├── Airbnb_NYC.csv (Raw dataset)
│ ├── Airbnb_NYC_Cleaned.csv (Processed dataset)
│── 📁 Visualizations
│ ├── price_distribution.png
│ ├── heatmap_correlation.png
│── README.md (Project documentation)

## 🚀 Next Steps  
- **Feature Engineering** for machine learning models.  
- **Predictive Modeling** for Airbnb pricing.  
- **Deployment of an interactive dashboard** for price recommendations.


## 📌 Dataset Source  
[New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)  




#2 Airbnb Price Prediction Project
##📌 Overview
This project aims to predict Airbnb listing prices in New York City using machine learning models. The dataset includes features such as location, room type, number of reviews, and availability. The goal is to build a model that can accurately predict the price of a listing based on these features.

##🛠️ Tools and Technologies
Programming Language: Python
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
Models: Linear Regression, Random Forest Regressor
Evaluation Metrics: R² Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)

##🚀 Methodology
1- Data Cleaning:
Removed rows with missing values.
Dropped irrelevant columns (id, name, host_id, host_name, last_review).
Encoded categorical variables (neighbourhood_group, room_type, neighbourhood) using one-hot encoding.

3- Model Training:
Split the data into training and testing sets (80% training, 20% testing).
Standardized the features using StandardScaler.
Trained two models:
Linear Regression
Random Forest Regressor

4- Model Evaluation:
Evaluated the models using R² Score, MAE, and RMSE.

5- Visualizations:
Visualized the result using matplotlib and seaborn.

##📊 Results
The initial results from the models are as follows:

Linear Regression:
R² Score: 0.17
Mean Absolute Error (MAE): 59.56
Root Mean Squared Error (RMSE): 166.96

Random Forest Regressor:
R² Score: 0.13
Mean Absolute Error (MAE): 53.81
Root Mean Squared Error (RMSE): 170.76

##📝 Analysis
The models' performance is currently suboptimal, as indicated by the low R² scores and high MAE/RMSE values. Key observations include: Low R² Scores: Both models explain only a small portion of the variance in the target variable (price).
High MAE and RMSE: The models' predictions are off by an average of 
54–60, with some significant outliers.

## 📂 Project Structure  
📁 ML-Mastery
│── 📁 Notebooks
│ ├── Airbnb_price_prediction.ipynb (Main analysis & visualizations)
│── 📁 Datasets
│ ├── Airbnb_NYC_Cleaned.csv (Processed dataset)
│── 📁 Visualizations
│ ├── Actual vs predicted
│ ├── Feature Engineering
│── README.md (Project documentation)
