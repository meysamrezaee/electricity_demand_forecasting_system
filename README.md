# Electricity Demand Forecasting System

An end-to-end machine learning project for forecasting hourly electricity demand across Ontario using data published by the Independent Electricity System Operator (IESO).

The project builds a complete forecasting pipeline, from data ingestion and preprocessing to model training, evaluation, and benchmarking. Multiple years of historical hourly electricity consumption data are used to compare classical machine learning approaches against deep learning sequence models.


## Project Overview

This project forecasts **hourly electricity demand** using historical consumption patterns.

Two forecasting approaches were implemented and compared:

- **XGBoost Regressor**
- **PyTorch LSTM (Long Short-Term Memory)**

The project investigates:

- Time-series feature engineering
- Lag-based forecasting
- Sequence modeling
- Model performance as historical training data increases
- Benchmarking against naïve forecasting baselines


## Project Structure

```text
electricity_demand_forecasting_system/

├── data/
│   └── raw_consumption_by_fsa/
│       ├── *.zip
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_training_more_data.ipynb
│
├── images/
│   ├── xgboost_actual_vs_predicted.png
│   └── lstm_actual_vs_predicted.png
│
├── models/
│   ├── xgboost.pkl
│   └── lstm.pth
│
└── README.md
```


## Skills Demonstrated

- Time Series Forecasting
- Electricity Demand Forecasting
- Feature Engineering
- Lag-Based Modeling
- Sequence Modeling
- XGBoost
- PyTorch
- LSTM Neural Networks
- Model Benchmarking
- Forecast Evaluation (MAE, RMSE)
- Data Preprocessing
- Pandas
- NumPy
- Scikit-Learn
- Data Visualization


## Dataset

Source:

IESO (Independent Electricity System Operator)

https://reports-public.ieso.ca/public/HourlyConsumptionByFSA/

Dataset:

**Hourly Consumption by Forward Sortation Area (FSA)**

Data from March 2024 through February 2026 was used for model development and evaluation.

- PUB_HourlyConsumptionByFSA_202403_v1.zip
- ...
- PUB_HourlyConsumptionByFSA_202602_v1.zip

The raw data is distributed as monthly ZIP archives. Each archive contains hourly electricity demand records segmented by:

- FSA (Forward Sortation Area)
- Customer Type
- Price Plan

Example fields:

| Column | Description |
|----------|----------|
| FSA | Forward Sortation Area |
| DATE | Observation date |
| HOUR | Hour of day |
| CUSTOMER_TYPE | Residential or Small General Service |
| PRICE_PLAN | TOU, Tiered, ULO, Retailer |
| TOTAL_CONSUMPTION | Hourly electricity consumption |
| PREMISE_COUNT | Number of service premises |


## Data Pipeline

The project loads data directly from ZIP archives without extracting files to disk.

Workflow:

1. Read monthly ZIP files.
2. Load CSV contents into memory.
3. Create hourly timestamps.
4. Aggregate demand across Ontario.
5. Generate forecasting features.
6. Train forecasting models.
7. Evaluate performance on unseen data.

Raw CSV files are never persisted after loading.


## Forecasting Target

The forecasting objective is:

> Predict Ontario-wide hourly electricity consumption one hour ahead.

Target variable:

```text
TOTAL_CONSUMPTION
```

Prediction horizon:

```text
t + 1 hour
```


## Feature Engineering

### Calendar Features

```text
hour
day_of_week
day_of_month
month
```

### Lag Features

```text
lag_1
lag_24
lag_168
```

Where:

- `lag_1` = previous hour
- `lag_24` = same hour previous day
- `lag_168` = same hour previous week

### Rolling Statistics

```text
rolling_mean_24
rolling_mean_168
```


## Models

### Baseline Forecast

Persistence forecast:

```text
Prediction(t) = Consumption(t-1)
```

This provides a benchmark for evaluating machine learning models.


### XGBoost

Model:

```python
XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05
)
```

Advantages:

- Strong performance on tabular data
- Interpretable feature importance
- Fast training


### LSTM

Architecture:

```text
Input Sequence: 24 Hours

LSTM Layer
    hidden_size = 64
    num_layers = 2

Fully Connected Layer

Output:
    Next-Hour Demand Forecast
```

Training:

```text
Loss Function: MSE
Optimizer: Adam
Epochs: 50
Batch Size: 64
```


## Results

### Naïve Baseline (previous hour)

| Training Data | MAE |
|------------|---------:|
| 2 Months | 249,238 |
| 24 Months | 255,917 |

### XGBoost

| Training Data | MAE |
|------------|---------:|
| 2 Months | 103,569 |
| 24 Months | 85,839 |

The additional historical data improved XGBoost performance by approximately 17%.

### LSTM

| Training Data | MAE |
|------------|---------:|
| 24 Months | 76,038 |

The LSTM outperformed XGBoost after tuning sequence length, hidden layer size, batch training, and training duration.


## Feature Importance (XGBoost)

Top predictors:

| Feature | Importance |
|----------|----------:|
| lag_1 | 0.800 |
| lag_24 | 0.171 |
| hour | 0.019 |
| rolling_mean_24 | 0.003 |

Key takeaway:

Hourly electricity demand exhibits strong short-term persistence, making recent consumption history the most influential predictor.


## Forecast Results

### XGBoost 

images/xgboost_actual_vs_predicted.png

Actual versus predicted hourly electricity demand on the test set.


### LSTM 

images/lstm_actual_vs_predicted.png

Actual versus predicted hourly electricity demand using a sequence-based LSTM model.


## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- PyTorch
- Matplotlib
- Jupyter Notebook


## Key Learnings

- Implemented a complete time-series forecasting pipeline.
- Compared traditional machine learning against deep learning approaches.
- Demonstrated improvements from increasing historical training data.
- Learned the importance of lag features and temporal seasonality in electricity forecasting.
- Tuned LSTM architectures using sequence length, hidden state size, mini-batch training, and epoch scheduling.


## Future Improvements

Potential extensions include:

- Weather integration (temperature, wind, humidity)
- Holiday and calendar effects
- Multi-step forecasting (24-hour demand forecasts)
- Hyperparameter optimization
- Transformer-based forecasting models
- Model deployment through a REST API or dashboard


## License

This project is licensed under the MIT License. See the LICENSE file for details.


## Credits

Created by Meysam Rezaee