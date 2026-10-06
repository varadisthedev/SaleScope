<div align="center">

# SaleScope

**Monthly store & product demand forecasting, in one page.**

[![Live app](https://img.shields.io/badge/Live_app-mysalescope.streamlit.app-8A5A3B?style=for-the-badge&logo=streamlit&logoColor=white)](https://mysalescope.streamlit.app)
[![Kaggle dataset](https://img.shields.io/badge/Dataset-Kaggle-4A2C20?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)

<img src="https://skillicons.dev/icons?i=python,sklearn,pandas,numpy,plotly,github,vscode,jupyter&theme=light" alt="Tech stack" />

</div>

<br>

![SaleScope dashboard](repo_assets/image1.png)

![SaleScope demand map and top forecasts](repo_assets/image2.png)

## What it does

Pick a **store**, a **product** and a **month**, and SaleScope predicts the total units that pair will sell that month.

- **Forecast card** with an animated count-up, change vs. the previous month, and a gauge showing where the forecast sits in that pair's historical low-to-peak range.
- **Demand trend** from Jan 2013 up to the month you selected: history as a solid line, forecast as a dashed line.
- **Demand map** heatmap of the top 10 products across all 10 stores for the selected month.
- **Top forecasts** table plus a CSV download of the whole month.
- Forecast any month from **2018-01 to 2027-12**.

> The training data ends in March 2018 and a Random Forest cannot extrapolate trends. Months far past 2018 repeat the seasonal pattern rather than growing, so treat them as rough estimates.

## Model

| | |
|---|---|
| Model | Random Forest regressor (scikit-learn pipeline) |
| Target | Monthly units sold per store-item pair |
| Features | `year`, `month`, `quarter`, `time_index`, `store`, `item` |
| Validation R² | 0.878 |
| RMSE / MAE | 18.32 / 15.25 units |

Training happens in [`notebook/`](notebook); the app only loads the saved pipeline.

## Tech stack

| | Tool | Used for |
|---|---|---|
| <img src="https://skillicons.dev/icons?i=python" width="28"> | Python | Everything |
| <img src="https://skillicons.dev/icons?i=sklearn" width="28"> | scikit-learn 1.6.1 | Model and preprocessing pipeline |
| <img src="https://skillicons.dev/icons?i=pandas" width="28"> | pandas | Data prep and aggregation |
| <img src="https://skillicons.dev/icons?i=numpy" width="28"> | NumPy | Numerics |
| <img src="https://skillicons.dev/icons?i=plotly" width="28"> | Plotly | Trend chart and heatmap |
| <img src="https://skillicons.dev/icons?i=jupyter" width="28"> | Jupyter | Training and evaluation notebook |
| <img src="https://streamlit.io/images/brand/streamlit-mark-color.svg" width="28"> | Streamlit | Web app and hosting |

## Project structure

```text
SaleScope/
├── app.py                 # Streamlit app
├── requirements.txt
├── favicon.png
├── .streamlit/config.toml # light theme
├── data/
│   ├── monthly.csv        # train.csv pre-aggregated by month (used by the app)
│   ├── train.csv          # raw Kaggle data (only needed for retraining)
│   └── test.csv
├── models/
│   └── best_model.joblib
├── notebook/              # training notebook
└── repo_assets/           # README screenshots
```

## Run locally

```bash
git clone https://github.com/varadisthedev/SaleScope.git
cd SaleScope
pip install -r requirements.txt
streamlit run app.py
```

## Deploying on Streamlit Cloud

The model was pickled with **scikit-learn 1.6.1**, which has no wheel for Python 3.14. In **Advanced settings**, choose **Python 3.12** or the build will be very slow or fail. If you retrain with a newer scikit-learn, update the pin in `requirements.txt` to match.

## Data

[Store Item Demand Forecasting](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data): 5 years of daily sales for 50 items across 10 stores, aggregated to monthly totals.
