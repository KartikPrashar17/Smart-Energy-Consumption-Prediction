# Smart Electricity Consumption Prediction and Energy Usage Analysis Using Machine Learning

A Streamlit-based educational project using the UCI Appliances Energy Prediction dataset.

## Included
- `app.py` — Streamlit dashboard with Dashboard, Prediction, Consumption Analysis, Model Comparison and Model Insights pages.
- `data/energydata_complete.csv` — dataset used by the application.
- `notebook/Smart_Electricity_Consumption_Prediction.ipynb` — project analysis notebook.
- `requirements.txt` — Python dependencies.
- `run_app.bat` — Windows launch helper.

## Run on Windows
1. Extract the ZIP.
2. Open PowerShell in the extracted `Smart_Energy_Final_Project` folder (the folder containing `app.py`).
3. Optional but recommended: create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Start the application:

   ```powershell
   python -m streamlit run app.py
   ```

Or double-click `run_app.bat` after dependencies are installed.

## Methodology
- Records are sorted chronologically.
- Date/time features are derived from the timestamp.
- `previous_consumption` is created by shifting the target by one record; the first row is removed because it has no previous reading.
- The resulting 19,734 records are split chronologically: first 80% for training and final 20% for testing.
- Models compared: Linear Regression, Decision Tree Regressor, Random Forest Regressor, and a Dummy mean baseline.
- Metrics: MAE, RMSE and R².

## Reproducible evaluation from the included CSV and app pipeline
The current local run (scikit-learn 1.8.0) produced:

| Model | MAE (Wh) | RMSE (Wh) | R² |
|---|---:|---:|---:|
| Linear Regression | 28.2647 | 62.0270 | 0.5358 |
| Decision Tree | 116.8077 | 200.5192 | -3.8513 |
| Random Forest | 88.2370 | 122.5808 | -0.8130 |
| Dummy Baseline | 52.6861 | 91.0535 | -0.0003 |

Small differences can occur with different library versions. Generate report values from the same environment used for the final demonstration and retain the exact output.

## Important interpretation / limitations
- The model predicts `Appliances` energy use in Wh for the dataset's 10-minute sampling interval; it is not a whole-house electricity bill or kWh forecast.
- The Prediction page asks the user for six key values. The remaining model features are filled using training-set medians. The UI explicitly discloses this, so predictions are educational estimates rather than fully observed real-time predictions.
- The previous-consumption feature assumes the immediately preceding appliance reading is available. For a real deployment, the live meter/sensor pipeline must provide it.
- The test period is chronological and the reported scores show that Linear Regression performs best among the evaluated models. The negative R² values for tree models mean they perform worse than a constant-mean predictor on this held-out period.
- Random Forest feature importance is an internal model importance measure, not a causal explanation.

## Dataset citation
Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. Energy and Buildings, 140, 81–97. UCI Machine Learning Repository: Appliances Energy Prediction. DOI: 10.24432/C5VC8G.


## Final synchronized evaluation

The Jupyter notebook and Streamlit application use the same chronological 80:20 split and the same four evaluation models. The notebook now includes the Dummy Regressor baseline and residual/error analysis.

Verified on the bundled UCI dataset (19,735 original rows; 19,734 rows after creating the previous-consumption feature; 33 model features):

| Model | MAE (Wh) | RMSE (Wh) | R² |
|---|---:|---:|---:|
| Linear Regression | 28.2647 | 62.0270 | 0.5358 |
| Decision Tree | 116.8077 | 200.5192 | -3.8513 |
| Random Forest | 88.2370 | 122.5808 | -0.8130 |
| Dummy Baseline | 52.6861 | 91.0535 | -0.0003 |

Linear Regression is the best of these evaluated models on this chronological test set. The interactive prediction page asks for a limited set of inputs; remaining features are filled using training-set medians. Predictions should therefore be described as estimates under partially specified conditions, not live measurements.
