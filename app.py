from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Electricity Consumption",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main page */
.main {
    padding-top: 0.5rem;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.18);
}

/* Move complete sidebar content upward */
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0 !important;
    position: relative;
    top: -55px;
}

/* Sidebar title */
section[data-testid="stSidebar"] h2 {
    font-size: 22px;
    font-weight: 750;
    margin-top: 0 !important;
    margin-bottom: 4px;
}

/* Hero */
.hero {
    padding: 25px 28px;
    border-radius: 16px;
    margin-bottom: 22px;
    background: linear-gradient(
        135deg,
        #172033,
        #24324d
    );
    border: 1px solid rgba(255, 255, 255, 0.10);
}

.hero-title {
    font-size: 36px;
    font-weight: 750;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
}

.hero-subtitle {
    font-size: 16px;
    opacity: 0.78;
}

/* Section headings */
.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 8px;
    margin-bottom: 10px;
}

/* Dashboard cards */
.info-card {
    padding: 18px;
    border-radius: 13px;
    border: 1px solid rgba(128, 128, 128, 0.20);
    background: rgba(128, 128, 128, 0.055);
    min-height: 105px;
}

.info-title {
    font-size: 14px;
    opacity: 0.68;
}

.info-value {
    font-size: 24px;
    font-weight: 700;
    margin-top: 7px;
}

/* Prediction result */
.prediction-card {
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    background: linear-gradient(
        135deg,
        #123c2b,
        #176044
    );
    border: 1px solid rgba(255, 255, 255, 0.12);
    margin-top: 20px;
}

.prediction-card h3 {
    margin-bottom: 5px;
    font-size: 17px;
    font-weight: 600;
}

.prediction-card h1 {
    font-size: 42px;
    margin-top: 4px;
    margin-bottom: 0;
    font-weight: 800;
}

/* Buttons */
.stButton > button {
    border-radius: 9px;
    font-weight: 650;
    min-height: 44px;
}

/* Inputs */
div[data-baseweb="input"] {
    border-radius: 8px;
}

div[data-baseweb="select"] > div {
    border-radius: 8px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    padding: 13px 15px;
    border-radius: 11px;
    border: 1px solid rgba(128, 128, 128, 0.18);
    background: rgba(128, 128, 128, 0.045);
}

/* Tables */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Subheaders */
h3 {
    font-weight: 680;
}

/* Horizontal separators */
hr {
    margin-top: 18px;
    margin-bottom: 18px;
}

/* Footer */
.footer {
    text-align: center;
    opacity: 0.5;
    padding-top: 25px;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    project_root = Path(__file__).resolve().parent
    file_path = project_root / "data" / "energydata_complete.csv"

    data = pd.read_csv(file_path)

    data["date"] = pd.to_datetime(data["date"])

    data = data.sort_values("date").reset_index(drop=True)

    # Date features
    data["hour"] = data["date"].dt.hour
    data["day"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    # Previous consumption
    data["previous_consumption"] = data["Appliances"].shift(1)

    data = data.dropna().reset_index(drop=True)

    return data


df = load_data()


# =========================================================
# FEATURE SET
# =========================================================

# Use every prepared numerical feature, matching the final Colab notebook.
# This includes hour, day, month, day_of_week, is_weekend and previous_consumption.
FEATURES = [
    column for column in df.columns
    if column not in ["Appliances", "date"]
]

X = df[FEATURES]
y = df["Appliances"]


# =========================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# =========================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models(X_train, y_train):

    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        # Baseline: predicts the training-set mean for every test record.
        # This helps demonstrate whether ML models add predictive value.
        "Dummy Baseline": DummyRegressor(strategy="mean")
    }

    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


models = train_models(X_train, y_train)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

@st.cache_data
def calculate_results(_models, X_test, y_test):

    results = []

    for name, model in _models.items():

        predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R² Score": r2
        })

    return pd.DataFrame(results)


results_df = calculate_results(
    models,
    X_test,
    y_test
)


best_model_name = results_df.loc[
    results_df["R² Score"].idxmax(),
    "Model"
]

best_model = models[best_model_name]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## ⚡ Smart Energy")

st.sidebar.caption(
    "Machine Learning based electricity consumption analysis"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠  Dashboard",
        "⚡  Prediction",
        "📊  Consumption Analysis",
        "🤖  Model Comparison",
        "🔍  Model Insights"
    ],
    label_visibility="visible"
)

st.sidebar.divider()

st.sidebar.info(
    "Dataset: UCI Appliances Energy Prediction"
)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
⚡ Smart Electricity Consumption Prediction
</div>

<div class="hero-subtitle">
Energy Usage Analysis Using Machine Learning
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD
# =========================================================

# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠  Dashboard":

    st.markdown(
        '<div class="section-title">Project Dashboard</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Overview of the electricity consumption prediction project "
        "and its machine learning workflow."
    )

    # =====================================================
    # KEY METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">Dataset Records</div>
                <div class="info-value">{len(df):,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">Average Consumption</div>
                <div class="info-value">
                    {df["Appliances"].mean():.2f} Wh
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">Maximum Consumption</div>
                <div class="info-value">
                    {df["Appliances"].max():.0f} Wh
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">Best Model</div>
                <div class="info-value">
                    {best_model_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # =====================================================
    # ABOUT THE PROJECT
    # =====================================================

    st.markdown(
        '<div class="section-title">About the Project</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This project develops an intelligent machine learning system "
        "for predicting electricity consumption using historical energy "
        "usage and environmental conditions."
    )

    st.write(
        "The system analyzes factors such as temperature, humidity, "
        "time of day, previous consumption and weather conditions. "
        "Multiple regression models are trained and compared to identify "
        "an effective prediction approach."
    )

    st.divider()

    # =====================================================
    # KEY STATISTICS + PROJECT OBJECTIVE
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📌 Key Statistics")

        st.write(
            f"**Training Records:** {len(X_train):,}"
        )

        st.write(
            f"**Testing Records:** {len(X_test):,}"
        )

        st.write(
            f"**Input Features:** {len(FEATURES)}"
        )

        st.write(
            f"**Average Usage:** {df['Appliances'].mean():.2f} Wh"
        )

    with col2:

        st.subheader("💡 Project Objective")

        st.write("• Identify electricity consumption patterns")
        st.write("• Analyze environmental relationships")
        st.write("• Compare machine learning regression models")
        st.write("• Predict future electricity consumption")
        st.write("• Support efficient energy planning")

    st.divider()

    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    st.subheader("📋 Dataset Preview")

    st.caption(
        "Sample records from the UCI Appliances Energy Prediction dataset."
    )

    st.dataframe(
        df.head(10),
        hide_index=True,
        width="stretch"
    )


# =========================================================
# CONSUMPTION ANALYSIS
# =========================================================

elif page == "📊  Consumption Analysis":

    st.markdown(
        '<div class="section-title">📊 Consumption Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore electricity consumption trends, seasonal patterns, "
        "and the relationship between environmental conditions and energy usage."
    )

    st.divider()

    # =====================================================
    # ROW 1
    # =====================================================

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # HOURLY PATTERN
    # -----------------------------------------------------

    with col1:

        st.subheader("🕐 Hourly Consumption Pattern")

        hourly = df.groupby("hour")["Appliances"].mean()

        fig, ax = plt.subplots(figsize=(6, 3.5))

        ax.plot(
            hourly.index,
            hourly.values,
            marker="o"
        )

        ax.set_title("Average Consumption by Hour")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Consumption (Wh)")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig, width="stretch")

        peak_hour = hourly.idxmax()

        st.info(
            f"Peak average consumption: {peak_hour}:00 hours"
        )


    # -----------------------------------------------------
    # SEASONAL / MONTHLY PATTERN
    # -----------------------------------------------------

    with col2:

        st.subheader("📅 Seasonal / Monthly Pattern")

        monthly = df.groupby("month")["Appliances"].mean()

        fig, ax = plt.subplots(figsize=(6, 3.5))

        ax.plot(
            monthly.index,
            monthly.values,
            marker="o"
        )

        ax.set_title("Average Consumption by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Consumption (Wh)")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig, width="stretch")

        st.info(
            "The dataset contains records from January through May."
        )


    # =====================================================
    # ROW 2
    # =====================================================

    col3, col4 = st.columns(2)

    # -----------------------------------------------------
    # TEMPERATURE
    # -----------------------------------------------------

    with col3:

        st.subheader("🌡️ Temperature vs Consumption")

        fig, ax = plt.subplots(figsize=(6, 3.5))

        ax.scatter(
            df["T_out"],
            df["Appliances"],
            alpha=0.25
        )

        ax.set_title("Temperature vs Electricity Consumption")
        ax.set_xlabel("Outdoor Temperature (°C)")
        ax.set_ylabel("Consumption (Wh)")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig, width="stretch")


    # -----------------------------------------------------
    # HUMIDITY
    # -----------------------------------------------------

    with col4:

        st.subheader("💧 Humidity vs Consumption")

        fig, ax = plt.subplots(figsize=(6, 3.5))

        ax.scatter(
            df["RH_out"],
            df["Appliances"],
            alpha=0.25
        )

        ax.set_title("Humidity vs Electricity Consumption")
        ax.set_xlabel("Outdoor Humidity (%)")
        ax.set_ylabel("Consumption (Wh)")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig, width="stretch")

       

# =========================================================
# PREDICTION
# =========================================================

elif page == "⚡  Prediction":

    st.markdown(
        '<div class="section-title">⚡ Electricity Consumption Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the current energy and environmental conditions "
        "to estimate electricity consumption."
    )

    st.info(
        "Input note: The model was trained with 33 numerical features. "
        "This demo asks for six key values; remaining sensor features are "
        "filled using training-data medians. Therefore, treat this as an "
        "educational estimate, not a live meter reading or a fully observed scenario."
    )

    st.divider()

    # -----------------------------
    # INPUTS
    # -----------------------------

    st.subheader("Enter Conditions")

    col1, col2 = st.columns(2)

    with col1:

        previous_consumption = st.number_input(
            "Previous Electricity Consumption (Wh)",
            min_value=0.0,
            value=100.0,
            step=10.0
        )

        hour = st.slider(
            "Time of Day (Hour)",
            min_value=0,
            max_value=23,
            value=18
        )

        temperature = st.number_input(
            "Temperature (°C)",
            value=20.0,
            step=0.5
        )

    with col2:

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

        day_of_week = st.selectbox(
            "Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )

        month = st.selectbox(
            "Month / Season",
            list(range(1, 13)),
            index=2
        )

    day_number = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }

    selected_day = day_number[day_of_week]

    is_weekend = int(selected_day >= 5)

    st.divider()

    # -----------------------------
    # PREDICT
    # -----------------------------

    if st.button(
        "⚡ Predict Electricity Consumption",
        width="stretch"
    ):

        # Start with historical median values
        input_data = X_train.median().to_frame().T

        # User inputs
        input_data["previous_consumption"] = previous_consumption
        input_data["hour"] = hour
        input_data["day_of_week"] = selected_day
        input_data["month"] = month
        input_data["is_weekend"] = is_weekend
        input_data["T_out"] = temperature
        input_data["RH_out"] = humidity

        # Prediction using selected best model
        prediction = best_model.predict(
            input_data[FEATURES]
        )[0]

        prediction = max(0, prediction)

        st.markdown(
    f"""
    <div class="prediction-card">
        <h3>⚡ Predicted Electricity Consumption</h3>
        <h1>{prediction:.2f} Wh</h1>
    </div>
    """,
    unsafe_allow_html=True
)

        st.success(
            f"Prediction generated using {best_model_name}."
        )

        st.caption(
            f"Selected conditions: {hour:02d}:00, {day_of_week}, "
            f"{temperature:.1f} °C, {humidity:.0f}% humidity, "
            f"previous consumption {previous_consumption:.1f} Wh, month {month}."
        )
        if best_row_score := float(results_df.loc[results_df["Model"] == best_model_name, "R² Score"].iloc[0]):
            if best_row_score < 0:
                st.warning(
                    "The selected model has a negative test R² score. "
                    "This estimate has limited reliability on the held-out period."
                )

# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "🤖  Model Comparison":

    st.markdown(
        '<div class="section-title">🤖 Machine Learning Model Comparison</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Different machine learning regression models are compared "
        "to identify an effective approach for electricity consumption prediction."
    )

    st.divider()

    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.subheader("📊 Model Performance")

    model_columns = st.columns(len(results_df))

    for column, (_, row) in zip(model_columns, results_df.iterrows()):
        with column:
            st.metric(
                row["Model"],
                f"R² {row['R² Score']:.4f}"
            )
            st.write(f"**MAE:** {row['MAE']:.2f} Wh")
            st.write(f"**RMSE:** {row['RMSE']:.2f} Wh")

    st.divider()

    # =====================================================
    # BEST MODEL
    # =====================================================

    st.subheader("🏆 Best Performing Model")

    st.success(
        f"Best Model: {best_model_name}"
    )

    best_row = results_df[
        results_df["Model"] == best_model_name
    ].iloc[0]

    st.write(
        f"The **{best_model_name}** achieved the highest R² score "
        "among the evaluated models and is selected for final prediction."
    )

    st.write(
        f"**R² Score:** {best_row['R² Score']:.4f}  |  "
        f"**MAE:** {best_row['MAE']:.2f} Wh  |  "
        f"**RMSE:** {best_row['RMSE']:.2f} Wh"
    )


    st.divider()

    # =====================================================
    # DETAILED PERFORMANCE TABLE
    # =====================================================

    st.subheader("📋 Detailed Performance Comparison")

    display_results = results_df.copy()

    display_results["MAE"] = (
        display_results["MAE"].round(2)
    )

    display_results["RMSE"] = (
        display_results["RMSE"].round(2)
    )

    display_results["R² Score"] = (
        display_results["R² Score"].round(4)
    )

    st.dataframe(
        display_results,
        hide_index=True,
        width="stretch"
    )


    st.divider()

    # =====================================================
    # R² SCORE GRAPH
    # =====================================================

    st.subheader("📈 R² Score Comparison")

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        results_df["Model"],
        results_df["R² Score"]
    )

    ax.set_title(
        "Model Comparison - R² Score"
    )

    ax.set_xlabel(
        "Machine Learning Model"
    )

    ax.set_ylabel(
        "R² Score"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        width="stretch"
    )

    st.caption(
        "A higher R² score indicates better model performance. The Dummy Baseline predicts the training-set mean and acts as a reference; a useful model should outperform it on the same chronological test set."
    )

    if best_row["R² Score"] < 0:
        st.warning(
            "The best evaluated model has a negative R² score. It performs worse than predicting the test-set mean, so interpret predictions cautiously and review the model/features before deployment."
        )

# =========================================================
# MODEL INSIGHTS
# =========================================================

elif page == "🔍  Model Insights":

    st.markdown(
        '<div class="section-title">🔍 Model Insights</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore the most important factors affecting electricity "
        "consumption and compare actual consumption with model predictions."
    )

    st.divider()

    # =====================================================
    # TOP FEATURES + ACTUAL VS PREDICTED
    # =====================================================

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # TOP FEATURES - LEFT
    # -----------------------------------------------------

    with col1:

        st.subheader("📌 Top Features")

        rf_model = models["Random Forest"]

        importance = pd.Series(
            rf_model.feature_importances_,
            index=FEATURES
        ).sort_values(ascending=False)

        top_features = importance.head(10)

        fig, ax = plt.subplots(
            figsize=(5, 3.5)
        )

        top_features.sort_values().plot(
            kind="barh",
            ax=ax
        )

        ax.set_title(
            "Top 10 Important Features",
            fontsize=11
        )

        ax.set_xlabel("Importance")
        ax.set_ylabel("")

        ax.tick_params(
            axis="both",
            labelsize=8
        )

        ax.grid(
            axis="x",
            alpha=0.3
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            width="stretch"
        )


    # -----------------------------------------------------
    # ACTUAL VS PREDICTED - RIGHT
    # -----------------------------------------------------

    with col2:

        st.subheader("🎯 Actual vs Predicted")

        predictions = best_model.predict(X_test)

        fig, ax = plt.subplots(
            figsize=(5, 3.5)
        )

        ax.scatter(
            y_test,
            predictions,
            alpha=0.30
        )

        min_value = min(
            y_test.min(),
            predictions.min()
        )

        max_value = max(
            y_test.max(),
            predictions.max()
        )

        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--"
        )

        ax.set_title(
            f"Actual vs Predicted - {best_model_name}",
            fontsize=11
        )

        ax.set_xlabel(
            "Actual Consumption (Wh)"
        )

        ax.set_ylabel(
            "Predicted Consumption (Wh)"
        )

        ax.tick_params(
            axis="both",
            labelsize=8
        )

        ax.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            width="stretch"
        )

    # -----------------------------------------------------
    # PREDICTION ERROR SUMMARY
    # -----------------------------------------------------
    st.divider()
    st.subheader("📉 Prediction Error Analysis")
    insight_predictions = best_model.predict(X_test)
    residuals = y_test.to_numpy() - insight_predictions
    error_col1, error_col2, error_col3 = st.columns(3)
    error_col1.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, insight_predictions):.2f} Wh")
    error_col2.metric("Root Mean Squared Error", f"{np.sqrt(mean_squared_error(y_test, insight_predictions)):.2f} Wh")
    error_col3.metric("Mean Error (Bias)", f"{np.mean(residuals):.2f} Wh")
    error_fig, error_ax = plt.subplots(figsize=(8, 3))
    error_ax.hist(residuals, bins=35, edgecolor="white")
    error_ax.axvline(0, linestyle="--", linewidth=1.5)
    error_ax.set_title("Distribution of Prediction Errors (Actual − Predicted)")
    error_ax.set_xlabel("Prediction Error (Wh)")
    error_ax.set_ylabel("Number of Test Records")
    error_ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    st.pyplot(error_fig, width="stretch")
    st.caption("Positive errors mean the model under-predicted actual consumption; negative errors mean it over-predicted.")

    # =====================================================
    # FEATURE IMPORTANCE TABLE
    # =====================================================

    st.divider()

    st.subheader("📋 Feature Importance Details")

    feature_table = top_features.reset_index()

    feature_table.columns = [
        "Feature",
        "Importance"
    ]

    feature_table["Importance"] = (
        feature_table["Importance"].round(4)
    )

    st.table(feature_table)


    # =====================================================
    # HIGH CONSUMPTION ANALYSIS
    # =====================================================

    st.divider()

    st.subheader("⚠️ High Consumption Analysis")

    threshold = df["Appliances"].quantile(0.95)

    high_usage = df[
        df["Appliances"] >= threshold
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "High Consumption Threshold",
            f"{threshold:.0f} Wh"
        )

    with col2:

        st.metric(
            "High Consumption Records",
            f"{len(high_usage):,}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
Smart Energy • UCI Appliances Dataset • Educational ML Demonstration
</div>
""", unsafe_allow_html=True)