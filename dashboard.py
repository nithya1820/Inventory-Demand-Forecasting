import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set up Streamlit app
st.title("Inventory Demand Forecasting Dashboard")

# Data distribution
st.header("Data Distribution Before Preprocessing")
fig, ax = plt.subplots()
sns.boxplot(data=X, ax=ax)
st.pyplot(fig)

st.header("Data Distribution After Preprocessing")
fig, ax = plt.subplots()
sns.boxplot(data=X_scaled, ax=ax)
st.pyplot(fig)

# Model performance comparison
st.header("Model Performance Comparison")
metrics = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R^2'],
    'Raw Data': [mean_absolute_error(y_test, y_pred), 
                 mean_squared_error(y_test, y_pred, squared=False), 
                 r2_score(y_test, y_pred)],
    'Preprocessed Data': [mean_absolute_error(y_test_pp, y_pred_pp), 
                          mean_squared_error(y_test_pp, y_pred_pp, squared=False), 
                          r2_score(y_test_pp, y_pred_pp)],
})
st.dataframe(metrics)

st.bar_chart(metrics.set_index('Metric'))

