import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Result for model metrics
results = {
    "Model": ["XGBoost", "Logistic Regression (Bayesian)"],
    "Accuracy": [0.65, 0.61],
    "Precision": [0.59, 0.54],
    "Recall": [0.65, 0.62],
    "F1-Score": [0.60, 0.59],
    "NDCG Score": [0.83, 0.82]
}

results_df = pd.DataFrame(results)

# Display Model Metrics
st.write("### Model Performance Metrics")
st.dataframe(results_df)  # Displays the DataFrame in a scrollable format

# Set up the figure and axes
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Create a bar width and x positions for bars
bar_width = 0.2
x = np.arange(len(results_df['Model']))

# Plot Accuracy, Recall, Precision, F1-Score (on the first subplot)
ax[0].bar(x - bar_width*1.5, results_df['Accuracy'], width=bar_width, label="Accuracy")
ax[0].bar(x - bar_width/2, results_df['Recall'], width=bar_width, label="Recall")
ax[0].bar(x + bar_width/2, results_df['Precision'], width=bar_width, label="Precision")
ax[0].bar(x + bar_width*1.5, results_df['F1-Score'], width=bar_width, label="F1-Score")
ax[0].set_xticks(x)
ax[0].set_xticklabels(results_df['Model'])
ax[0].legend()
ax[0].set_title("Accuracy, Recall, Precision, F1-Score Comparison")

# Plot NDCG Score (on the second subplot)
ax[1].bar(results_df['Model'], results_df['NDCG Score'], label="NDCG Score", color='purple')
ax[1].legend()
ax[1].set_title("NDCG Score Comparison")

# Display the plot in Streamlit
st.pyplot(fig)

# Load and display the 'result_Best_Score.csv'
sub_whole_df = pd.read_csv('result_Best_Score.csv')

# Dropdown for test_ids
selected_test_id = st.selectbox("Select test_id", sub_whole_df['id'].unique())
filtered_by_test_id = sub_whole_df[sub_whole_df['id'] == selected_test_id]
st.write(f"### Results for id {selected_test_id}")
st.write(filtered_by_test_id)

# Dropdown for lbl_encoder
selected_lbl_encoder = st.selectbox("Select lbl_encoder", sub_whole_df['country'].unique())
filtered_by_lbl_encoder = sub_whole_df[sub_whole_df['country'] == selected_lbl_encoder]
st.write(f"### test_ids for country {selected_lbl_encoder}")
st.write(filtered_by_lbl_encoder)