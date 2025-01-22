import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from tqdm import tqdm

# Import your existing functions here
from federated_learning import (
    load_and_preprocess_data,
    initialize_model,
    train_local_model_with_dp,
    aggregate_models,
    evaluate_model,
    federated_learning_with_dp
)


# Streamlit App Configuration
st.title("Federated Learning with Differential Privacy")
st.sidebar.header("Configuration")

# Step 1: Upload Data Files
st.subheader("1. Upload Data Files")
uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)
if uploaded_files:
    st.write("Uploaded files:")
    for file in uploaded_files:
        st.write(file.name)

# Step 2: Select Target Column
st.subheader("2. Select Target Column")
target_column = st.text_input("Enter Target Column Name", "")

# Step 3: Set Epsilon for Differential Privacy
st.subheader("3. Set Epsilon for Differential Privacy")
epsilon = st.slider("Epsilon Value", min_value=0.1, max_value=10.0, step=0.1, value=1.0)

# Step 4: Number of Epochs
st.subheader("4. Set Number of Epochs")
num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=100, step=1, value=20)

# Step 5: Run Federated Learning
st.subheader("5. Run Federated Learning")
if st.button("Run"):
    if uploaded_files and target_column:
        # Process uploaded files
        data_files = []
        for uploaded_file in uploaded_files:
            data = pd.read_csv(uploaded_file)
            file_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
            data_files.append(file_data)

        # Run federated learning
        st.write("Running Federated Learning...")
        federated_learning_with_dp(data_files, target_column, epsilon=epsilon, NUM_EPOCHS=num_epochs)
        st.success("Federated Learning Completed!")
    else:
        st.warning("Please upload data files and specify the target column.")

# Step 6: Visualization
st.subheader("6. Visualize Results")
st.write("Confusion Matrix and Accuracy will be displayed here after running the model.")
