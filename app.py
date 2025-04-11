import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Caching data loading and model loading to avoid re-loading each time
@st.cache
def load_data():
    # Simulate loading a large dataset (replace with actual file path)
    return pd.read_csv('large_data.csv')

@st.cache
def load_model():
    # Load pre-trained KMeans model
    return joblib.load('rfm_kmeans_model.pkl')

# Caching data preprocessing
@st.cache
def preprocess_data(df):
    # Example preprocessing: Standardizing the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['Quantity', 'UnitPrice']])
    return df_scaled

# Load data and model
df = load_data()
model = load_model()

# Preprocess the data
df_scaled = preprocess_data(df)

# Apply KMeans clustering only if not already done
if 'Cluster' not in df.columns:
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

# Apply PCA for visualization (reduce to 2D)
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(df_scaled)
df['PCA1'] = rfm_pca[:, 0]
df['PCA2'] = rfm_pca[:, 1]

# Streamlit app interface
st.title('Customer Segmentation Dashboard')

# Dropdown for selecting the number of clusters
num_clusters = st.selectbox("Select number of clusters", [2, 3, 4, 5], index=2)

# Update the model based on the selected number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Display cluster summary
cluster_summary = df.groupby('Cluster').mean().round(2)
st.write("Cluster Summary", cluster_summary)

# Visualize the customer segments
st.subheader('Customer Segments Visualization using PCA')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=70, alpha=0.8)
plt.title('Customer Segments Visualization using PCA', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.tight_layout()
st.pyplot()

# Download CSV button
st.subheader("Download Segmented Data")
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(label="Download CSV", data=csv_data, file_name="segmented_data.csv")

# Displaying a sample of the data
st.subheader("Sample Data")
st.write(df.head())

