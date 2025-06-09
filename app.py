import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("online_shoppers_intention.csv")

df = load_data()

st.title("🛒 Online Shoppers Intention ML App")

# Sidebar - Data Insights without plots
st.sidebar.header("📊 Data Insights")
st.sidebar.write(f"**Number of rows:** {df.shape[0]}")
st.sidebar.write(f"**Number of columns:** {df.shape[1]}")

# Show basic stats for numeric columns
st.sidebar.subheader("Numeric Column Summary")
st.sidebar.write(df.describe())

# Show target variable distribution as a table
st.sidebar.subheader("Revenue Value Counts")
st.sidebar.write(df["Revenue"].value_counts())

# Show raw data option
if st.checkbox("Show raw data"):
    st.write(df.head())

# Preprocessing
df_clean = df.copy()
df_clean = pd.get_dummies(df_clean, drop_first=True)

# Features and target
X = df_clean.drop("Revenue", axis=1)
y = df_clean["Revenue"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
st.write(f"📊 Model Accuracy: **{acc:.2%}**")

# User input for prediction
st.subheader("🔍 Predict Customer Behavior")

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

user_input = {}

with st.form(key='user_input_form'):
    # Numeric inputs
    for col in numeric_cols:
        user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))
    # Categorical inputs
    for col in categorical_cols:
        options = df[col].unique().tolist()
        user_input[col] = st.selectbox(f"{col}", options)

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    user_df = pd.DataFrame([user_input])
    user_df_encoded = pd.get_dummies(user_df, drop_first=True)
    user_df_encoded = user_df_encoded.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(user_df_encoded)[0]
    st.success(f"Prediction: {'🟢 Will Purchase' if prediction else '🔴 Will Not Purchase'}")
