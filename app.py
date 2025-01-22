import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import callbacks
from sklearn.metrics import confusion_matrix, classification_report

# Function to preprocess data
def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['year'] = data.Date.dt.year
    data['month'] = data.Date.dt.month
    data['day'] = data.Date.dt.day

    # Encode cyclic features
    def encode(data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
        return data

    data = encode(data, 'month', 12)
    data = encode(data, 'day', 31)

    # Fill missing values
    object_cols = data.select_dtypes(include='object').columns
    for col in object_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    num_cols = data.select_dtypes(include='float64').columns
    for col in num_cols:
        data[col].fillna(data[col].median(), inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for col in object_cols:
        data[col] = label_encoder.fit_transform(data[col])

    return data

# Streamlit app
st.title('Rain Prediction using ANN')

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file):", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Preprocess data
    data = preprocess_data(data)
    st.write("Data after preprocessing:")
    st.dataframe(data.head())

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corrmat = data.select_dtypes(include=['number']).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corrmat, cmap='coolwarm', annot=True, fmt='.2f')
    st.pyplot(plt)

    # Prepare features and target
    target_col = 'RainTomorrow'
    features = data.drop(columns=[target_col, 'Date', 'day', 'month'])
    target = data[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build ANN model
    model = Sequential([
        Dense(32, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dropout(0.25),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.00009), loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping
    early_stopping = callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # Train the model
    st.subheader("Model Training")
    epochs = st.slider("Select the number of epochs:", min_value=5, max_value=100, value=10, step=5)
    if st.button("Train Model"):
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=32, callbacks=[early_stopping])

        # Plot training history
        st.write("Training and Validation Loss")
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        st.pyplot(plt)

        st.write("Training and Validation Accuracy")
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        st.pyplot(plt)

    # Model evaluation
    st.subheader("Model Evaluation")
    if st.button("Evaluate Model"):
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)

        # Confusion matrix
        st.write("Confusion Matrix")
        cf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(cf_matrix, annot=True, cmap='coolwarm', fmt='d', cbar=False)
        st.pyplot(plt)

        # Classification report
        st.write("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
