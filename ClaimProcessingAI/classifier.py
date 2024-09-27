import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from config import CLASSES_TO_KEEP
import numpy as np
from scipy.sparse import hstack

def train(sample):
    # Define the list of classes to keep
    classes_to_keep =  CLASSES_TO_KEEP
    
    # Mark other classes as 'others'
    sample['branch_code'] = sample['branch_code'].apply(lambda x: x if x in classes_to_keep else 'others')
    
    # Initialize the MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    
    # Transform the detected_code_list
    X = mlb.fit_transform(sample['detected_code_list'])
    
    # Target variable
    y = sample['branch_code']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Balance the training dataset
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    # Initialize the model
    model = RandomForestClassifier(random_state=42)
    
    # Train the model
    model.fit(X_train_resampled, y_train_resampled)

    return model

import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import pandas as pd


def preprocess_data(df, vectorizer=None, le=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
    if le is None:
        le = LabelEncoder()

    # Encode labels if 'class' column exists
    if 'class' in df.columns:
        df['class'] = le.transform(df['class'])

    # Vectorize text data
    X = vectorizer.transform(df['text'])

    return X, vectorizer, le


def train_lgb_model(X_train, labels_train, classes_weights, num_round=50):
    # Convert data to LightGBM Dataset format
    train_data = lgb.Dataset(X_train, label=labels_train, weight=classes_weights)

    # Set parameters for LightGBM model (example parameters)
    params = {
        'objective': 'multiclass',
        'num_class': len(set(labels_train)),
        'metric': 'multi_logloss',
        'verbosity': 2,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'reg_lambda': 0.4,
        'seed': 42
    }

    # Train the LightGBM model
    model = lgb.train(params, train_data, num_round)

    return model

def predict_branch_code_v2(model, text, vectorizer, onehot_encoder, scaler, le, rest_info):
    X_test_text = vectorizer.transform([text])
    
    # Extract numerical features and standardize
    numerical_features = ['age']
    X_test_numerical = scaler.transform(rest_info[numerical_features])
    
    # Extract categorical features and one-hot encode
    categorical_features = ['hospid', 'gender']
    X_test_categorical = onehot_encoder.transform(rest_info[categorical_features])
    
    # Combine all features
    print(X_test_text.shape, X_test_numerical.shape,X_test_categorical.shape)
    X_test = hstack((X_test_text, X_test_numerical, X_test_categorical))
    
    # Make predictions on the preprocessed data
    predictions = model.predict(X_test)

    # Flatten predictions to get a 1-dimensional array
    predictions_flat = np.argmax(predictions, axis=1)

    # Get raw scores (before applying softmax)
    raw_scores = model.predict(X_test, raw_score=True)
    
    # Convert raw scores to probabilities using softmax
    probabilities = np.exp(raw_scores) / np.sum(np.exp(raw_scores), axis=1, keepdims=True)

    # Get the top two probabilities and their corresponding class indices
    top_indices = np.argsort(probabilities[0])[::-1]  # Sort indices based on probabilities, descending

    # Top two probabilities
    top_probabilities = probabilities[0][top_indices[:2]]
    print("top_probabilities",top_probabilities)
    # Top two labels
    top_labels = le.inverse_transform(top_indices[:2])

    # Return the labels with highest and second highest probability
    return top_labels[0], top_probabilities[0], top_labels[1], top_probabilities[1]


def predict_branch_code(model, text, vectorizer, le):
    # Preprocess the input text data and transform using the vectorizer
    X = vectorizer.transform([text.lower()])

    # Make predictions on the preprocessed data
    predictions = model.predict(X)
    
    # Get raw scores (before applying softmax)
    raw_scores = model.predict(X, raw_score=True)

    # Convert raw scores to probabilities using softmax
    probabilities = np.exp(raw_scores) / np.sum(np.exp(raw_scores), axis=1, keepdims=True)

    # Get the top two probabilities and their corresponding class indices
    top_indices = np.argsort(probabilities[0])[::-1]  # Sort indices based on probabilities, descending

    # Top two probabilities
    top_probabilities = probabilities[0][top_indices[:2]]
    
    # Top two labels
    top_labels = le.inverse_transform(top_indices[:2])

    # Return the labels with highest and second highest probability
    return top_labels[0], top_probabilities[0], top_labels[1], top_probabilities[1]

# def predict(test):
#     y_pred = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)
    
#     # Print classification report
#     test["predictions"] = y_pred
#     test["probabilities"] = y_pred_proba
#     return test
    
    # Function to predict branch code for new data
# def predict_branch_code(new_detected_code_list, model,mlb):
#     new_X = mlb.transform([new_detected_code_list])
#     y_pred = model.predict(new_X)
#     y_pred_proba = model.predict_proba(new_X)
#     return y_pred, y_pred_proba

def evaluate(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    print(report)
    
