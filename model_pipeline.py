import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder

def prepare_data(file_path, test_size=0.2, random_state=42, apply_pca=True, n_components=2):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Create new features
    df['Total minutes'] = (df['Total day minutes'] + 
                           df['Total eve minutes'] + 
                           df['Total night minutes'] + 
                           df['Total intl minutes'])

    df['Total charge'] = (df['Total day charge'] + 
                          df['Total eve charge'] + 
                          df['Total night charge'] + 
                          df['Total intl charge'])

    df['Total calls'] = (df['Total day calls'] + 
                         df['Total eve calls'] + 
                         df['Total night calls'] + 
                         df['Total intl calls'])

    # Select relevant features
    selected_features = ['Total minutes', 'Total charge', 'Total calls', 
                         'International plan', 'Customer service calls']

    # Apply Label Encoding to categorical columns
    encoder = LabelEncoder()
    df['International plan'] = encoder.fit_transform(df['International plan'])
    df['Customer service calls'] = encoder.fit_transform(df['Customer service calls'])

    X = df[selected_features]
    y = df['Churn']  # Adjust if needed

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test




def train_model(X_train, Y_train):
    """Train a Gradient Boosting Classifier and return the best model."""

    # Define hyperparameter grid
    param_grid_gbm = {
        'n_estimators': [100],
        'learning_rate': [0.2],
        'max_depth': [8],
        'min_samples_split': [2],
        'min_samples_leaf': [5],
    }

    # Initialize Gradient Boosting model
    gbm = GradientBoostingClassifier(random_state=42)

    # Set up GridSearchCV
    grid_search_gbm = GridSearchCV(
        estimator=gbm, param_grid=param_grid_gbm, cv=5, n_jobs=-1, verbose=1
    )

    # Train the model
    grid_search_gbm.fit(X_train, Y_train)

    # Best model
    best_gbm = grid_search_gbm.best_estimator_

    print(f"Best parameters from Grid Search: {grid_search_gbm.best_params_}")

    return best_gbm  # Return trained model



def evaluate_model(model, X_test, Y_test):
    """Evaluate the trained model on test data and display key metrics."""
    
    # Predictions
    Y_pred = model.predict(X_test)
    Y_probs = model.predict_proba(X_test)[:, 1]  # Get probability for positive class

    # Metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    class_report = classification_report(Y_test, Y_pred)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Not Churn", "Churn"], yticklabels=["Not Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Plot ROC-AUC Curve
    fpr, tpr, _ = roc_curve(Y_test, Y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend(loc="lower right")
    plt.show()

    return accuracy, conf_matrix, class_report, roc_auc




def save_model(model, filename="best_model.pkl"):
    """Save the trained model to a file using joblib."""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename="best_model.pkl"):
    """Load a trained model from a file using joblib."""
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model
