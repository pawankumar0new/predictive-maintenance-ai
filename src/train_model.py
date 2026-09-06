import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import load_and_preprocess_data, get_features_target

def train_and_save_model(data_path, model_path):
    """Train Random Forest model and save to disk."""
    # Load and preprocess
    df = load_and_preprocess_data(data_path)
    X, y, feature_cols = get_features_target(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        max_depth=12
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("=== Model Evaluation ===")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    joblib.dump(model, model_path)
    joblib.dump(feature_cols, 'models/feature_names.pkl')
    print(f"\nModel saved to: {model_path}")
    
    return model, feature_cols

if __name__ == "__main__":
    train_and_save_model('../data/ai4i2020.csv', '../models/random_forest_model.pkl')