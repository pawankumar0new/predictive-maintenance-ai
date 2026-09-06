import pandas as pd
import numpy as np

def load_and_preprocess_data(filepath):
    """Load the AI4I dataset and apply preprocessing."""
    df = pd.read_csv(filepath)
    
    # Drop leakage columns (failure mode indicators)
    leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'UDI', 'Product ID']
    df_clean = df.drop(columns=[col for col in leakage_cols if col in df.columns])
    
    # Feature engineering
    df_clean['Temp_Diff'] = df_clean['Process temperature [K]'] - df_clean['Air temperature [K]']
    df_clean['Power'] = df_clean['Torque [Nm]'] * df_clean['Rotational speed [rpm]'] * (2 * np.pi / 60)
    df_clean['Wear_Torque'] = df_clean['Tool wear [min]'] * df_clean['Torque [Nm]']
    
    return df_clean

def get_features_target(df):
    """Extract features and target from processed dataframe."""
    feature_cols = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                    'Temp_Diff', 'Power', 'Wear_Torque']
    
    X = df[feature_cols]
    y = df['Machine failure']
    
    return X, y, feature_cols