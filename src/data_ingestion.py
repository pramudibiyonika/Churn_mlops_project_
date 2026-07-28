import pandas as pd
import os

def load_data(filepath):
    """Load raw data from CSV file"""
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def save_data(df, output_path):
    """Save dataframe to CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    raw_path = "data/raw/telco_customer_churn_data.csv"
    df = load_data(raw_path)
    save_data(df, "data/processed/churn_data.csv")
    print("Data ingestion complete!")