import pandas as pd 
from pathlib import Path

#project root path 
data_path = Path("data/raw")

def load_data():
    print("Loading data...")
    
    train=pd.read_csv(data_path / "train.csv")
    features=pd.read_csv(data_path / "features.csv")
    stores=pd.read_csv(data_path / "stores.csv")
    
    print("Train shape:", train.shape)
    print("Features shape:", features.shape)
    print("Stores shape:", stores.shape)
    
    return train, features, stores

def merge_datasets(train, features, stores):
    print("Merging datasets...")
    
    # Merge train with features on 'Store' and 'Date'
    df =pd.merge(train,features,on=["Store","Date","IsHoliday"],how="left")
    
    # merge stores
    df =pd.merge(df,stores,on="Store",how="left")
    
    print("Merged dataset shape:", df.shape)
    return df
    

if __name__ == "__main__":
    train, features, stores = load_data()
    df = merge_datasets(train, features, stores)
    
    print("\nMerged Sample:")
    print(df.head())
    
    