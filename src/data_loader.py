# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from config.config import DATA_PATH, TEST_SIZE, RANDOM_STATE

data = pd.read_csv(DATA_PATH)

def check_data():
    data.head()
    data.info()

def load_data():
    X = data[['Age', 'EstimatedSalary']]
    y = data['Purchased']
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
