import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    # Load dataset from data folder
    df = pd.read_csv("../data/water_potability.csv")