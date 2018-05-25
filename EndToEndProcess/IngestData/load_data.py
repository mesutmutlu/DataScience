#script number 01
import pandas as pd
import os

def load_housing_data(housing_path="D:/handsonml/dataset/"):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

if __name__ == "__main__":

    print(load_housing_data())

    #now take a quick look to data