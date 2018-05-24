#script number 02
from EndToEndProcess.IngestData.load_data import load_housing_data

if __name__ == "__main__":

    housing = load_housing_data()
    print(housing.head())
