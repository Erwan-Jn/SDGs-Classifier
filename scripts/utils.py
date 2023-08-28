import os
import pandas as pd

class DataProcess():

    def load_data():
        path = os.path.join(os.path.dirname(os.getcwd()), "raw_data", "data.csv")
        df = pd.read_csv(path, sep="\t")
        return df
