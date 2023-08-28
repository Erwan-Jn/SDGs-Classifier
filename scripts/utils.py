import os
import pandas as pd

class DataProcess():
    """Class to load and preprocess data. Requires a set up with
    parent directory:
            ---raw_data
            ---file being run"""

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.getcwd()), "raw_data", "data.csv")

    def load_data(self):
        df = pd.read_csv(self.path, sep="\t")
        df["sdg"] = df["sdg"].astype(str)
        df["lenght_text"] = df["text"].map(lambda row: len(row.split()))
        df["nb_reviewers"] = df["labels_negative"] + df["labels_positive"]
        return df
