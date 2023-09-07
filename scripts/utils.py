import os
import pandas as pd
import string
import numpy as np

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

from scripts.clean_data import *
from scripts.params import LOCAL_RAW_PATH, LOCAL_DATA_PATH

class DataProcess():
    """
    Define as dp
    Class to load and preprocess data. Requires a set up with
    parent directory:
    ---raw_data
    ---file being run
    """

    def __init__(self):
        '''Class carries 2 information:
        --The path
        --A sdg dict linking numbers to the SDG's themes'''
        self.path = os.path.join(os.path.dirname(os.getcwd()), "raw_data", "data.csv")

        self.sdg = dict(zip( #Dictionnary matching sdg numbers to name
            [str(num) for num in np.arange(1, 17)],
            ["Poverty", "Hunger", "Health", "Education", "Gender",
             "Water", "Clean", "Work", "Inno", "Inequalities", "Cities",
             "Cons&Prod", "Climate", "OceanLife", "LandLife", "Peace"]
        )
                        )

        self.esg = dict(zip(np.arange(1, 17), #Dictionnary matching sdgs (16) to esgs (3)
                            [1, 3, 3, 3, 3, 3, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3]
                            )
                        )

    def load_data(self, abs_path:bool = False)-> pd.DataFrame:
        """
        Takes no argument (path given in __init__)
        Returns a pd.DataFrame
        Adds 4 columns to base data
        Converts 2 columns 2 different types
        """
        if abs_path:
            if len(os.listdir(LOCAL_RAW_PATH))>0:
                full_file_path = os.path.join(LOCAL_RAW_PATH, "data.csv")
                if not os.path.exists(full_file_path):
                    files = [os.path.join(LOCAL_RAW_PATH, file) for file in os.listdir(LOCAL_RAW_PATH) if file.endswith(".csv")]
                    full_file_path = max(files, key=os.path.getctime)

                df = pd.read_csv(full_file_path, sep=",")
            else:
                full_file_path = os.path.join(os.getcwd(), "raw_data", "data.csv")
                df = pd.read_csv(full_file_path, sep="\t")
                full_file_path_new = os.path.join(LOCAL_RAW_PATH, "data.csv")
        else:
            df = pd.read_csv(self.path, sep="\t")

        df["sdg"] = df["sdg"].astype(str)
        df["lenght_text"] = df["text"].map(lambda row: len(row.split()))
        df["nb_reviewers"] = df["labels_negative"] + df["labels_positive"]
        df["nb_reviewers"] = df["nb_reviewers"].astype(int)
        df["agreement_large"] = df["labels_positive"] / df["nb_reviewers"]
        df["sdg_txt"] = df["sdg"].map(self.sdg)

        if len(os.listdir(LOCAL_RAW_PATH))==0:
            df.to_csv(full_file_path_new)


        return df

    def clean_data_short(self, agreement:float=0, grouped:bool=False)-> pd.DataFrame:
        """
        Takes no argument (path given in __init__)
        Returns a pd.DataFrame
        Adds 3 columns of cleaned text + 1 column with length of cleaned text
        """
        translator_p = str.maketrans('', '', string.punctuation)
        translator_d = str.maketrans('', '', string.digits)
        stop_words = set(stopwords.words('english'))

        df = self.load_data()
        df = df.loc[df["agreement"]>=agreement, : ]
        df["cleaned_text"] = clean_vec(df["text"])

        df["cleaned_text"] = df["cleaned_text"].map(lambda row: " ".join([w for w in iter(word_tokenize(row)) if w not in stop_words]))

        if grouped:
            df["sdg"] = df["sdg"].astype(int)
            df["esg"] = df["sdg"].map(self.esg)
            return df

        return df

    def clean_data(self, agreement:float=0, grouped:bool=True, abs_path:bool = False)-> pd.DataFrame:
        """
        Takes 1 parameter, agreement. Agreement will keep all values
        above the agreement threshold for the data. The path is given
        in __init__
        Returns a pd.DataFrame
        Adds 3 columns of cleaned text + 1 column with length of cleaned text
        """
        translator_p = str.maketrans('', '', string.punctuation)
        translator_d = str.maketrans('', '', string.digits)
        stop_words = set(stopwords.words('english'))
        lemmer = WordNetLemmatizer()

        df = self.load_data(abs_path = abs_path)
        df = df.loc[df["agreement"]>=agreement, : ]

        df["cleaned_text"] = clean_vec(df["text"])
        df["lemma"] = preprocess_spacy(df["cleaned_text"].values)
        df["lenght_text_cleaned"] = df["lemma"].map(lambda row: len(row.split()))

        if grouped:
            df["sdg"] = df["sdg"].astype(int)
            df["esg"] = df["sdg"].map(self.esg)
            return df

        return df

def load_processed_data(file_name:str = None)-> pd.DataFrame:

    if file_name==None:
        full_file_path = os.path.join(LOCAL_DATA_PATH, "None")
    else:
        full_file_path = os.path.join(LOCAL_DATA_PATH, file_name)

    if not os.path.exists(full_file_path):
        files = [os.path.join(LOCAL_DATA_PATH, file) for file in os.listdir(LOCAL_DATA_PATH) if file.endswith(".csv")]

        if len(files) == 0:
            print("No processed data, please use preprocess first")
            return None

        print("No corresponding csv file, returning latest saved csv")
        full_file_path = max(files, key=os.path.getctime)

    data_processed = pd.read_csv(full_file_path)

    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None

    return data_processed

def get_top_features(vectoriser, clf, selector = None, top_n: int = 25, how: str = 'long'):
    """
    Convenience function to extract top_n predictor per class from a model.
    """

    assert hasattr(vectoriser, 'get_feature_names_out')
    assert hasattr(clf, 'coef_')
    assert hasattr(selector, 'get_support')
    assert how in {'long', 'wide'}, f'how must be either long or wide not {how}'

    features = vectoriser.get_feature_names_out()
    if selector is not None:
        features = features[selector.get_support()]
    axis_names = [f'feature_{x + 1}' for x in range(top_n)]

    if len(clf.classes_) > 2:
        results = list()
        for c, coefs in zip(clf.classes_, clf.coef_):
            idx = coefs.argsort()[::-1][:top_n]
            results.extend(tuple(zip([c] * top_n, features[idx], coefs[idx])))
    else:
        coefs = clf.coef_.flatten()
        idx = coefs.argsort()[::-1][:top_n]
        results = tuple(zip([clf.classes_[1]] * top_n, features[idx], coefs[idx]))

    df_lambda = pd.DataFrame(results, columns =  ['SDG', 'feature', 'coef'])

    if how == 'wide':
        df_lambda = pd.DataFrame(
            np.array_split(df_lambda['feature'].values, len(df_lambda) / top_n),
            index = clf.classes_ if len(clf.classes_) > 2 else [clf.classes_[1]],
            columns = axis_names
        )

    df_lambda["SDG"] = df_lambda["SDG"].astype(str)
    df_lambda["SDG"] = df_lambda["SDG"].map(DataProcess().sdg)

    return df_lambda

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, figsize = (16, 9)):
    """
    Convenience function to display a confusion matrix in a graph.
    """
    labels = sorted(list(set(y_true)))
    df_lambda = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index = labels,
        columns = labels
    )
    acc = accuracy_score(y_true, y_pred)
    f1s = f1_score(y_true, y_pred, average = 'weighted')

    fig, ax = plt.subplots(figsize = figsize)
    sns.heatmap(
        df_lambda, annot = True, square = True, cbar = False,
        fmt = 'd', linewidths = .5, cmap = 'YlGnBu',
        ax = ax
    )
    ax.set(
        title = f'Accuracy: {acc:.2f}, F1 (weighted): {f1s:.2f}',
        xlabel = 'Predicted',
        ylabel = 'Actual'
    )
    fig.suptitle('Confusion Matrix')
    plt.tight_layout()

def sdg_explainer(df: pd.DataFrame)-> px.bar:
    colors = px.colors.qualitative.Dark24[:20]
    template = 'SDG: %{customdata}<br>Feature: %{y}<br>Coefficient: %{x:.2f}'

    fig = px.bar(
        data_frame = df,
        x = 'coef',
        y = 'feature',
        custom_data = ['SDG'],
        facet_col = 'SDG',
        facet_col_wrap = 3,
        facet_col_spacing = .15,
        height = 1200,
        labels = {
            'coef': 'Coefficient',
            'feature': ''
        },
        title = 'Top 15 Strongest Predictors by SDG'
    )

    fig.for_each_trace(lambda x: x.update(hovertemplate = template))
    fig.for_each_trace(lambda x: x.update(marker_color = colors.pop(0)))
    #fig.for_each_annotation(lambda x: x.update(text = fix_sdg_name(x.text.split("=")[-1])))
    fig.update_yaxes(matches = None, showticklabels = True)
    fig.show()

    return fig
