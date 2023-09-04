import spacy_fastlang
import spacy
from spacy.language import Language
import numpy as np
import pandas as pd

def language_detect(corpus: "pd.Series"):
    """Method to detect texts that are not in English in the corpus.
    Takes a panda series of texts and returns a np array with 2 columns: the language and the confidence score"""

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("language_detector")

    tmp_ = corpus.map(nlp)

    result = np.zeros((len(tmp_, 2)))

    result[:, 0] = list(map(lambda row : row._.language, tmp_))
    result[:, 1] = list(map(lambda row : row._.language_score, tmp_))

    pd.DataFrame(result, )

    return result
