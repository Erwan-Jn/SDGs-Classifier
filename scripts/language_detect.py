import spacy_fastlang
import spacy
from spacy.language import Language
import numpy as np

def language_detect(corpus: "pd.Series"):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("language_detector")

    tmp_ = corpus.map(nlp)

    result = np.zeros((len(tmp_, 2)))

    result[:, 0] = list(map(lambda row : row._.language, tmp_))
    result[:, 1] = list(map(lambda row : row._.language_score, tmp_))

    return result
