import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for idx in test_set.get_all_Xlengths():
        x, x_lengths = test_set.get_item_Xlengths(idx)

        idx_probs = {}

        for word in models:
            try:
                log_l = models[word].score(x, x_lengths)
                idx_probs[word] = log_l
            except:
                idx_probs[word] = float("-inf")

        best_guess = max(idx_probs, key=idx_probs.get)

        guesses.append(best_guess)
        probabilities.append(idx_probs)

    return probabilities, guesses
