import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_score = float("inf")
        best_model = None

        for state in range(self.min_n_components, self.max_n_components + 1):

            try:
                hmm_model = GaussianHMM(n_components=state, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                # Suggested P calculation from forum: https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/12
                p = hmm_model.n_components + hmm_model.n_components * (hmm_model.n_components - 1 ) + len(hmm_model.means_) * len(hmm_model.covars_)
                l_log = hmm_model.score(self.X, self.lengths)

                bic_score = -2 * l_log  + p * math.log(len(self.X))

                if bic_score < best_score:
                    best_score = bic_score
                    best_model = hmm_model

            except:
                break

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))))
    '''

    def select(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = []

        # generate models
        for state in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = GaussianHMM(n_components=state, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                
                log_l = hmm_model.score(self.X, self.lengths)
                
                others_l = []

                for word in self.words:
                    if word != self.this_word:
                        try:
                            other_X, other_l = self.hwords[word]
                            other_score = hmm_model.score(other_X, other_l)
                            others_l.append(other_score)
                        except:
                            continue
    
                models.append((hmm_model, log_l, np.mean(others_l)))
            except:
                break
 
        best_model = None
        best_score = float("-inf")

        for model, log_l, others_avg_l in models:
            dic_score = log_l - others_avg_l

            if dic_score > best_score:
                best_score = dic_score
                best_model = model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
 
        models = []

        for state in range(self.min_n_components, self.max_n_components + 1):
            n_splits = 3 if len(self.sequences) > 3 else 2

            split_method = KFold(n_splits=n_splits)
            scores = []

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                seq_X, seq_len = combine_sequences(cv_train_idx, self.sequences)
                test_seq_X, test_seq_len = combine_sequences(cv_test_idx, self.sequences)

                try:
                    hmm_model = GaussianHMM(n_components=state, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(seq_X, seq_len)
                    
                    scores.append(hmm_model.score(test_seq_X, test_seq_len))                    
                except:
                    break


            models.append((hmm_model, np.mean(scores)))


        best_score = float("inf")
        best_model = None

        for model, score in models:
            if score < best_score:
                best_model = model
                best_score = score

        return best_model
