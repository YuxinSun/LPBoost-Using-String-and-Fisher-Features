__author__ = 'yuxinsun'

import itertools
from collections import OrderedDict
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator

# amino acid alphabet
alphabet = 'ACDEFGHIKLMNPQRSTVWY'


def create_word_list(alphabet=alphabet, p=3):
    """
    Create a list of all possible substrings of length p given an alphabet
    Parameters
    -------
    :param alphabet: string
        alphabet containing all relevant symbols
    :param p: int
        length of substrings to be generated
    Returns
    -------
    :return: list
        all possible substrings of length p belonging to given alphabet
    """
    word_list = list(itertools.product(alphabet, repeat=p))
    for i in range(0,len(word_list)):
        word_list[i] = ''.join((itertools.chain(word_list[i])))

    return word_list


class feature_generation(BaseEstimator):
    """
    This class allows for generating numeric feature spaces given text files of sequences.
    It only processes files with following format:
    "
    ['CASSLETGGYEQYFGPG CASSRQNSDYTFGSG ...', 'CASSLHLSYEQYFGPG ...', ...]
    "
    where each sequence is separated by a space.
    Parameters
    -------
    alphabet: string, optional
        Alphabet to be used, default amino acid abbreviations
    p: int, optional
        Length of substrings
    feature_type: string, optional
        Type of features, must be either 'string' or 'fisher.
    n_transition: int, optional
        Number of transitions.
    proba_threshold: float, optional
        Threshold of transition probabilities, all transitions with probability that is lower than the threshold will be
        discarded
    verbose: int, optional
        Enable verbose output. If greater than 0 then it prints current processing step.
    Attributes
    -------
    log_proba_: array_like, shape (n_edges, )
        logarithm of 1/transition probability, only used when feature_type is 'fisher'
    edges_: list
        list of all edges, only used when feature_type is 'fisher'
    """
    def __init__(self, alphabet=alphabet, p=3, feature_type='string', n_transition=1, proba_threshold=0., verbose=1):
        self.p = p
        self.alpahbet = alphabet
        self.feature_type = feature_type
        self.n_transition = n_transition
        self.proba_threshold = 0.
        self.verbose = verbose

        if feature_type not in ['string', 'fisher']:
            raise ValueError('Invalid value for feature type: %s.' % feature_type)

    def _process_string(self, data):
        """
        Compute unnormalised string features from sequence files.
        Parameters
        -------
        :param data: list, length n_samples
            List of sequences. Each component is the pool of sequences of a sample, where sequences are separated
             by spaces.
        Returns
        -------
        :return: array_like, shape (n_samples, n_features)
            Data matrix of string features.
        """
        words_ = create_word_list(self.alpahbet, self.p)
        p = self.p
        kern = []

        for i, data_item in enumerate(data):
            if self.verbose > 0:
                print('Processing %d-th data item.' % i)
            dic = dict.fromkeys(words_, 0)
            dic = OrderedDict(sorted(dic.items(), key=lambda t: t[0]))
            for j in range(len(data_item)-p+1):
                if data_item[j:j+p] in dic:
                    dic[data_item[j:j+p]] += 1
            kern.append(dic.values())
        from scipy.io import savemat
        savemat('toy', {'kern': np.asarray(kern).astype(float)}, oned_as='row')
        return np.asarray(kern).astype(float)

    def _normalise_string(self, kern):
        """
        Normalise string features (l2 normalisation).
        Parameters
        -------
        :param kern: array_like, shape (n_samples, n_features)
            Data matrix of string features
        :return: array_like, shape (n_samples, n_features)
            Normalised features
        """
        return normalize(kern, norm='l2')

    def process(self, data):
        if self.feature_type == 'string':
            kern = self._process_string(data)
            return self._normalise_string(kern)
        elif self.feature_type == 'fisher':
            print('Processing Fisher features has not been implemented yet.')
            exit()

    def _proba(self, G):
        for node in G.nodes():
            s = (np.sum(G[node][x]['kern_unnorm_']) for x in G.successors(node))
            s = sum(s)
            for successor_ in G.successors(node):
                if s == 0:
                    G[node][successor_]['proba_'] = 0.
                else:
                    G[node][successor_]['proba_'] = np.sum(G[node][successor_]['kern_unnorm_'])/s
                if G[node][successor_]['proba_'] < self.proba_threshold:
                    G.remove_edge(node, successor_)

        isolated_ = nx.isolates(G)
        G.remove_nodes_from(isolated_)

        return G

    def _log_proba(self, G):
        proba_ = nx.get_edge_attributes(G, 'proba_')
        proba_ = OrderedDict(sorted(proba_.items(), key=lambda t: t[0]))
        proba_ = 1/np.asarray(proba_.values(), dtype=float)
        proba_[np.where(proba_ == 0.)] = np.inf
        log_proba_ = np.log(1/proba_)  # to be checked: isnan?

        self.log_proba_ = log_proba_

    def _normlise_DAG(self, G):
        kern_ = nx.get_edge_attributes(G, 'kern_unnorm_')
        kern_ = OrderedDict(sorted(kern_.items(), key=lambda t: t[0]))
        val_ = np.asarray(kern_.values(), dtype=float)
        key_ = kern_.keys()

        if len(val_.shape) == 2:
            kern_ = normalize(val_ * self.log_proba_[:, None], norm='l2', axis=0)
        else:
            kern_ = (val_ * self.log_proba_)/np.linalg.norm(kern_)

        kern_ = dict(zip(key_, kern_))
        nx.set_edge_attributes(G, 'kern_', kern_)

        return G
