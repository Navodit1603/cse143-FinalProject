
"""
---How To Use---

Import:
 - from category_classifier import CategoryClassifier

Use:
 - cc = CategoryClassifier(quiet=True)
 - cc.classify('dog')

The classify method will return either a string or None if there is no category it fits
"""

from typing import Optional
from gensim.models import Word2Vec
import numpy as np
from unidecode import unidecode


WORD_EMBEDDING_PATH = './data_extracted/word2vec/wikipedia_embedding.model'

CATEGORIES = {
    'Body Part':
        ['arm', 'leg', 'foot', 'hand', 'head', 'body', 'eye', 'ear', 'mouth', 'teeth', 'elbow', 'shoulder'],
    'Animal':
        ['dog', 'cat', 'bird', 'fish', 'wolf', 'chicken', 'cow', 'deer', 'horse', 'pig', 'lion', 'tiger', 'wasp', 'bee', 'snake', 'bear', 'panda', 'sheep', 'goat'],
    'Sport':
        ['basketball', 'volleyball', 'soccer', 'football', 'tennis', 'badminton', 'boxing', 'rugby', 'running', 'swimming'],
    'Language':
        ['spanish', 'russian', 'french', 'portuguese', 'chinese', 'japanese', 'hindi', 'korean', 'vietnamese', 'german']
}

class CategoryClassifier:
    def __init__(self, quiet: bool=False):
        if not quiet:
            print('Loading precomputed word embedding...')

        self._word_embedding = Word2Vec.load(WORD_EMBEDDING_PATH)

        if not quiet:
            print('Finished loading precomputed word embedding.')
        
        self._category_word_vectors = {}
        for category, words in CATEGORIES.items():
            word_vecs = []

            for word in words:
                word_vecs.append(self._word_embedding.wv.get_vector(word))

            self._category_word_vectors[category] = word_vecs

        self._category_adjuster_weights = {}
        for category, words in CATEGORIES.items():
            word_vecs = self._category_word_vectors[category]

            adjuster_weights = np.ones(shape=(self._word_embedding.vector_size), dtype=float)

            for i in range(0, len(adjuster_weights)):
                good_weights = []

                for word1 in word_vecs:
                    for word2 in word_vecs:
                        # TODO: if word1[i] is 0 or even near 0
                        good_weight = word2[i] / word1[i]
                        good_weights.append(good_weight)

                good_weights = np.array(good_weights)
                good_weights = filter_out_outliers(good_weights)

                avg = np.average(good_weights)
                adjuster_weights[i] = avg

            self._category_adjuster_weights[category] = adjuster_weights

    def classify(self, word_in_question: str, cutoff: float=0.4) -> Optional[str]:
        word_in_question = unidecode(word_in_question).lower()

        max_found = 'NoneFound'
        max_found_val = float('-inf')

        for category in CATEGORIES.keys():
            cos_sims = []
            for word in CATEGORIES[category]:
                try:
                    vec1 = self._word_embedding.wv.get_vector(word_in_question)
                except KeyError:
                    return None
                vec1 = np.multiply(self._category_adjuster_weights[category], vec1)

                vec2 = self._word_embedding.wv.get_vector(word)

                cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_sims.append(cos_sim)
            cos_sims = np.array(cos_sims)
            cos_sims = filter_out_outliers(cos_sims)
            found_val = np.average(cos_sims)

            if found_val > max_found_val:
                max_found = category
                max_found_val = found_val
        
        if max_found_val >= cutoff:
            return max_found
        else:
            return None
    

def filter_out_outliers(data: np.ndarray, m=2.0):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.0)
    return data[s < m]
