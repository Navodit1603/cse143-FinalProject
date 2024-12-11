
from gensim.models import Word2Vec
import numpy as np


WORD_EMBEDDING_PATH = './data_extracted/word2vec/wikipedia_embedding.model'

CATEGORIES = {
    'body_parts':
        ['arm', 'leg', 'foot', 'hand', 'head', 'body', 'eye', 'ear', 'mouth'],
    'animals':
        ['dog', 'cat', 'bird', 'fish', 'wolf', 'chicken', 'cow', 'deer', 'horse', 'pig', 'lion', 'tiger', 'wasp', 'bee', 'snake', 'bear', 'panda', 'sheep', 'goat'],
    'sports':
        ['basketball', 'volleyball', 'soccer', 'football', 'tennis', 'badminton', 'boxing', 'rugby', 'running', 'swimming']
}


def main():
    print('Loading precomputed word embedding...')
    word_embedding = Word2Vec.load(WORD_EMBEDDING_PATH)
    print('Finished loading precomputed word embedding.')
    print()

    category_word_vectors = {}
    for category, words in CATEGORIES.items():
        word_vecs = []

        for word in words:
            word_vecs.append(word_embedding.wv.get_vector(word))

        category_word_vectors[category] = word_vecs

    category_adjuster_weights = {}
    for category, words in CATEGORIES.items():
        word_vecs = category_word_vectors[category]

        adjuster_weights = np.ones(shape=(word_embedding.vector_size), dtype=float)

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

        category_adjuster_weights[category] = adjuster_weights
    
    word_in_question = 'toes'
    print(f'Word: {word_in_question}')
    for category in CATEGORIES.keys():
        cos_sims = []
        for word in CATEGORIES[category]:
            vec1 = word_embedding.wv.get_vector(word_in_question)
            vec1 = np.multiply(category_adjuster_weights[category], vec1)

            vec2 = word_embedding.wv.get_vector(word)

            cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_sims.append(cos_sim)
        cos_sims = np.array(cos_sims)
        cos_sims = filter_out_outliers(cos_sims)
        print(f'  {category} got {np.average(cos_sims)}')


def filter_out_outliers(data: np.ndarray, m=2.0):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.0)
    return data[s < m]


if __name__ == '__main__':
    main()
