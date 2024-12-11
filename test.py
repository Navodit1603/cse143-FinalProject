
from gensim.models import Word2Vec
import numpy as np


WORD_EMBEDDING_PATH = './data_extracted/word2vec/wikipedia_embedding.model'


def main():
    print('Loading precomputed word embedding...')
    word_embedding = Word2Vec.load(WORD_EMBEDDING_PATH)
    print('Finished loading precomputed word embedding.')
    print()

    # print(word_embedding.wv.most_similar(positive=['dog', 'cat', 'bird', 'fish', 'wolf', 'chicken', 'cow', 'deer', 'horse', 'pig', 'lion', 'tiger', 'wasp', 'bee', 'snake', 'bear', 'panda', 'sheep', 'goat']))
    # exit()
    
    body_parts = ['arm', 'leg', 'foot', 'hand', 'head', 'body', 'eye', 'ear', 'mouth']
    diff_vec = np.zeros(shape=(word_embedding.vector_size), dtype=float)
    for item1 in body_parts:
        for item2 in body_parts:
            for i in range(0, word_embedding.vector_size):
                diff = abs(word_embedding.wv.get_vector(item1)[i] - word_embedding.wv.get_vector(item2)[i])
                diff_vec[i] += diff
    vec_min = np.min(diff_vec)
    vec_max = np.max(diff_vec)
    # diff_vec = np.array([remap(vec_min, vec_max, 1.0, 100.0, val) for val in diff_vec])
    body_parts_weights = np.array([1.0 / np.exp(remap(vec_min, vec_max, 0.0, 10.0, val)) for val in diff_vec])
    # print(body_parts_weights)
    # exit()

    animals = ['dog', 'cat', 'bird', 'fish', 'wolf', 'chicken', 'cow', 'deer', 'horse', 'pig', 'lion', 'tiger', 'wasp', 'bee', 'snake', 'bear', 'panda', 'sheep', 'goat']
    diff_vec = np.zeros(shape=(word_embedding.vector_size), dtype=float)
    for item1 in animals:
        for item2 in animals:
            for i in range(0, word_embedding.vector_size):
                diff = abs(word_embedding.wv.get_vector(item1)[i] - word_embedding.wv.get_vector(item2)[i])
                diff_vec[i] += diff
    animals_inverted_diff_vec = np.array([1.0/val for val in diff_vec])

    sports = ['basketball', 'volleyball', 'soccer', 'football', 'tennis', 'badminton', 'boxing', 'rugby', 'running', 'swimming']
    diff_vec = np.zeros(shape=(word_embedding.vector_size), dtype=float)
    for item1 in sports:
        for item2 in sports:
            for i in range(0, word_embedding.vector_size):
                diff = abs(word_embedding.wv.get_vector(item1)[i] - word_embedding.wv.get_vector(item2)[i])
                diff_vec[i] += diff
    sports_inverted_diff_vec = np.array([1.0/val for val in diff_vec])

    word_in_question = 'finger'

    body_parts_sims = []
    for word in body_parts:
        vec1 = word_embedding.wv.get_vector(word_in_question)
        vec1 = np.multiply(body_parts_weights, vec1)

        vec2 = word_embedding.wv.get_vector(word)
        vec2 = np.multiply(body_parts_weights, vec2)

        # cos_sim = word_embedding.wv.similarity(word_in_question, word)
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        body_parts_sims.append(cos_sim)
        print(f'{word:<10} ---> {cos_sim: }')
        # print(f'{word} ---> {word_embedding.wv.similarity(word_in_question, word)}')
    print(heuristic_similarity(body_parts_sims))
    print()
    # exit()

    animals_sims = []
    for word in animals:
        cos_sim = word_embedding.wv.similarity(word_in_question, word)
        animals_sims.append(cos_sim)
        print(f'{word:<10} ---> {cos_sim: }')
    print(heuristic_similarity(animals_sims))
    print()

    sports_sims = []
    for word in sports:
        cos_sim = word_embedding.wv.similarity(word_in_question, word)
        sports_sims.append(cos_sim)
        print(f'{word:<10} ---> {cos_sim: }')
    print(heuristic_similarity(sports_sims))
    print()


def heuristic_similarity(sims: list[float]) -> float:
    # return \
    #     0.7 * max(sims) \
    #     + 0.3 * float(np.average(filter_out_outliers(np.array(sims))))
    return float(np.average(np.array(sims)))


def filter_out_outliers(data, m=2.0):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]


def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b

def inv_lerp(a: float, b: float, v: float) -> float:
    return (v - a) / (b - a)

def remap(i_min: float, i_max: float, o_min: float, o_max: float, v: float) -> float:
    return lerp(o_min, o_max, inv_lerp(i_min, i_max, v))


if __name__ == '__main__':
    main()
