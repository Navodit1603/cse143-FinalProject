
from gensim.models import Word2Vec


WORD_EMBEDDING_PATH = './data_extracted/word2vec/wikipedia_embedding.model'


def main():
    print('Loading precomputed word embedding...')
    word_embedding = Word2Vec.load(WORD_EMBEDDING_PATH)
    print('Finished loading precomputed word embedding.')
    print()
    
    body_parts = ['arm', 'leg', 'foot', 'hand', 'head', 'body', 'torso', 'eye', 'ear', 'mouth']
    animals = ['dog', 'cat', 'bird', 'mouse', 'wolf', 'chicken', 'cow', 'deer', 'horse', 'pig']
    sports = ['basketball', 'volleyball', 'soccer', 'football', 'tennis', 'badminton', 'boxing', 'rugby', 'running', 'swimming']
    
    word_in_question = 'golf'

    body_part_sim = 0.0
    for word in body_parts:
        body_part_sim += word_embedding.wv.similarity(word_in_question, word)
        # body_part_sim += word_embedding.wv.relative_cosine_similarity(word, word_in_question, topn=30)
        print(f'{word} ---> {word_embedding.wv.similarity(word_in_question, word)}')
        # print(f'{word} ---> {word_embedding.wv.relative_cosine_similarity(word, word_in_question, topn=30)}')
    body_part_sim /= len(body_parts)
    print(body_part_sim)
    print()

    animals_sim = 0.0
    for word in animals:
        animals_sim += word_embedding.wv.similarity(word_in_question, word)
        print(f'{word} ---> {word_embedding.wv.similarity(word_in_question, word)}')
    animals_sim /= len(animals)
    print(animals_sim)
    print()

    sports_sim = 0.0
    for word in sports:
        sports_sim += word_embedding.wv.similarity(word_in_question, word)
        print(f'{word} ---> {word_embedding.wv.similarity(word_in_question, word)}')
    sports_sim /= len(sports)
    print(sports_sim)
    print()

    # print(word_embedding.wv.distance())
    # print(word_embedding.wv.cosine_similarities('finger', body_parts))

if __name__ == '__main__':
    main()
