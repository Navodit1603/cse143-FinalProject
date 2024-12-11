
from nltk.corpus import wordnet
from gensim.models import Word2Vec
import numpy as np
import nltk.tokenize
from unidecode import unidecode


WORD_EMBEDDING_PATH = './data_extracted/word2vec/wikipedia_embedding.model'


def main():
    print('Loading precomputed word embedding...')
    word_embedding = Word2Vec.load(WORD_EMBEDDING_PATH)
    print('Finished loading precomputed word embedding.')
    print()

    print(word_embedding.wv.most_similar(['lake', 'ocean', 'sea', 'river', 'pond', 'pool']))
    exit()

    sentence_str = 'There is a cricket ground, a swimming pool, and a basketball court.'
    sentence = nltk.tokenize.word_tokenize(unidecode(sentence_str).lower())
    
    vecs = []
    for word in sentence:
        if word != 'pool':
            vecs.append(word_embedding.wv.get_vector(word))
    
    scal_prod = np.ones(shape=(word_embedding.vector_size), dtype=float)
    for vec in vecs:
        scal_prod = np.multiply(scal_prod, vec)
    
    print(word_embedding.wv.most_similar(scal_prod, topn=10))

    # meanings = wordnet.synsets('pool', pos=wordnet.NOUN)
    # for meaning in meanings:
    #     print(meaning)
    #     print(f'    {meaning.definition()}')


if __name__ == '__main__':
    main()
