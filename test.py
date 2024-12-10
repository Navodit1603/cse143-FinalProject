
from gensim.models import Word2Vec
from nltk.corpus import brown, movie_reviews, treebank
import nltk.tokenize


WIKIPEDIA_SENTENCES_PATH = './data_extracted/sentences/wikipedia/wikipedia_sentences.txt'
OUTPUT_PATH = './data_extracted/word2vec/wikipedia_embedding.model'


def main():
    print('Starting word2vec...')
    make_word_embedding_for_wikipedia_sentences()
    print('...Done')
    print()

    # word_embedding = Word2Vec.load(OUTPUT_PATH)
    # print(word_embedding.wv.most_similar('the', topn=10))


def make_word_embedding_for_wikipedia_sentences():
    with open(WIKIPEDIA_SENTENCES_PATH, 'r') as in_file:
        lines = []
        line = in_file.readline()
        while line:
            lines.append(nltk.tokenize.word_tokenize(line))
            line = in_file.readline()
        
        print('..done reading lines...')
        
        word_embedding = Word2Vec(sentences=lines, vector_size=100, window=5, min_count=1, workers=4)
        word_embedding.save(OUTPUT_PATH)


if __name__ == '__main__':
    main()
