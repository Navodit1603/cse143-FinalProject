
from io import TextIOWrapper
from typing import Optional
from gensim.models import Word2Vec
from nltk.corpus import brown, movie_reviews, treebank
import nltk.tokenize


WIKIPEDIA_SENTENCES_PATH = './data_extracted/sentences/wikipedia/wikipedia_sentences.txt'
OUTPUT_PATH = './data_extracted/word2vec/wikipedia_embedding.model'


def main():
    print('Starting word2vec...')
    # make_word_embedding_for_wikipedia_sentences()
    wikipedia_sentences_iterator = WikipediaSentencesIterator()
    with wikipedia_sentences_iterator as wsi:
        # bruh = iter(wsi)
        # print(next(bruh))
        # print(next(bruh))
        # print(next(bruh))
        # print(next(bruh))
        # print(next(bruh))
        word_embedding = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
        word_embedding.build_vocab(iter(wsi))
        word_embedding.save(OUTPUT_PATH)
    print('...Done')
    print()

    word_embedding = Word2Vec.load(OUTPUT_PATH)
    print(word_embedding.wv.most_similar('the', topn=10))


class WikipediaSentencesIterator:
    def __init__(self, max_lines: int = 500):
        self._max_lines: int = max_lines
        self._in_file: Optional[TextIOWrapper] = None
        self._current_line: Optional[str] = ''
        self._current_line_num: int = 0
    
    def __enter__(self):
        self._in_file: Optional[TextIOWrapper] = open(WIKIPEDIA_SENTENCES_PATH, 'r')
        return self
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self._in_file:
            self._in_file.close()
    
    def __iter__(self):
        return self

    def __next__(self):
        if self._current_line_num >= self._max_lines:
            raise StopIteration
        else:
            if self._current_line is None:
                raise StopIteration
            else:
                if not self._in_file:
                    raise Exception('Did not correctly open file for WikipediaSentenceIterator.')
                else:
                    self._current_line = self._in_file.readline()
                    self._current_line_num += 1
                    return nltk.tokenize.word_tokenize(self._current_line)

def make_word_embedding_for_wikipedia_sentences(max_lines=10000):
    with open(WIKIPEDIA_SENTENCES_PATH, 'r') as in_file:
        lines = []
        line = in_file.readline()
        for i in range(0, max_lines):
            if not line:
                break

            lines.append(nltk.tokenize.word_tokenize(line))
            line = in_file.readline()
        
        print('..done reading lines...')
        
        word_embedding = Word2Vec(sentences=lines, vector_size=100, window=5, min_count=1, workers=4)
        word_embedding.save(OUTPUT_PATH)


if __name__ == '__main__':
    main()
