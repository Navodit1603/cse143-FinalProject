
from io import TextIOWrapper
import pathlib
from typing import Optional
from gensim.models import Word2Vec
import nltk.tokenize
from unidecode import unidecode
import datetime


WIKIPEDIA_SENTENCES_PATH = './data_extracted/sentences/wikipedia/wikipedia_sentences.txt'
OUTPUT_PATH = './data_extracted/word2vec/wikipedia_embedding.model'


def main():
    pathlib.Path('./data_extracted/word2vec').mkdir(parents=True, exist_ok=True)

    print('This may take a while...')
    print()

    print('Starting word2vec for Wikipedia Sentences...')
    print(datetime.datetime.now())
    wikipedia_sentences_iterator = WikipediaSentencesIterator(max_lines=7000000)
    with wikipedia_sentences_iterator as wsi:
        word_embedding = Word2Vec(sentences=iter(wsi), vector_size=128, window=15, min_count=3, epochs=5)
        word_embedding.save(OUTPUT_PATH)
    print('Finished word2vec for Wikipedia Sentences.')
    print(datetime.datetime.now())
    print()

    # word_embedding = Word2Vec.load(OUTPUT_PATH)
    # print(word_embedding.wv.most_similar('the', topn=10))


class WikipediaSentencesIterator:
    def __init__(self, max_lines: int = 7871824):
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
        if not self._in_file:
            raise Exception('Did not correctly open file for WikipediaSentenceIterator.')
        else:
            self._in_file.seek(0)
            self._current_line = ''
            self._current_line_num = 0
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
                    self._current_line = unidecode(self._current_line).lower()

                    self._current_line_num += 1

                    return nltk.tokenize.word_tokenize(self._current_line)


if __name__ == '__main__':
    main()
