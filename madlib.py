import nltk
from nltk import word_tokenize, pos_tag
import re

text = "I love eating pizza?"

def ntlk_pos_tagger(text):
    # run only on the first time
    # -------------------------------------------------
    # nltk.download('punkt_tab')
    # nltk.download('averaged_perceptron_tagger_eng')

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    print(tagged)

def pos_tagger(text):
    tokens = re.findall(r'\w+', text)
    pos_tags = []
    print(tokens)


ntlk_pos_tagger(text)