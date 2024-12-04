import nltk
from nltk import word_tokenize, pos_tag

# run only on the first time
# -------------------------------------------------
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')

text = "I love eating pizza?"
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

print(tagged)