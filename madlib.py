# Madlib
import nltk as tk
import pandas as pd
import re
import numpy as np
import random as rand

# define tagger

# list of parts of speech (from Penn Treebase tagset) useable for madlib replacement
madlib_pos = ['CD', 'JJ', 'NN', 'NNS', 'VB', 'VBG', 'VBD']
pos_names = {
    'CD': 'Number',
    'JJ': 'Adjective',
    'NN': 'Noun',
    'NNS': 'Plural Noun',
    'VB': 'Verb',
    'VBG': 'Verb ending in ing',
    'VBD': 'Past tense verb',
}

punctuation = ['.', ',', ':', '!', '?'] 

# read rate my professor data
path = "./rmp.csv"
data = pd.read_csv(path)
reviews = data['comments']

def split_into_sentences(paragraph):
    # Regular expression pattern
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    sentences = re.split(sentence_endings, paragraph)    
    return sentences

text = "No Comments"

while text == "No Comments":
    rev_idx = round(rand.random() * len(reviews))
    text = reviews[rev_idx]

output = ""
# take out one word in each sentence
for line in split_into_sentences(text):
    tokens = tk.word_tokenize(line)
    tags = tk.tag_pos(tokens)

    line_tags = []
    for i in range(len(tags)):
        if tags[i][1] in madlib_pos:
            line_tags.append(tags[i])
    
    # replace words
    # pick word in each sentence
    if len(line_tags) > 0:
        remove = rand.choice(line_tags)
        (og_word, pos) = remove
        word = input(pos_names[remove[1]] + ": ")
        for i in range(len(tags)):
            if tags[i] == remove:
                tags[i] = (word.upper(), pos)
    # construct new sentence
    output_sentence = ""
    for i in range(len(tags)):
        if tags[i][0] not in punctuation:
            output_sentence += " "
        output_sentence += (tags[i][0])
    output += output_sentence

print("Madlib: ", output)
