# Madlib
import nltk as tk
import pandas as pd
import re
import numpy as np
import random as rand
from heapq import nlargest

# define tagger
def tagger(tokens):
    return tk.pos_tag(tokens)

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

# pick random review
text = "No Comments"
while text == "No Comments":
    rev_idx = np.floor(rand.random() * len(reviews))
    text = reviews[rev_idx]

def split_into_sentences(paragraph):
    # Regular expression pattern
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    sentences = re.split(sentence_endings, paragraph)    
    return sentences

def tf(t, d):
    # computes term frequency of term t in document d containing tuples of (term, pos)
    t = t.lower()
    count = 0
    for (word, _) in d:
        if word.lower() == t:
            count += 1
    return count

def idf(t):
    # computes inverse document frequency for term t in document set
    t = t.lower()
    count = 0
    for i in range(len(reviews)):
        if t in str(reviews[i]).lower():
            count += 1
    return np.log(len(reviews) / count)

def tf_idf(t, d):
    # computes tf-idf for term t in document d
    return tf(t, d) * idf(t)

def replace_words(tagged_line, replaceable):
    if len(replaceable) > 0:
        # create dictionary mapping each word to its tf-idf
        importance = {}
        for (word, pos) in replaceable:
            importance[word] = tf_idf(word, tagged_line)

        
        to_replace = []
        # compile a list of all replaceable words in each sentence
        # pick one or two words with the highest importance to replace
        for line in split_into_sentences(text):
            line_replace = {}
            for i in range(len(replaceable)):
                if replaceable[i][0] in str(line).lower():
                    line_replace[replaceable[i]] = importance[replaceable[i][0]]
            
            keys = list(line_replace.keys())
            values = list(line_replace.values())
            sorted_value_index = np.argsort(values)
            sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

            if len(sorted_dict) > 0:
                # replace approx 30% of replaceable words
                n = int(np.floor(len(sorted_dict) * 0.3))
                if n == 0: n = 1
                for i in range(n):
                    to_replace.append(max(sorted_dict, key=sorted_dict.get))
                    sorted_dict.pop(to_replace[len(to_replace) - 1])

        for (og_word, pos) in to_replace:
            word = input(pos_names[pos] + ": ")
            for i in range(len(tagged_line)):
                if tagged_line[i] == (og_word, pos):
                    tagged_line[i] = (word.upper(), pos) 


    return tagged_line



output = ""

# take out one word in each sentence
#for line in split_into_sentences(text):
for i in range(1):
    line = text
    tokens = tk.word_tokenize(line)
    tagged_line = tagger(tokens)

    replaceable = []
    for i in range(len(tagged_line)):
        if tagged_line[i][1] in madlib_pos:
            replaceable.append(tagged_line[i])
    

    tags = replace_words(tagged_line, replaceable)

    # construct new sentence
    output_sentence = ""
    for i in range(len(tags)):
        if tags[i][0] not in punctuation:
            output_sentence += " "
        output_sentence += (tags[i][0])
    output += output_sentence

print("Original:\n", text)

print("Madlib:\n ", output)
