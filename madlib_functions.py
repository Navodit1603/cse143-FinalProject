# madlib_functions.py

import nltk as tk
import random as rand
import numpy as np
import pandas as pd
from heapq import nlargest

# Define POS tagging function
def pos_tag(tokens):
    return tk.pos_tag(tokens)

# Define madlib related constants and data
madlib_pos = ['CD', 'JJ', 'NN', 'NNS', 'VB', 'VBG', 'VBD']
pos_labels = {
    'CD': 'Number',
    'JJ': 'Adjective',
    'NN': 'Noun',
    'NNS': 'Plural Noun',
    'VB': 'Verb',
    'VBG': 'Verb ending in ing',
    'VBD': 'Past tense verb',
}
punctuation = ['.', ',', ':', '!', '?']

path = "./data_extracted/sentences/rate_my_professor/rate_my_professor.csv"
data = pd.read_csv(path)
reviews = data['comments']
names = data['professor_name']
schools = data['school_name']
scores = data['star_rating']

# Placeholder sentiment_analysis function
def sentiment_analysis(revs, scores, candidates):
    return {}

# Modify replace_words to interactively replace parts of speech
def replace_words(text, tagged_line, replaceable, sentiment, replacements):
    if len(replaceable) > 0:
        to_replace = []
        for line in tk.sent_tokenize(text):
            line_replace = {}
            for i in range(len(replaceable)):
                if replaceable[i][0] in str(line).lower():
                    line_replace[replaceable[i]] = 1
            
            if len(line_replace) > 0:
                n = int(np.floor(len(line_replace) * 0.4))
                if n == 0: n = 1
                to_replace.append(max(line_replace, key=line_replace.get))

        # Replacing words from user input
        for (og_word, pos) in to_replace:
            word = replacements.get(pos, "")
            for i in range(len(tagged_line)):
                if tagged_line[i] == (og_word, pos):
                    tagged_line[i] = (word.upper(), pos)

    return tagged_line

# Function to select a random review and process the Madlib
def generate_madlib(replacements):
    # Randomly select a review
    text = "No Comments"
    while text == "No Comments":
        rev_idx = np.floor(rand.random() * len(reviews))
        text = reviews[rev_idx]

    professor = names[rev_idx]
    school = schools[rev_idx]
    score = scores[rev_idx]

    tokens = tk.word_tokenize(text)
    tagged_line = pos_tag(tokens)

    # Dynamically generate pos_names from the tagged sentence
    dynamic_pos_names = {}
    for word, tag in tagged_line:
        if tag in madlib_pos and tag not in dynamic_pos_names:
            dynamic_pos_names[tag] = pos_labels.get(tag, tag)

    replaceable = [tagged_line[i] for i in range(len(tagged_line)) if tagged_line[i][1] in madlib_pos]

    sentiment = sentiment_analysis(reviews, scores, [replaceable[i][0] for i in range(len(replaceable))])
    tags = replace_words(text, tagged_line, replaceable, sentiment, replacements)

    output = ""
    prev_token = None
    for i in range(len(tags)):
        if prev_token is not None and tags[i][0] not in punctuation and "'" not in tags[i][0]:
            output += ' '
        output += tags[i][0]
        prev_token = tags[i][0]

    output = f"{professor}, {school}:\n{output}\n{score}/5"

    return dynamic_pos_names, output
