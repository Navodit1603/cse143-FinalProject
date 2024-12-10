# Madlib
import nltk as tk
import pandas as pd
import re
import numpy as np
import random as rand
from heapq import nlargest

# define tagger
def pos_tag(tokens):
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
path = "./data_extracted/sentences/rate_my_professor/rate_my_professor.csv"
data = pd.read_csv(path)
reviews = data['comments']
names = data['professor_name']
schools = data['school_name']
scores = data['star_rating']

def split_into_sentences(paragraph):
    # Regular expression pattern
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    sentences = re.split(sentence_endings, paragraph)    
    return sentences

def sentiment_analysis(revs, scores, candidates):
    positivity = {}
    i = 0
    while i < len(revs):
        review = str(revs[i])
        if review == "No Comments":
            i += 1
            continue
        score = round(float(scores[i]), 1)
        tokens = tk.word_tokenize(review)
        for tok in tokens:
            if tok in candidates:
                # record sentiment
                positivity[tok] = positivity.get(tok, 0) + score
        i += 1
    # normalize
    lower_bound = min(positivity.values())
    upper_bound = max(positivity.values())
    bound = upper_bound - lower_bound
    for key in positivity.keys():
        if positivity[key] > 0 and bound > 0:
            positivity[key] -= lower_bound
            positivity[key] /= bound / 2
            positivity[key] -= 1
    return positivity

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

def replace_words(text, tagged_line, replaceable, sentiment):
    # text: the original input string
    # tagged_line: a list of tuples containing (word, pos) for each token in the input
    # replaceable: a list of tuples (word, pos) for each pos in the set of replaceable pos
    # returns: a modified tagged_line with certain words replaced with user input
    if len(replaceable) > 0:
        # create dictionary mapping each word to its tf-idf
        importance = {}
        # importance determined by tf_idf and sentiment (highly positive or highly negative)
        for (word, pos) in replaceable:
            importance[word] = tf_idf(word, tagged_line) * np.abs(sentiment[word])

        to_replace = []
        # compile a list of all replaceable words in each sentence
        # pick approx 40% of words with the highest importance to replace
        for line in split_into_sentences(text):
            line_replace = {}
            for i in range(len(replaceable)):
                if replaceable[i][0] in str(line).lower():
                    line_replace[replaceable[i]] = importance[replaceable[i][0]]
            
            if len(line_replace) > 0:
                # replace approx 40% of replaceable words
                n = int(np.floor(len(line_replace) * 0.4))
                if n == 0: n = 1
                for i in range(n):
                    # rule eliminates replacement of "lot" with a noun, which tends to not make sense
                    if max(line_replace, key=line_replace.get) == "lot":
                        line_replace.pop(to_replace[len(to_replace) - 1])
                    to_replace.append(max(line_replace, key=line_replace.get))
                    line_replace.pop(to_replace[len(to_replace) - 1])

        # take user input
        for (og_word, pos) in to_replace:
            word = input(pos_names[pos] + ": ")
            for i in range(len(tagged_line)):
                if tagged_line[i] == (og_word, pos):
                    tagged_line[i] = (word.upper(), pos)

    return tagged_line

def __main__():
    # pick random review
    text = "No Comments"
    while text == "No Comments":
        rev_idx = np.floor(rand.random() * len(reviews))
        text = reviews[rev_idx]

    professor = names[rev_idx]
    school = schools[rev_idx]
    score = scores[rev_idx]

    output = ""

    tokens = tk.word_tokenize(text)
    tagged_line = pos_tag(tokens)
    replaceable = []
    for i in range(len(tagged_line)):
        if tagged_line[i][1] in madlib_pos:
            replaceable.append(tagged_line[i])

    madlib_candidates = [replaceable[i][0] for i in range(len(replaceable))]
    sentiment = sentiment_analysis(reviews, scores, madlib_candidates)
    tags = replace_words(text, tagged_line, replaceable, sentiment)

    # rebuild text with new tags
    prev_token = None
    for i in range(len(tags)):
        if prev_token is not None and tags[i][0] not in punctuation and "'" not in tags[i][0]:
            output += ' '
        output += tags[i][0]
        prev_token = tags[i][0]

    #print("Original:\n", text)
    #print("Madlib:\n", output)
    print("\n{0}, {1}:\n{2}\n{3}/5".format(professor, school, output, score))

__main__()
