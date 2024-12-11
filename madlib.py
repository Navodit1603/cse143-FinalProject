# Madlib
import nltk as tk
import pandas as pd
import re
import numpy as np
import random as rand
from heapq import nlargest
from category_classifier import CategoryClassifier

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
    # normalize to a number between -1 and 1
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

def idf(t, doc_set):
    # computes inverse document frequency for term t in document set
    t = t.lower()
    count = 1
    for i in range(len(doc_set)):
        if t in str(doc_set[i]).lower():
            count += 1
    return np.log(len(doc_set) / count)

def tf_idf(t, d, doc_set):
    # computes tf-idf for term t in document d
    return tf(t, d) * idf(t, doc_set)

def replace_words(text, tagged_line, replaceable, sentiment, doc_set, cat_classifier: CategoryClassifier):
    # text: the original input string
    # tagged_line: a list of tuples containing (word, pos) for each token in the input
    # replaceable: a list of tuples (word, pos) for each pos in the set of replaceable pos
    # returns: a modified tagged_line with certain words replaced with user input
    if len(replaceable) > 0:
        # create dictionary mapping each word to its tf-idf
        importance = {}
        # importance determined by tf_idf and sentiment (highly positive or highly negative)
        for (word, pos) in replaceable:
            if sentiment:
                importance[word] = tf_idf(word, tagged_line, doc_set) * np.abs(sentiment[word])
            else:
                importance[word] = tf_idf(word, tagged_line, doc_set)
        to_replace = []
        # compile a list of all replaceable words in each sentence
        # pick approx 40% of words with the highest importance to replace
        for line in tk.sent_tokenize(text):
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
            classification = cat_classifier.classify(og_word)
            if classification == None:
                user_input_ask = pos_names[pos]
            else:
                user_input_ask = classification

            word = input(user_input_ask + ": ")
            for i in range(len(tagged_line)):
                if tagged_line[i] == (og_word, pos):
                    tagged_line[i] = (word.upper(), pos)

    return tagged_line

def rebuild_text(tags):
    # Rebuilds the text from tagged words.
    output = ""
    skip_space = False
    for i, (token, _) in enumerate(tags):
        # Check if the next token is part of a contraction
        if token:
            skip_space = token[0] == "'"
        if i > 0 and not skip_space and token not in punctuation:
            output += ' '
        # Remove escape characters from apostrophes
        token = token.replace("\\", "")
        output += token
        
    return output

def __main__():
    cat_classifier = CategoryClassifier(quiet=True)

    # Ask the user for the type of MadLib
    data_type = input("What type of MadLib? (rmp, aesop, roc): ").strip().lower()

    if data_type == "rmp":
        # read rate my professor data
        path = "./data_extracted/sentences/rate_my_professor/rate_my_professor.csv"
        data = pd.read_csv(path)
        reviews = data['comments']
        names = data['professor_name']
        schools = data['school_name']
        scores = data['star_rating']
        # pick random review
        text = "No Comments"
        while text == "No Comments":
            rev_idx = rand.randint(0, len(reviews) - 1)
            text = reviews[rev_idx]

        professor = names[rev_idx]
        school = schools[rev_idx]
        score = scores[rev_idx]

        output = ""

        tokens = tk.word_tokenize(text)
        tagged_line = pos_tag(tokens)

        replaceable = [(word, tag) for word, tag in tagged_line if tag in madlib_pos]
        madlib_candidates = [word for word, _ in replaceable]

        sentiment = sentiment_analysis(reviews, scores, madlib_candidates)
        tags = replace_words(text, tagged_line, replaceable, sentiment, reviews, cat_classifier)

        # rebuild text with new tags
        output = rebuild_text(tags)

        print(f"\n{professor}, {school}:\n{output}\n{score}/5")

    elif data_type in ["aesop", "roc"]:
        # Determine the correct file path
        path = "./data_extracted/sentences/aesop_fables/aesop_fables.txt" if data_type == "aesop" else "./data_extracted/sentences/roc_stories/roc_stories.txt"
        with open(path, 'r') as file:
            stories = [line for line in file.read().splitlines()]
        # Pick a random story
        text = stories[rand.randint(0, len(stories) - 1)]

        if data_type == "aesop":
            # Limit text to a maximum of 5 sentences
            sentences = text.split('. ')
            text = '. '.join(sentences[:5]) + ('.' if len(sentences) > 5 else '')
        
        # Tokenize and tag text
        tokens = tk.word_tokenize(text)
        tagged_line = pos_tag(tokens)

        # Identify replaceable words based on POS tags
        replaceable = [word for word, tag in tagged_line if tag in madlib_pos]
        replaceable = []
        for i in range(len(tagged_line)):
            if tagged_line[i][1] in madlib_pos:
                replaceable.append(tagged_line[i])

        madlib_candidates = [replaceable[i][0] for i in range(len(replaceable))]
        tags = replace_words(text, tagged_line, replaceable, None, stories, cat_classifier)
        # rebuild text with new tags
        output = ""
        prev_token = None
        for i in range(len(tags)):
            if prev_token is not None and tags[i][0] not in punctuation and "'" not in tags[i][0]:
                output += ' '
            output += tags[i][0]
            prev_token = tags[i][0]
        print(output)

__main__()
