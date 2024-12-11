from readfile import *
import re


def train(fileList, fileList2):
    pos = {}  # {word: {WORD_OCCUR: n, POS1: m1, POS2: m2, ...}}
    transitions = {"SEN_START": {"POS_OCCUR": 0}}  # {prev_pos: {POS_OCCUR: n, next_pos1: m1, next_pos2: m2, ...}}

    prev_pos = "SEN_START"  # Start every sentence with SEN_START
    print(type(fileList))
    for row in fileList:
        if len(row) < 3:  # Skip malformed rows
            continue

        # Extract word and POS
        word = row[1].lower()
        current_pos = row[2]

        # Update pos dictionary
        if word not in pos:
            pos[word] = {"WORD_OCCUR": 1, current_pos: 1}
        else:
            pos[word]["WORD_OCCUR"] += 1
            pos[word][current_pos] = pos[word].get(current_pos, 0) + 1

        # Update transitions dictionary
        if prev_pos not in transitions:
            transitions[prev_pos] = {"POS_OCCUR": 1, current_pos: 1}
        else:
            transitions[prev_pos]["POS_OCCUR"] += 1
            transitions[prev_pos][current_pos] = transitions[prev_pos].get(current_pos, 0) + 1

        # Update prev_pos only if row[0] is empty
        prev_pos = current_pos if row[0] == "" else "SEN_START"

    
    for sen in fileList2:
        prev = "SEN_START"
        for partOfSpeech in sen:
            if prev not in transitions:
                transitions[prev] = {"POS_OCCUR": 1, partOfSpeech: 1}
            else:
                transitions[prev]["POS_OCCUR"] += 1
                transitions[prev][partOfSpeech] = transitions[prev].get(partOfSpeech, 0) + 1
            prev = partOfSpeech
            
    
    
    return pos, transitions



def test(pos, transitions, sentence):
    # Ensure all words in the sentence are lowercase
    sentence = [word.lower() for word in sentence]

    # Viterbi variables
    viterbi = [{}]  # List of dictionaries to store probabilities
    backpointer = [{}]  # List of dictionaries to store backpointers

    # Set a smoothing value for unseen words
    smoothing_value = 1e-6

    # Initialization step
    for current_pos in transitions["SEN_START"]:
        if current_pos == "POS_OCCUR":
            continue
        p_trans = transitions["SEN_START"].get(current_pos, 0) / transitions["SEN_START"]["POS_OCCUR"]
        if sentence[0] in pos:
            p_pos = pos[sentence[0]].get(current_pos, 0) / pos[sentence[0]]["WORD_OCCUR"]
        else:
            # Assign a uniform probability for unseen words
            p_pos = smoothing_value
        viterbi[0][current_pos] = p_trans * p_pos
        backpointer[0][current_pos] = None

    # Recursion step
    for t in range(1, len(sentence)):
        viterbi.append({})
        backpointer.append({})
        word = sentence[t]

        for current_pos in transitions:
            if current_pos == "POS_OCCUR" or current_pos == "SEN_START":
                continue

            max_prob = 0
            best_prev_pos = None
            for prev_pos in viterbi[t - 1]:
                p_trans = transitions[prev_pos].get(current_pos, 0) / transitions[prev_pos]["POS_OCCUR"]

                if word in pos:
                    p_pos = pos[word].get(current_pos, 0) / pos[word]["WORD_OCCUR"]
                else:
                    # Assign a uniform probability for unseen words
                    p_pos = smoothing_value

                prob = viterbi[t - 1][prev_pos] * p_trans * p_pos
                if prob > max_prob:
                    max_prob = prob
                    best_prev_pos = prev_pos

            viterbi[t][current_pos] = max_prob
            backpointer[t][current_pos] = best_prev_pos

    # Termination step and backtracking
    best_path = []
    last_pos = max(viterbi[-1], key=viterbi[-1].get)  # Get the best last POS
    best_path.append(last_pos)

    for t in range(len(sentence) - 1, 0, -1):
        last_pos = backpointer[t][last_pos]
        best_path.insert(0, last_pos)

    return best_path



def predict(partsOfSpeech, transitions, word_arr):
    words = []
    pos = []
    count = 0
    for word in word_arr:
        if len(word[0]) > 0:
            words.append(word[1])
            pos.append(word[2])
        else:
            path = test(partsOfSpeech, transitions, words)
            for i in range(len(path)):
                if pos[i] == str(path[i]):
                    count += 1
            words = [word[1]]
            pos = [word[2]]
    return count / len(word_arr)            


def stringToList(s):
    return re.findall( r'\w+|[^\s\w]+', s)


'''
# Training the data
# ----------------------------------------------------
train_arr = fileToList1('train.csv')
dev_arr = fileToList1('dev.csv')
test_arr = fileToList1('test.csv')
pos_tag_arr = fileToList2('pos_tags.csv')

pos, transitions = train(train_arr, pos_tag_arr)

saveTrainToJson('pos.json', pos)
saveTrainToJson('transitions.json', transitions)
# -----------------------------------------------------
'''

pos = openJson('pos.json')
transitions = openJson('transitions.json')


sentence = "Veenstra is a good professor and fair in grading, assignments are time consuming."

# Printing Results
# ----------------------------------------------------------------
results = test(pos, transitions, stringToList(sentence))
thing = ""
for result in results:
    thing = thing + result + ", "
print(thing)
# ----------------------------------------------------------------

'''
# Get Accuracy Score
# ----------------------------------------------------------------
percent = predict(pos, transitions, test_arr)
print(percent)
# ----------------------------------------------------------------
'''
