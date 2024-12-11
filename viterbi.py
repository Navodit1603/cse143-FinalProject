from readfile import *
import re


def train(fileList, fileList2, pos_weight=7.0, transition_weight=1.0):
    pos = {}  # {word: {WORD_OCCUR: n, POS1: m1, POS2: m2, ...}}
    transitions = {"SEN_START": {"POS_OCCUR": 0}}  # {prev_pos: {POS_OCCUR: n, next_pos1: m1, next_pos2: m2, ...}}

    prev_pos = "SEN_START"  # Start every sentence with SEN_START

    for row in fileList:
        if len(row) < 3:
            continue

        word = row[1].lower()
        current_pos = row[2]

        if word not in pos:
            pos[word] = {"WORD_OCCUR": pos_weight, current_pos: pos_weight}
        else:
            pos[word]["WORD_OCCUR"] += pos_weight
            pos[word][current_pos] = pos[word].get(current_pos, 0) + pos_weight

        if prev_pos not in transitions:
            transitions[prev_pos] = {"POS_OCCUR": transition_weight, current_pos: transition_weight}
        else:
            transitions[prev_pos]["POS_OCCUR"] += transition_weight
            transitions[prev_pos][current_pos] = transitions[prev_pos].get(current_pos, 0) + transition_weight

        prev_pos = current_pos if row[0] == "" else "SEN_START"

    for sen in fileList2:
        prev = "SEN_START"
        for partOfSpeech in sen:
            if prev not in transitions:
                transitions[prev] = {"POS_OCCUR": transition_weight, partOfSpeech: transition_weight}
            else:
                transitions[prev]["POS_OCCUR"] += transition_weight
                transitions[prev][partOfSpeech] = transitions[prev].get(partOfSpeech, 0) + transition_weight
            prev = partOfSpeech

    return pos, transitions

def test(pos, transitions, sentence, unseen_word_weight=1e-6, transition_weight=1.0):
    sentence = [word.lower() for word in sentence]

    viterbi = [{}]
    backpointer = [{}]

    for current_pos in transitions["SEN_START"]:
        if current_pos == "POS_OCCUR":
            continue
        p_trans = (transitions["SEN_START"].get(current_pos, 0) / transitions["SEN_START"]["POS_OCCUR"]) * transition_weight
        if sentence[0] in pos:
            p_pos = pos[sentence[0]].get(current_pos, 0) / pos[sentence[0]]["WORD_OCCUR"]
        else:
            p_pos = unseen_word_weight
        viterbi[0][current_pos] = p_trans * p_pos
        backpointer[0][current_pos] = None

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
                p_trans = (transitions[prev_pos].get(current_pos, 0) / transitions[prev_pos]["POS_OCCUR"]) * transition_weight

                if word in pos:
                    p_pos = pos[word].get(current_pos, 0) / pos[word]["WORD_OCCUR"]
                else:
                    p_pos = unseen_word_weight

                prob = viterbi[t - 1][prev_pos] * p_trans * p_pos
                if prob > max_prob:
                    max_prob = prob
                    best_prev_pos = prev_pos

            viterbi[t][current_pos] = max_prob
            backpointer[t][current_pos] = best_prev_pos

    best_path = []
    try:
        last_pos = max(viterbi[-1], key=viterbi[-1].get)
    except ValueError:
        return ["UNKNOWN"] * len(sentence)

    best_path.append(last_pos)

    for t in range(len(sentence) - 1, 0, -1):
        if last_pos not in backpointer[t]:
            best_path.insert(0, "UNKNOWN")
        else:
            last_pos = backpointer[t][last_pos]
            best_path.insert(0, last_pos)

    return best_path

def predict(partsOfSpeech, transitions, word_arr):
    words = []
    correct_pos = []
    point = 0
    for word in word_arr:
        if len(word[0]) > 0:
            print(word[0])
            if len(words) != 0:
                predicted_tags = test(partsOfSpeech, transitions, words)
                for i in range(len(predicted_tags)):
                    #print("predicted: ", predicted_tags)
                    #print("correct: ", correct_pos)
                    if str(predicted_tags[i]) == correct_pos[i]:
                        point += 1

            words = []
            words.append(word[1])
            correct_pos = []
            correct_pos.append(word[2])
        else:
            words.append(word[1])
            correct_pos.append(word[2])
    
    return point / len(word_arr)

def stringToList(s):
    return re.findall( r'\w+|[^\s\w]+', s)

def sampleDevSet(dev_arr, fraction=0.1, random_seed=42):
    np.random.seed(random_seed)
    sample_size = int(len(dev_arr) * fraction)
    sampled_indices = np.random.choice(len(dev_arr), sample_size, replace=False)
    return dev_arr[sampled_indices]

def callTrain():
    train_arr, dev_arr, test_arr = trainTestSplit("ner_dataset.csv")
    pos_tag_arr = fileToList2('pos_tags.csv')
    pos, transitions = train(train_arr, pos_tag_arr)
    saveTrainToJson('pos.json', pos)
    saveTrainToJson('transitions.json', transitions)
    return train_arr, dev_arr, test_arr

def runSingleSentence(sentence):
    pos = openJson('pos.json')
    transitions = openJson('transitions.json')
    results = test(pos, transitions, stringToList(sentence))
    pos_results = ""
    for result in results:
        pos_results = pos_results + result + ", "
    print(pos_results)

def runSet(set_num = 1):
    pos = openJson('pos.json')
    transitions = openJson('transitions.json')
    train_arr, dev_arr, test_arr = trainTestSplit("ner_dataset.csv")
    if set_num == 0:
        set = train_arr
    elif set_num == 1:
        set = dev_arr
    else:
        set = test_arr
    # sampled_dev_arr = sampleDevSet(set, fraction=0.3) # Run if only want fraction of a set.
    percent = predict(pos, transitions, set)
    print(f"Accuracy on sampled dev set: {percent * 100:.2f}%")



# runSingleSentence(sentence)
# runSet(set_num)
    # set_num = 0 -> train
    # set_num = 1 -> dev
    # set_num: everything else -> test
# callTrain()
    # to change training data json
    # call if training function or training data is changed.


runSingleSentence("Professor Veenstra is a good professor and fair in grading, assignments are time consuming.")


