import csv
import numpy as np

def readFile():
    pos = {}      #{word: {occurance: 6, pos: 3}}
    transitions = {"SEN_START": {"POS_OCCUR": 0}}  #{pos: {occurance: 6, pos: 3}}

    with open('train.csv', 'r', encoding='cp1252') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:

            if len(row) < 4:  # Skip malformed rows
                continue

            # for POS count for each word
            word = row[1].lower()
            if word not in pos:
                pos[word] = {"WORD_OCCUR": 1}
                pos[word][row[2]] = 1
            else:
                pos[word]["WORD_OCCUR"] += 1
                if row[2] in pos[word]:
                    pos[word][row[2]] += 1
                else:
                    pos[word][row[2]] = 1

            # Next POS for every POS
            if len(row[0]) > 0:
                prev = "SEN_START"
                #transitions["SEN_START"]["POS_OCCUR"] += 1
            
            if prev not in transitions:
                transitions[prev] = {"POS_OCCUR": 1}
                transitions[prev][row[2]] = 1
            else:
                transitions[prev]["POS_OCCUR"] += 1
                if row[2] in transitions[prev]:
                    transitions[prev][row[2]] += 1
                else:
                    transitions[prev][row[2]] = 1
            prev = row[2]
    
    return pos, transitions
