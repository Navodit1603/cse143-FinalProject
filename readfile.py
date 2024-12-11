import csv
import numpy as np
import json


def fileToList1(filename):
    fileList = []
    with open(filename, 'r', encoding='cp1252') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:
            appending = []
            appending.append(row[0])
            appending.append(row[1])
            appending.append(row[2])
            fileList.append(appending)
        #print(fileList)
        np_filelist = np.array(fileList)
    return np_filelist

def fileToList2(filename):
    fileList = []
    appending = []
    with open(filename, 'r', encoding='cp1252') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        sentenceNum = 0
        for row in csv_reader:
            if sentenceNum != row[0]:
                fileList.append(appending)
                appending = []
                appending.append(row[2])
                sentenceNum = row[0]
            else:
                appending.append(row[2])
    return fileList

def trainTestSplit(filename):
    train = []
    dev = []
    test = []
    count = -1
    with open(filename, 'r', encoding='cp1252') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            if len(row[0]) > 0:
                count += 1
            if count > 4:
                count = 0
            appending = []
            appending.append(row[0])
            appending.append(row[1])
            appending.append(row[2])
            if count == 1:
                dev.append(appending)
            elif count == 3:
                test.append(appending)
            else:
                train.append(appending)
        print(dev)
    return np.array(train), np.array(dev), np.array(test)

def saveTrainToJson(filename, dictionary):
    with open(filename, "w") as file:
        json.dump(dictionary, file, indent=4)

def openJson(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    
    return data

