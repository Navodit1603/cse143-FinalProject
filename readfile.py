import csv
import numpy as np

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


fileToList2("pos_tags.csv")