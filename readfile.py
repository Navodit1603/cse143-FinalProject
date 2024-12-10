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

#print(train(fileToList('train.csv')))