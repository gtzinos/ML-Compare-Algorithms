import csv
import numpy
from preprocessing import *
from algorithms import *

numpy.set_printoptions(threshold=numpy.nan)

def importFile(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        return np.array(list(reader))

def getTrainList():
    # Import set
    trainList = importFile('train.csv')
    # Exclude features like name and id
    trainList = np.delete(trainList, [0, 3], 1)
    # Transform empty strings to 0
    trainList[trainList == ''] = '0'
    return trainList

def getTestList():
    # Import set
    testList = importFile('test.csv')
    # Exclude features like name and id
    testList = np.delete(testList, [0, 2], 1)
    # Exclude first row with features names
    testList = testList[1:]
    # Transform empty strings to 0
    testList[testList == ''] = '0'

    return testList

def main():
    # Train set
    trainList = getTrainList()
    # Test set
    testList = getTestList()

    # Split train data to variables
    features = trainList[0, 1:]
    train_x = trainList[1:, 1:]
    train_y = trainList[1:, 0].astype(np.float32)

    # Merge train and test dataset
    trainLength = len(train_x)
    compined_sets = np.concatenate((train_x, testList))

    # encode strings to numeric
    compined_sets = oneHotEncoding(compined_sets, [1, 5, 7, 8])

    # Normalize data
    compined_sets = normalize_array(compined_sets, [0, 1, 4])

    # print(compined_sets[0])
    # split sets
    train_x = compined_sets[0: trainLength]
    test_x = compined_sets[trainLength:]

    print("------------------------------------------------------")
    print("            Classification Algorithms")
    print("------------------------------------------------------")

    predictions = RunRandomForestClassifier(train_x, train_y, test_x)
    #show_predictions(test_x, predictions)

    predictions = RunKNeighborsClassifier(train_x, train_y, test_x)
    #show_predictions(test_x, predictions)

    predictions = RunAdaBoostClassifier(train_x, train_y, test_x)
    #show_predictions(test_x, predictions)

    predictions = RunSVC(train_x, train_y, test_x)
    #show_predictions(test_x, predictions)

    print("------------------------------------------------------")
    print("            Clustering algorithm")
    print("------------------------------------------------------")

    RunKMeans(train_x, train_y, test_x)
    #show_predictions(test_x, predictions)


if __name__ == "__main__":
    main()