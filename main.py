import csv
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import cross_validate, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors, KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plot

import numpy

numpy.set_printoptions(threshold=numpy.nan)

import sklearn

print('The scikit-learn version is {}.'.format(sklearn.__version__))


def importFile(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        return np.array(list(reader))


def oneHotEncoding(data, indexes: []):
    for index in indexes:
        # Encode string to number (For field name area)
        toEncode = data[:, index]
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(np.array(toEncode))

        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        # data = np.delete(data, index, 1)
        data = np.append(data, onehot_encoded, 1)

    # ***Delete old features***
    data = np.delete(data, indexes, 1)
    # Parse data to float
    data = data.astype(np.float32)

    return data


def normalizeArray(array, indexes):
    for index in indexes:
        scaler = MinMaxScaler()
        data = array[:, index].reshape(-1, 1)
        scaler.fit(data)
        array[:, index] = np.array(scaler.transform(data).reshape(1, -1))

    return array


def startTraining(algorithm, train_x, train_y, test_x):
    split_x_train, split_x_test, split_y_train, split_y_test = train_test_split(train_x, train_y, test_size=0.3,
                                                                                random_state=10)

    algorithm.fit(split_x_train, split_y_train)
    #accuracy = algorithm.score(split_x_test, split_y_test)
    predictions = algorithm.predict(split_x_test)

    accuracy = accuracy_score(split_y_test, predictions)

    precision, recall, fscore, support = precision_recall_fscore_support(split_y_test, predictions,
                                                                         average='macro')
    print(
        "Validation accuracy: %f -- precision: %f -- recall: %f -- fscore: %f" % (accuracy, precision, recall, fscore))

    # Train test data
    algorithm.fit(split_x_test, split_y_test)

    # predict unknown set
    predictions = algorithm.predict(test_x)

    # ShowGraph(algorithm)
    # print(predictions)

    # scoring = ['recall', 'f1', 'accuracy']
    # print(predictions)
    # scores = cross_validate(algorithm, train_x, train_y, scoring=scoring,
    #                        cv=10, return_train_score=True)

    # print("Accuracy: " + str(scores['test_accuracy'].mean()))
    # print("Recall: " + str(scores['test_recall'].mean()))
    # print("F1: " + str(scores['test_f1'].mean()))

    # cross_val_predict(algorithm, test_x)
    # print(predictions)


def ShowGraph(algorithm):
    plot.plot(algorithm.history['loss'])
    plot.plot(algorithm.history['val_loss'])
    plot.title('model loss')
    plot.ylabel('loss')
    plot.xlabel('epoch')
    plot.legend(['train', 'test'], loc='upper left')
    plot.show()


def RunRandomForestClassifier(train_x, train_y, test_x):
    print("Random Forest")

    # constants
    seed = 7

    # algorithm = tree.DecisionTreeClassifier(random_state=seed, criterion="gini", max_depth=2)
    algorithm = RandomForestClassifier(10, criterion="entropy", random_state=seed)

    startTraining(algorithm, train_x, train_y, test_x)


def RunKNeighborsClassifier(train_x, train_y, test_x):
    print("KNeighborsClassifier")
    algorithm = KNeighborsClassifier()

    startTraining(algorithm, train_x, train_y, test_x)


def RunGaussianProcessClassifier(train_x, train_y, test_x):
    print("GaussianProcessClassifier")
    algorithm = GaussianProcessClassifier(1.0 * RBF(1.0))

    startTraining(algorithm, train_x, train_y, test_x)

    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # MLPClassifier(alpha=1),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()]


def RunKMeans(train_x, train_y, test_x):
    print("KMeans")
    algorithm = KMeans(2)

    startTraining(algorithm, train_x, train_y, test_x)

def RunAdaBoostClassifier(train_x, train_y, test_x):
    print("AdaBoostClassifier")
    algorithm = AdaBoostClassifier()

    startTraining(algorithm, train_x, train_y, test_x)


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
    compined_sets = normalizeArray(compined_sets, [0, 1, 4])

    # print(compined_sets[0])
    # split sets
    train_x = compined_sets[0: trainLength]
    test_x = compined_sets[trainLength:]

    print("------------------------------------------------------")
    print("            Classification Algorithms")
    print("------------------------------------------------------")

    RunRandomForestClassifier(train_x, train_y, test_x)
    RunKNeighborsClassifier(train_x, train_y, test_x)
    RunAdaBoostClassifier(train_x, train_y, test_x)
    RunGaussianProcessClassifier(train_x, train_y, test_x)

    print("------------------------------------------------------")
    print("            Clustering algorithm")
    print("------------------------------------------------------")

    RunKMeans(train_x, train_y, test_x)


if __name__ == "__main__":
    main()
