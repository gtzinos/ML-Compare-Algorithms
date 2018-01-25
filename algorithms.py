from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plot

# constants
CONST_RANDOM_STATE = 1000


def show_predictions(all_unlabeled_data, predictions):
    for index, current_unlabeled in enumerate(all_unlabeled_data):
        print(str(current_unlabeled) + " is " + str(predictions[index]))


def fit_and_predict(algorithm, train_x, train_y, test_x, showPredictions=False):
    split_x_train, split_x_test, split_y_train, split_y_test = train_test_split(train_x, train_y, test_size=0.3,
                                                                                random_state=CONST_RANDOM_STATE)

    algorithm.fit(split_x_train, split_y_train)
    # accuracy = algorithm.score(split_x_test, split_y_test)10
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

    if hasattr(algorithm, 'history'):
        ShowGraph(algorithm)

    if showPredictions:
        show_predictions(test_x, predictions)

    return predictions

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


def RunRandomForestClassifier(train_x, train_y, test_x, showPredictions=False, n_estimators=10, criterion="entropy"):
    print("Random Forest")

    algorithm = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, random_state=CONST_RANDOM_STATE)

    return fit_and_predict(algorithm, train_x, train_y, test_x, showPredictions)


def RunKNeighborsClassifier(train_x, train_y, test_x, showPredictions=False, n_neighbors=2):
    print("KNeighborsClassifier")
    algorithm = KNeighborsClassifier(n_neighbors=n_neighbors)

    return fit_and_predict(algorithm, train_x, train_y, test_x, showPredictions)


def RunSVC(train_x, train_y, test_x, showPredictions=False, c=1):
    print("SVC")
    algorithm = SVC(kernel="linear", C=c, random_state=CONST_RANDOM_STATE)

    return fit_and_predict(algorithm, train_x, train_y, test_x, showPredictions)

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


def RunKMeans(train_x, train_y, test_x, showPredictions=False, n_clusters=2):
    print("KMeans")
    algorithm = KMeans(random_state=CONST_RANDOM_STATE, n_clusters=n_clusters)

    return fit_and_predict(algorithm, train_x, train_y, test_x, showPredictions)


def RunAdaBoostClassifier(train_x, train_y, test_x, showPredictions=False):
    print("AdaBoostClassifier")
    algorithm = AdaBoostClassifier()

    return fit_and_predict(algorithm, train_x, train_y, test_x, showPredictions)
