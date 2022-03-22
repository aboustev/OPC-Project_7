"""
The purpose of this python file is to regroup every functions from the external kernel
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import MinMaxScaler
import time
import gc
import matplotlib.pyplot as plt
import seaborn as sns


class TrainTestGrid:
    """
    gridsearch the best parameters and display the results of the method chosen
    """
    def __init__(self, x, y, method='knn', n_esti=None):
        # Classification parameters and train test sets
        self.params = None
        self.mthd = None
        self.cls = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Classification results
        self.fbeta_score_ = None

        # input to preprocess
        self.x = None

        self.preprocess_x(x)
        self.y = y
        self.method = method
        self.n_esti = n_esti

        self.set_params()
        self.tt_split()
        self.grid()

        if self.n_esti:
            self.bagged_fit_and_pred()
        else:
            self.fit_and_pred()

        return

    def preprocess_x(self, x):
        """
        preprocess data before splitting
        """
        scaler = MinMaxScaler()
        self.x = scaler.fit_transform(x)

    def set_params(self):
        """
        set parameters of the class
        :return:
        """
        if self.method == 'knn':
            self.knn_params()
        elif self.method == 'svc':
            self.svc_params()
        elif self.method == 'sgd':
            self.sgd_params()
        elif self.method == 'rfc':
            self.rfc_params()
        elif self.method == 'dummy':
            self.cls = DummyClassifier(random_state=0)
        else:
            print('Wrong method input, please make sure you entered either ["knn", "svc", "sgd", "rfc", "dummy"]')
            return

    def knn_params(self):
        """
        knn parameters and method
        """
        params = {
            'n_neighbors': list(range(10, 30)),
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        self.params = params
        self.mthd = KNeighborsClassifier()
        del params
        gc.collect()

    def svc_params(self):
        """
        svc parameters and method
        """
        params = {
            'penalty': ['l1', 'l2'],
            'multi_class': ['crammer_singer', 'ovr']
        }
        self.params = params
        self.mthd = LinearSVC(random_state=0, max_iter=2000, tol=0.0001)
        del params
        gc.collect()

    def sgd_params(self):
        """
        sgq parameters and method
        """
        params = {
            'penalty': ['l1', 'l2'],
            'tol': [0.0001, 0.001, 0.01, 0.1]
        }
        self.params = params
        self.mthd = SGDClassifier(random_state=0)
        del params
        gc.collect()

    def rfc_params(self):
        """
        rfc parameters and method
        """
        params = {
            'criterion': ['gini', 'entropy']
        }
        self.params = params
        self.mthd = RandomForestClassifier(random_state=0)
        del params
        gc.collect()

    def tt_split(self):
        """
        train test split
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=0.33,
            random_state=0)

    def grid(self):
        """
        gridsearch on the params
        """
        if self.mthd is not None:  # Check if mthd exists to not use gridsearch on dummy
            grid = GridSearchCV(self.mthd, param_grid=self.params)
            grid.fit(self.X_train, self.y_train)
            print('Grid best score :', grid.score(self.X_test, self.y_test))
            print('Grid best params :', grid.best_params_)

            self.cls = self.mthd.set_params(**grid.best_params_)
            del grid
            gc.collect()

    def fit_and_pred(self):
        """
        fit method, predict results and display confusion matrix
        """
        start_train_time = time.time()
        self.cls.fit(self.X_train, self.y_train)
        train_time = time.time() - start_train_time
        print('{} train time : {}'.format(self.method.upper(), train_time))

        pred = self.cls.predict(self.X_test)
        true = self.y_test.values
        self.fbeta_score_ = fbeta_score(true, pred, beta=0.5)
        plt.figure(figsize=(10, 5))
        sns.heatmap(confusion_matrix(true, pred), annot=True, cmap="Blues")
        plt.title('{} confusion matrix \n score = {} \n fbeta score = {}'.format(
            self.method.upper(),
            self.cls.score(self.X_test, self.y_test),
            self.fbeta_score_)
        )
        plt.show()
        del train_time, pred, true
        gc.collect()

    def bagged_fit_and_pred(self):
        """
        makes bags, fit method, predict results and display confusion matrix
        """
        print('Bagging with {} estimators'.format(self.n_esti))
        bagged_cls = BaggingClassifier(self.cls, n_estimators=self.n_esti)
        start_train_time = time.time()
        bagged_cls.fit(self.X_train, self.y_train)
        train_time = time.time() - start_train_time
        print('Bagged {} train time : {}'.format(self.method.upper(), train_time))

        start_train_time = time.time()
        self.cls.fit(self.X_train, self.y_train)
        train_time = time.time() - start_train_time
        print('{} train time : {}'.format(self.method.upper(), train_time))

        pred = self.cls.predict(self.X_test)
        pred_bag = bagged_cls.predict(self.X_test)
        true = self.y_test.values

        self.fbeta_score_ = [fbeta_score(true, pred_bag, beta=0.5), fbeta_score(true, pred, beta=0.5)]
        fig, ax = plt.subplots(ncols=2, figsize=(20, 5))

        sns.heatmap(confusion_matrix(true, pred_bag), annot=True, cmap="Blues", ax=ax[0])
        ax[0].set_title('bagged {} confusion matrix \n score = {} \n fbeta score = {}'.format(
            self.method.upper(),
            bagged_cls.score(self.X_test, self.y_test),
            self.fbeta_score_[0])
        )

        sns.heatmap(confusion_matrix(true, pred), annot=True, cmap="Blues", ax=ax[1])
        ax[1].set_title('{} confusion matrix \n score = {} \n fbeta score = {}'.format(
            self.method.upper(),
            self.cls.score(self.X_test, self.y_test),
            self.fbeta_score_[1])
        )
        plt.show()
        del train_time, pred, true, fig, ax, pred_bag
        gc.collect()


def heatmap_print(true, pred, title):
    """
    display heatmap of classification results
    :param true:
    :param pred:
    :param title:
    """
    plt.figure(figsize=(10, 5))
    sns.heatmap(confusion_matrix(true, pred, normalize='all'), annot=True, cmap="Blues")
    plt.title(title)
    plt.show()


def classify_with_proba(probability_pred, proba_0=0.5):
    """
    classify using probability
    :param probability_pred:
    :param proba_0:
    :return:
    """
    pred_0 = probability_pred[0]
    if pred_0 >= proba_0:
        return 0
    else:
        return 1

