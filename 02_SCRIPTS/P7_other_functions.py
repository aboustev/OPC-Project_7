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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, f1_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import time
import gc
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

path_to_scripts = r'..\02_SCRIPTS'
sys.path.insert(0, path_to_scripts)
from P7_dataprep_function import main


def preprocess_x(x):
    """
    preprocess data before splitting
    """
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(x)
    df_out = pd.DataFrame(scaled_df, columns=x.columns)
    return df_out


class TrainTestGrid:
    """
    gridsearch the best parameters and display the results of the method chosen
    """

    def __init__(self, df, topfeat, prepro_mthd=None, method='knn', n_esti=None):
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
        self.f_measure = None
        self.precision = None
        self.recall = None

        # input to preprocess
        self.df = df
        self.topfeat = topfeat
        self.prepro_mthd = prepro_mthd
        self.y = None
        self.x = None

        # Min max scaler on original dataset only
        self.x_ori = preprocess_x(self.df[self.topfeat])
        self.y_ori = self.df.TARGET

        self.resampling()
        self.adjust_nan()
        if self.x is None:
            print('None x input')
        else:
            self.x = preprocess_x(self.x)
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

    def adjust_nan(self):
        imputer = KNNImputer(n_neighbors=2)
        self.x = pd.DataFrame(imputer.fit_transform(self.x), columns=[self.topfeat])

    def resampling(self):
        """
        function for resampling
        """
        if self.prepro_mthd == "under":
            self.undersamp()
        elif self.prepro_mthd == "over":
            self.oversamp()
        elif self.prepro_mthd == "smote":
            self.smotesamp()
        else:
            self.x = self.df[self.topfeat]
            self.y = self.df['TARGET']

    def undersamp(self):
        """
        function of undersampling
        """
        class_count_0, class_count_1 = self.df['TARGET'].value_counts()
        class_0 = self.df[self.df['TARGET'] == 0]
        class_1 = self.df[self.df['TARGET'] == 1]
        class_0_under = class_0.sample(class_count_1)
        df_samp = pd.concat([class_0_under, class_1])
        self.x = df_samp[self.topfeat]
        self.y = df_samp['TARGET']

    def oversamp(self):
        """
        function of oversampling
        """
        class_count_0, class_count_1 = self.df['TARGET'].value_counts()
        class_0 = self.df[self.df['TARGET'] == 0]
        class_1 = self.df[self.df['TARGET'] == 1]
        class_1_over = class_1.sample(class_count_0, replace=True)
        df_samp = pd.concat([class_1_over, class_0])
        self.x = df_samp[self.topfeat]
        self.y = df_samp['TARGET']

    def smotesamp(self):
        """
        function for smote
        """
        smote = SMOTE()
        # fit predictor and target variable
        self.x, self.y = smote.fit_resample(self.df[self.topfeat], self.df.TARGET)

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
        elif self.method == 'lgbm':
            self.lgbm_params()
        elif self.method == 'lreg':
            self.lreg_params()
        elif self.method == 'dummy':
            self.cls = DummyClassifier(random_state=0)
        else:
            print('Wrong method input, please make sure you \
                entered either ["knn", "svc", "sgd", "rfc", "lgbm", "lreg", "dummy"]')
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

    def lreg_params(self):
        """
        Logistic regression parameters and method
        """
        params = {
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [200, 300, 400]
        }
        self.params = params
        self.mthd = LogisticRegression(random_state=0)
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

    def lgbm_params(self):
        """
        lgbm parameters and method
        """
        params = {
            'n_estimators': list(range(1000, 10000, 1000)),
            'learning_rate': [0.01, 0.02, 0.03, 0.04]
        }
        self.params = params
        self.mthd = LGBMClassifier(
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775)
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

        pred = self.cls.predict(self.x_ori)
        true = self.y_ori.values
        self.fbeta_score_ = fbeta_score(true, pred, beta=0.5)
        self.f_measure = f1_score(true, pred)
        self.precision = precision_score(true, pred)
        plt.figure(figsize=(10, 5))
        sns.heatmap(confusion_matrix(true, pred), annot=True, cmap="Blues")
        plt.title('{} confusion matrix\nscore = {}\nfbeta score = {}\nf1 score = {}\nprecision = {}'.format(
            self.method.upper(),
            self.cls.score(self.X_test, self.y_test),
            self.fbeta_score_,
            self.f_measure,
            self.precision)
        )
        plt.show()
        del train_time, pred, true
        gc.collect()

    def bagged_fit_and_pred(self):
        """
        makes bags, fit method, predict results and display confusion matrix
        """
        bagged_cls = BaggingClassifier(self.cls, n_estimators=self.n_esti)
        start_train_time = time.time()
        bagged_cls.fit(self.X_train, self.y_train)
        train_time = time.time() - start_train_time
        print('Bagged {} train time : {}'.format(self.method.upper(), train_time))

        start_train_time = time.time()
        self.cls.fit(self.X_train, self.y_train)
        train_time = time.time() - start_train_time
        print('{} train time : {}'.format(self.method.upper(), train_time))

        pred = self.cls.predict(self.x_ori)
        pred_bag = bagged_cls.predict(self.x_ori)
        true = self.y_ori.values

        self.fbeta_score_ = [fbeta_score(true, pred_bag, beta=0.5), fbeta_score(true, pred, beta=0.5)]
        self.f_measure = [f1_score(true, pred_bag), f1_score(true, pred)]
        self.precision = [precision_score(true, pred_bag), precision_score(true, pred)]
        fig, ax = plt.subplots(ncols=2, figsize=(20, 5))

        sns.heatmap(confusion_matrix(true, pred_bag), annot=True, cmap="Blues", ax=ax[0])
        ax[0].set_title(
            'Bagged {} confusion matrix\nscore = {}\nfbeta score = {}\nf1 score = {}\nprecision = {}'.format(
                self.method.upper(),
                bagged_cls.score(self.X_test, self.y_test),
                self.fbeta_score_[0],
                self.f_measure[0],
                self.precision[0])
        )

        sns.heatmap(confusion_matrix(true, pred), annot=True, cmap="Blues", ax=ax[1])
        ax[1].set_title('{} confusion matrix\nscore = {}\nfbeta score = {}\nf1 score = {}\nprecision = {}'.format(
            self.method.upper(),
            self.cls.score(self.X_test, self.y_test),
            self.fbeta_score_[1],
            self.f_measure[1],
            self.precision[1])
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


if __name__ == '__main__':
    df = main(nrows=100000)
    top_feat = ['AMT_ANNUITY',
                'AMT_CREDIT',
                'AMT_INCOME_TOTAL',
                'AMT_REQ_CREDIT_BUREAU_QRT',
                'AMT_REQ_CREDIT_BUREAU_YEAR',
                'ANNUITY_INCOME_PERC',
                'APARTMENTS_AVG',
                'BASEMENTAREA_AVG',
                'COMMONAREA_MEDI',
                'DAYS_BIRTH',
                'DAYS_EMPLOYED',
                'DAYS_ID_PUBLISH',
                'DAYS_LAST_PHONE_CHANGE',
                'DAYS_REGISTRATION',
                'EXT_SOURCE_1',
                'EXT_SOURCE_2',
                'EXT_SOURCE_3',
                'HOUR_APPR_PROCESS_START',
                'INCOME_CREDIT_PERC',
                'INCOME_PER_PERSON',
                'LANDAREA_AVG',
                'LIVINGAREA_MODE',
                'NONLIVINGAREA_AVG',
                'OBS_30_CNT_SOCIAL_CIRCLE',
                'OWN_CAR_AGE',
                'PAYMENT_RATE',
                'REGION_POPULATION_RELATIVE',
                'TOTALAREA_MODE',
                'YEARS_BEGINEXPLUATATION_AVG',
                'YEARS_BEGINEXPLUATATION_MODE']
    df_modele = df[top_feat + ['TARGET']]
    df_modele_train = df_modele[df_modele['TARGET'].notnull()]
    df_modele_test = df_modele[df_modele['TARGET'].isnull()]
    under_lreg_results = TrainTestGrid(df_modele_train, top_feat, prepro_mthd="under", method='lreg')
