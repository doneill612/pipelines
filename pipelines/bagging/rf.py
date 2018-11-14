# -*- coding: utf-8 -*-
# Supress useless Scikitlearn warnings...
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import copy
import numpy as np
import pandas as pd
from core import model_logging
from core.abstract_model import AbstractModel
from core.defaults import default_data_path, default_rf_params
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class RandomForest(AbstractModel):

    def __init__(self, data_loc, params):
        super(RandomForest, self).__init__(data_loc, params)

    def eval_metrics(self):
        forest_params = self._params['forest_params']
        return copy.deepcopy(forest_params['eval_metric'])

    def train(self):
        os_features, os_labels = self.perform_smote()
        os_features_train, os_features_val, os_labels_train, os_labels_val = \
            self.build_train_test_set(os_features, os_labels)
        self._model = RandomForestClassifier(**self._params['forest_params'])
        self._model.fit(os_features_train, os_labels_train)
        model_logging.info('Training complete!')
        v_actual = os_labels_val
        v_predicted = self._model.predict(os_features_val)
        model_logging.info('Printing validation confusion matrix...')
        print('tn, fp, fn, tp => ', confusion_matrix(v_actual, v_predicted).ravel(), '\n')
        model_logging.info('Printing validation AUROC/Recall/F1...')
        fpr, tpr, thresholds = roc_curve(v_actual, v_predicted)
        roc_auc = auc(fpr, tpr)
        model_logging.info('AUROC = %f' % roc_auc)
        model_logging.info('Recall = %f' % recall_score(v_actual, v_predicted))
        model_logging.info('F1 = %f\n' % f1_score(v_actual, v_predicted))

    def test(self):
        actual = self._train_test_split[3]
        predicted = self._model.predict(self._train_test_split[1])
        model_logging.info('Printing test confusion matrix...')
        res = confusion_matrix(actual, predicted).ravel()
        print('tn, fp, fn, tp =', res)
        model_logging.info('Total tested = %i' % np.sum(res, axis=0))
        fpr, tpr, thresholds = roc_curve(actual, predicted)
        roc_auc = auc(fpr, tpr)
        model_logging.info('Printing test Recall... (0.5 threshold)')
        model_logging.info(recall_score(actual, predicted))
        model_logging.info('Printing test F1... (0.5 threshold)')
        model_logging.info('{}\n'.format(f1_score(actual, predicted)))
        predicted_proba = self._model.predict_proba(self._train_test_split[1])
        predictions = [predicted_proba[i][1] for i in range(len(predicted_proba))]
        actual_list = [int(a) for a in actual]
        zipped = zip(actual_list, predictions)
        thresholds = [0.35, 0.5, 0.6]
        for t in thresholds:
            self.perform_risk_assesement(copy.deepcopy(zipped), t)

    def perform_risk_assesement(self, zipped, threshold):
        model_logging.info('PERFORMING RISK ASSESMENT, threshold = %f\n' % threshold)

        very_low_risk = []
        low_risk = []
        moderate_risk = []
        high_risk = []

        very_low_prob = []
        low_prob = []
        moderate_prob = []
        high_prob = []

        for true_value, probability in zipped:
            if probability >= 0 and probability <= .25:
                very_low_risk.append(true_value)
                very_low_prob.append(probability)
            if probability > .25 and probability <= .50:
                low_risk.append(true_value)
                low_prob.append(probability)
            if probability > .50 and probability <= .75:
                moderate_risk.append(true_value)
                moderate_prob.append(probability)
            elif probability > .75 and probability <= 1.0:
                high_risk.append(true_value)
                high_prob.append(probability)

        vlow_pnd_count = sum(very_low_risk)
        very_low_prob_count = sum([1 if i > threshold else 0 for i in very_low_prob])
        vlow_total = len(very_low_risk)

        low_pnd_count = sum(low_risk)
        low_prob_count = sum([1 if i > threshold else 0 for i in low_prob])
        low_total = len(low_risk)

        moderate_pnd_count = sum(moderate_risk)
        moderate_prob_count = sum([1 if i > threshold else 0 for i in moderate_prob])
        moderate_total = len(moderate_risk)

        high_pnd_count = sum(high_risk)
        high_prob_count = sum([1 if i > threshold else 0 for i in high_prob])
        high_total = len(high_risk)

        model_logging.info('Very low PND count: %i' % vlow_pnd_count)
        model_logging.info('Very low predictions: %i' % vlow_total)
        model_logging.info('Very low predictions above threshold %f: %i\n' %
                           (threshold, very_low_prob_count))

        model_logging.info('Low PND count: %i' % low_pnd_count)
        model_logging.info('Low predictions: %i' % low_total)
        model_logging.info('Low predictions above threshold %f: %i\n' %
                           (threshold, low_prob_count))

        model_logging.info('Moderate PND count: %i' % moderate_pnd_count)
        model_logging.info('Moderate predictions: %i' % moderate_total)
        model_logging.info('Moderate predictions above threshold %f: %i\n' %
                           (threshold, moderate_prob_count))

        model_logging.info('High PND count: %i' % high_pnd_count)
        model_logging.info('High predictions: %i' % high_total)
        model_logging.info('High predictions above threshold %f: %i\n' %
                           (threshold, high_prob_count))

    def to_string(self):
        return 'Random Forest Classifier'

    def perform_smote(self):
        features = self.extract_features()
        labels = self.extract_labels()
        self._train_test_split = self.build_train_test_set(features, labels)
        model_logging.info("Train size: %i" % len(self._train_test_split[0]))
        model_logging.info("Test size: %i" % len(self._train_test_split[1]))
        model_logging.info("Before SMOTE")
        model_logging.info(
            "0 labels train: %i" %
            len(self._train_test_split[2][self._train_test_split[2]==0]))
        model_logging.info(
            "1 labels train: %i" %
            len(self._train_test_split[2][self._train_test_split[2]==1]))
        model_logging.info(
            "0 labels test: %i" %
            len(self._train_test_split[3][self._train_test_split[3]==0]))
        model_logging.info(
            "1 labels test: %i" %
            len(self._train_test_split[3][self._train_test_split[3]==1]))
        oversampler = SMOTE(random_state=1337, ratio=1.0)
        os_features, os_labels = oversampler.fit_sample(
                                    self._train_test_split[0],
                                    self._train_test_split[2])
        model_logging.info("After SMOTE")
        model_logging.info("0 labels: %i" % len(os_labels[os_labels==0]))
        model_logging.info("1 labels: %i" % len(os_labels[os_labels==1]))
        return os_features, os_labels
