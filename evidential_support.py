# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# evidential_support.py
#
# Identification: /evidential_support.py
#
# Ben Chaddha
#
# ===----------------------------------------------------------------------===//


from copy import copy
from scipy.sparse import *
from scipy.stats import t
from sklearn.linear_model import LinearRegression
import logging
import numpy as np
import gml_utils



class Regression:
    '''
    Linear regression related classes, perform linear regression on all features
    Input: a feature
    Output: regression object
    '''
    def __init__(self, each_feature_easys, n_job, effective_training_count_threshold=2):
        '''
        todo:
        Update strategy for feature regression: only regression evidence supports changed features
        '''
        self.effective_training_count = max(2, effective_training_count_threshold)
        self.n_job = n_job
        if len(each_feature_easys) > 0:
            XY = np.array(each_feature_easys)
            self.X = XY[:, 0].reshape(-1, 1)
            self.Y = XY[:, 1].reshape(-1, 1)
        else:
            self.X = np.array([]).reshape(-1, 1)
            self.Y = np.array([]).reshape(-1, 1)
        self.balance_weight_y0_count = 0
        self.balance_weight_y1_count = 0
        for y in self.Y:
            if y > 0:
                self.balance_weight_y1_count += 1
            else:
                self.balance_weight_y0_count += 1
        self.perform()

    def perform(self):
        '''Perform linear regression'''
        self.N = np.size(self.X)
        if self.N <= self.effective_training_count:
            self.regression = None
            self.residual = None
            self.meanX = None
            self.variance = None
            self.k = None
            self.b = None
        else:
            sample_weight_list = None
            if self.balance_weight_y1_count > 0 and self.balance_weight_y0_count > 0:
                sample_weight_list = list()
                sample_weight = float(self.balance_weight_y0_count) / self.balance_weight_y1_count
                for y in self.Y:
                    if y[0] > 0:
                        sample_weight_list.append(sample_weight)
                    else:
                        sample_weight_list.append(1)
            self.regression = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=self.n_job).fit(self.X, self.Y,
                                                                                                       sample_weight=sample_weight_list)
            self.residual = np.sum((self.regression.predict(self.X) - self.Y) ** 2) / (self.N - 2)
            self.meanX = np.mean(self.X)  # Average value of feature_value of all evidence variables of this feature
            self.variance = np.sum((self.X - self.meanX) ** 2)
            z = self.regression.predict(np.array([0, 1]).reshape(-1, 1))
            self.k = (z[1] - z[0])[0]
            self.b = z[0][0]

    def append(self, appendx, appendy):
        '''Add training data in real time'''
        self.X = np.append(self.X, [[appendx]], axis=0)
        self.Y = np.append(self.Y, [[appendy]], axis=0)
        if appendy > 0:
            self.balance_weight_y1_count += 1
        else:
            self.balance_weight_y0_count += 1
        self.perform()

    def disable(self, delx, dely):
        '''Delete training data in real time'''
        for index in range(0, len(self.X)):
            if self.X[index][0] == delx and self.Y[index][0] == dely:
                self.X = np.delete(self.X, index, axis=0)
                self.Y = np.delete(self.Y, index, axis=0)
                if dely > 0:
                    self.balance_weight_y1_count -= 1
                else:
                    self.balance_weight_y0_count -= 1
                break
        self.perform()


class EvidentialSupport:
    def __init__(self, variables, features, method='regression', evidence_interval_count=10, interval_evidence_count=200):
        self.variables = variables
        self.features = features
        self.features_easys = dict()  # Store all easy feature values of all features: feature_id: [[value1, bound], [value2, bound]...]
        self.tau_and_regression_bound = 10
        self.evidence_interval_count = evidence_interval_count  # The number of intervals is 10
        self.interval_evidence_count = interval_evidence_count  # The number of variables in each interval is 200
        self.NOT_NONE_VALUE = 1e-8
        self.n_job = 10
        self.delta = 2
        self.effective_training_count_threshold = 2
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        if method == 'regression':
            self.data_matrix = self.create_csr_matrix()
            self.evidence_interval = gml_utils.init_evidence_interval(self.evidence_interval_count)
            gml_utils.init_evidence(self.features, self.evidence_interval, self.observed_variables_set)

    def separate_feature_value(self):
        # Select the easy feature value of each feature for linear regression
        each_feature_easys = list()
        self.features_easys.clear()
        for feature in self.features:
            each_feature_easys.clear()
            for var_id, value in feature['weight'].items():
                # The feature_value of each easy variable owned by each feature
                if var_id in self.observed_variables_set:
                    each_feature_easys.append([value[1], (1 if self.variables[var_id]['label'] == 1 else -1) * self.tau_and_regression_bound])
            self.features_easys[feature['feature_id']] = copy(each_feature_easys)

    def create_csr_matrix(self):
        # Create a sparse matrix to store all feature values of all variables for subsequent Evidential Support calculation
        data = list()
        row = list()
        col = list()
        for index, var in enumerate(self.variables):
            feature_set = self.variables[index]['feature_set']
            for feature_id in feature_set:
                data.append(feature_set[feature_id][1] + self.NOT_NONE_VALUE)
                row.append(index)
                col.append(feature_id)
        return csr_matrix((data, (row, col)), shape=(len(self.variables), len(self.features)))

    def influence_modeling(self, update_feature_set):
        '''Perform linear regression on the updated features
        Store the regression results back to the features, with the key 'regression'
        '''
        if len(update_feature_set) > 0:
            self.init_tau_and_alpha(update_feature_set)
            logging.info("init tau&alpha finished")
            for feature_id in update_feature_set:
                # For features with empty features_easys, the regression will be None
                self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id], n_job=self.n_job)
            logging.info("feature regression finished")

    def init_tau_and_alpha(self, feature_set):
        '''Calculate tau and alpha for the given features

        Input: feature_set is a set or list of feature_ids
        Output: Modify the tau and alpha attributes directly in the features
        '''
        if type(feature_set) != list and type(feature_set) != set:
            print("Input parameter error, should be a set or list")
            return
        else:
            for feature_id in feature_set:
                # tau value is fixed to the upper bound
                self.features[feature_id]["tau"] = self.tau_and_regression_bound
                weight = self.features[feature_id]["weight"]
                labelvalue0 = 0
                num0 = 0
                labelvalue1 = 0
                num1 = 0
                for key in weight:
                    if self.variables[key]["is_evidence"] and self.variables[key]["label"] == 0:
                        labelvalue0 += weight[key][1]
                        num0 += 1
                    elif self.variables[key]["is_evidence"] and self.variables[key]["label"] == 1:
                        labelvalue1 += weight[key][1]
                        num1 += 1
                if num0 == 0 and num1 == 0:
                    continue
                if num0 == 0:
                    # If there are no label0 connected to the feature, set the value to the upper bound, currently set to 1
                    labelvalue0 = 1
                else:
                    # Average value of feature_value for label0
                    labelvalue0 /= num0
                if num1 == 0:
                    # Same as above
                    labelvalue1 = 1
                else:
                    # Average value of feature_value for label1
                    labelvalue1 /= num1
                alpha = (labelvalue0 + labelvalue1) / 2
                self.features[feature_id]["alpha"] = alpha

    def evidential_support_by_regression(self, update_feature_set):
        '''Calculate the Evidential Support for all latent variables using linear regression'''

        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        self.separate_feature_value()
        self.influence_modeling(update_feature_set)
        coo_data = self.data_matrix.tocoo()
        row, col, data = coo_data.row, coo_data.col, coo_data.data
        coefs = []
        intercept = []
        residuals = []
        Ns = []
        meanX = []
        variance = []
        delta = self.delta
        zero_confidence = []
        for feature in self.features:
            if feature['regression'].regression is not None and feature['regression'].variance > 0:
                coefs.append(feature['regression'].regression.coef_[0][0])
                intercept.append(feature['regression'].regression.intercept_[0])
                zero_confidence.append(1)
            else:
                coefs.append(0)
                intercept.append(0)
                zero_confidence.append(0)
            Ns.append(feature['regression'].N if feature['regression'].N > feature[
                'regression'].effective_training_count else np.NaN)
            residuals.append(feature['regression'].residual if feature['regression'].residual is not None else np.NaN)
            meanX.append(feature['regression'].meanX if feature['regression'].meanX is not None else np.NaN)
            variance.append(feature['regression'].variance if feature['regression'].variance is not None else np.NaN)
        coefs = np.array(coefs)[col]
        intercept = np.array(intercept)[col]
        zero_confidence = np.array(zero_confidence)[col]
        residuals, Ns, meanX, variance = np.array(residuals)[col], np.array(Ns)[col], np.array(meanX)[col], \
                                         np.array(variance)[col]
        tvalue = float(delta) / (residuals * np.sqrt(1 + 1.0 / Ns + np.power(data - meanX, 2) / variance))
        confidence = np.ones_like(data)
        confidence[np.where(residuals > 0)] = (1 - t.sf(tvalue, (Ns - 2)) * 2)[np.where(residuals > 0)]
        confidence = confidence * zero_confidence
        evidential_support = (1 + confidence) / 2  # Normalize
        # Write the calculated evidential support back
        csr_evidential_support = csr_matrix((evidential_support, (row, col)),
                                            shape=(len(self.variables), len(self.features)))
        for index, var in enumerate(self.variables):
            for feature_id in var['feature_set']:
                var['feature_set'][feature_id][0] = csr_evidential_support[index, feature_id]
        # Calculate approximate weights
        predict = data * coefs + intercept
        espredict = predict * evidential_support
        # espredict[np.where(polar_enforce == 0)] = np.minimum(espredict, 0)[np.where(polar_enforce == 0)]   # Set to 0 if less than 0
        # espredict[np.where(polar_enforce == 1)] = np.maximum(espredict, 0)[np.where(polar_enforce == 1)]   # Set to 1 if greater than 0
        espredict = espredict * zero_confidence
        # Verify that all evidential support values are between (0, 1)
        assert len(np.where(evidential_support < 0)[0]) == 0
        assert len(np.where((1 - evidential_support) < 0)[0]) == 0
        loges = np.log(evidential_support)  # Take natural logarithm
        logunes = np.log(1 - evidential_support)
        evidential_support_logit = csr_matrix((loges, (row, col)), shape=(len(self.variables), len(self.features)))
        evidential_unsupport_logit = csr_matrix((logunes, (row, col)), shape=(len(self.variables), len(self.features)))
        p_es = np.exp(np.array(evidential_support_logit.sum(axis=1)))
        p_unes = np.exp(np.array(evidential_unsupport_logit.sum(axis=1)))
        approximate_weight = csr_matrix((espredict, (row, col)), shape=(len(self.variables), len(self.features)))
        approximate_weight = np.array(approximate_weight.sum(axis=1)).reshape(-1)
        # Write the calculated approximate weights back to variables
        for index, var in enumerate(self.variables):
            var['approximate_weight'] = approximate_weight[index]
        logging.info("approximate_weight calculate finished")
        # Calculate the overall evidential support for each latent variable and write it back to self.variables
        for var_id in self.poential_variables_set:
            index = var_id
            var_p_es = p_es[index]
            var_p_unes = p_unes[index]
            self.variables[index]['evidential_support'] = float(var_p_es / (var_p_es + var_p_unes))
        logging.info("evidential_support calculate finished")

    def evidential_support_by_relation(self, update_feature_set):
        pass

    def evidential_support_by_custom(self, update_feature_set):
        pass
