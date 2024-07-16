# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# gml.py
#
# Benjamin Chaddha
#
# ===----------------------------------------------------------------------===//

import csv
import heapq
import pickle
from collections import namedtuple
from scipy.sparse import *
import numbskull
from numbskull.numbskulltypes import *
import random
import logging
import time
import warnings
import pandas as pd
from sklearn import metrics
import gml_utils
from evidential_support import EvidentialSupport
from easy_instance_labeling import EasyInstanceLabeling
from evidence_select import EvidenceSelect
from approximate_probability_estimation import ApproximateProbabilityEstimation
import helper

class GML:
    '''
    GML class: includes computing Evidential Support, Approximate Estimation of Inferred Probability,
    Construction of Inference Subgraph, etc.; does not include Feature Extraction and Easy Instance Labeling.
    In the implementation process, pay attention to distinguishing between instance variables and class variables.
    '''

    def __init__(self, variables, features, evidential_support_method, approximate_probability_method, evidence_select_method, top_m=2000, top_k=10, update_proportion=0.01, balance=False):
        '''
        Initialize GML object.
        :param variables: Dictionary of variables.
        :param features: Dictionary of features.
        :param evidential_support_method: Method for computing evidential support.
        :param approximate_probability_method: Method for estimating approximate probability.
        :param evidence_select_method: Method for selecting evidence.
        :param top_m: Number of top hidden variables to select.
        :param top_k: Number of top variables to select based on entropy.
        :param update_proportion: Proportion of variables to update evidential support.
        :param balance: Boolean indicating whether to balance the evidence.
        '''
        self.variables = variables
        self.features = features
        self.evidential_support_method = evidential_support_method
        self.evidence_select_method = evidence_select_method
        self.approximate_probability_method = approximate_probability_method
        self.labeled_variables_set = set()  # Set of all newly labeled variables
        self.top_m = top_m
        self.top_k = top_k
        self.update_proportion = update_proportion
        self.balance = balance
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(variables)
        self.support = EvidentialSupport(variables, features, evidential_support_method)
        self.select = EvidenceSelect(variables, features)
        self.approximate = ApproximateProbabilityEstimation(variables)
        logging.basicConfig(
            level=logging.INFO,  # Set the output information level
            format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s'  # Set the output format
        )

    def evidential_support(self, update_feature_set):
        '''
        Compute evidential support.
        :param update_feature_set: Set of features to update evidential support.
        '''
        method = 'self.support.evidential_support_by_' + self.evidential_support_method+'(update_feature_set)'
        eval(method)

    def approximate_probability_estimation(self, var_id):
        '''
        Compute approximate probability.
        :param var_id: Variable ID.
        '''
        method = 'self.approximate.approximate_probability_estimation_by_'+self.approximate_probability_method+'(var_id)'
        eval(method)

    def select_top_m_by_es(self, m):
        '''
        Select the top m hidden variables based on the computed Evidential Support (from largest to smallest).
        :param m: Number of hidden variables to select.
        :return: A list containing m variable IDs.
        '''
        # Only select from all potential variables
        poential_var_list = list()
        m_id_list = list()
        for var_id in self.poential_variables_set:
            poential_var_list.append([var_id, self.variables[var_id]['evidential_support']])
        topm_var = heapq.nlargest(m, poential_var_list, key=lambda s: s[1])
        for elem in topm_var:
            m_id_list.append(elem[0])
        logging.info('select m finished')
        return m_id_list

    def select_top_k_by_entropy(self, var_id_list, k):
        '''
        Compute entropy and select the top k variables with the smallest entropy.
        :param var_id_list: Range of selection.
        :param k: Number of variables to select.
        :return: A list containing k variable IDs.
        '''
        m_list = list()
        k_id_list = list()
        for var_id in var_id_list:
            var_index = var_id
            self.variables[var_index]['entropy'] = gml_utils.entropy(self.variables[var_index]['probability'])
            m_list.append(self.variables[var_index])
        k_list = heapq.nsmallest(k, m_list, key=lambda x: x['entropy'])
        for var in k_list:
            k_id_list.append(var['var_id'])
        logging.info('select k finished')
        return k_id_list

    def select_evidence(self, var_id):
        '''
        Select the edges, variables, and features needed to construct the subgraph for subsequent inference.
        :param var_id: Variable ID.
        :return: Connected variable set, connected edge set, connected feature set.
        '''
        method = 'self.select.select_evidence_by_'+self.evidence_select_method+"(var_id)"
        connected_var_set, connected_edge_set, connected_feature_set = eval(method)
        return connected_var_set, connected_edge_set, connected_feature_set

    def construct_subgraph(self, var_id):
        '''
        Construct the subgraph after selecting the top k hidden variables.
        :param var_id: Hidden variable ID.
        :return: Weight, Variable, Factor, Fmap, Domain_mask, Edges according to the requirements of numbskull.
        '''
        var_index = var_id
        feature_set = self.variables[var_index]['feature_set']
        evidence_set, partial_edges, connected_feature_set = self.select_evidence(var_id)
        # Balance
        if self.balance:
            label0_var = set()
            label1_var = set()
            for var_id in evidence_set:
                if variables[var_id]['label'] == 1:
                    label1_var.add(var_id)
                elif variables[var_id]['label'] == 0:
                    label0_var.add(var_id)
            sampled_label0_var = set(random.sample(list(label0_var), len(label1_var)))
            new_evidence_set = label1_var.union(sampled_label0_var)
            new_partial_edges = set()
            new_connected_feature_set = set()
            for edge in partial_edges:
                if edge[1] in new_evidence_set:
                    new_partial_edges.add(edge)
                    new_connected_feature_set.add(edge[0])
            evidence_set = new_evidence_set
            partial_edges = new_partial_edges
            connected_feature_set = new_connected_feature_set
        var_map = dict()  # Used to record the mapping between self.variables and numbskull's variable variables - (self, numbskull)
        # Initialize variables
        var_num = len(evidence_set) + 1  # Evidence variables + hidden variables
        variable = np.zeros(var_num, Variable)
        # Initialize hidden variables, the ID of the hidden variable is 0, distinguish between the overall variables and the variables in the subgraph
        variable[0]["isEvidence"] = False
        variable[0]["initialValue"] = self.variables[var_index]['label']
        variable[0]["dataType"] = 0  # datatype=0 indicates a boolean variable, 1 indicates a non-boolean variable.
        variable[0]["cardinality"] = 2
        var_map[var_index] = 0  # (self, numbskull) Hidden variable ID is 0
        i = 1
        # Initialize evidence variables
        for evidence_id in evidence_set:
            var_index = evidence_id
            variable[i]["isEvidence"] = True  # self.variables[var_index]['is_evidence']
            variable[i]["initialValue"] = self.variables[var_index]['label']
            variable[i]["dataType"] = 0  # datatype=0 indicates a boolean variable, 1 indicates a non-boolean variable.
            variable[i]["cardinality"] = 2
            var_map[evidence_id] = i  # One-to-one record
            i += 1
        # Initialize weights, multiple factors can share the same weight
        weight = np.zeros(len(connected_feature_set), Weight)  # The number of weights is equal to the number of features used by this hidden variable
        feature_map_weight = dict()  # Need to record the mapping between feature id and weight id [feature_id, weight_id]
        weight_index = 0
        for feature_id in connected_feature_set:
            weight[weight_index]["isFixed"] = False
            weight[weight_index]["parameterize"] = True
            weight[weight_index]["a"] = self.features[feature_id]['tau']
            weight[weight_index]["b"] = self.features[feature_id]['alpha']
            weight[weight_index]["initialValue"] = random.uniform(-5, 5)  # Here, a weight can have many weight values, here randomly initialize one, which should not be used later
            feature_map_weight[feature_id] = weight_index
            weight_index += 1

        # Initialize factors, fmap, edges according to the requirements of numbskull
        edges_num = len(connected_feature_set) + len(partial_edges)  # Number of edges
        factor = np.zeros(edges_num, Factor)  # Currently treating all as single factors, so there are as many factors as there are edges
        fmap = np.zeros(edges_num, FactorToVar)
        domain_mask = np.zeros(var_num, np.bool)
        edges = list()
        edge = namedtuple('edge', ['index', 'factorId', 'varId'])  # Edge for single variable factor
        factor_index = 0
        fmp_index = 0
        edge_index = 0
        # Initialize all single factors connected to this hidden variable first
        for feature_id in connected_feature_set:  # The dictionary is unordered, but the keys obtained are ordered
            factor[factor_index]["factorFunction"] = 18
            factor[factor_index]["weightId"] = feature_map_weight[feature_id]
            factor[factor_index]["featureValue"] = feature_set[feature_id][1]
            factor[factor_index]["arity"] = 1  # Single factor arity is 1
            factor[factor_index]["ftv_offset"] = fmp_index  # Offset increases by 1 each time
            # Save the edges on this hidden variable first
            edges.append(edge(edge_index, factor_index, 0))
            fmap[fmp_index]["vid"] = 0  # edges[factor_index][2]
            fmap[fmp_index]["x"] = feature_set[feature_id][1]  # feature_value
            fmap[fmp_index]["theta"] = feature_set[feature_id][0]  # theta
            fmp_index += 1
            factor_index += 1
            edge_index += 1
        # Initialize single factors connected to evidence variables
        for elem in partial_edges:  # [feature_id, var_id]
            var_index = elem[1]
            factor[factor_index]["factorFunction"] = 18
            factor[factor_index]["weightId"] = feature_map_weight[elem[0]]
            factor[factor_index]["featureValue"] = self.variables[var_index]['feature_set'][elem[0]][1]
            factor[factor_index]["arity"] = 1  # Single factor arity is 1
            factor[factor_index]["ftv_offset"] = fmp_index  # Offset increases by 1 each time
            edges.append(edge(edge_index, factor_index, var_map[elem[1]]))
            fmap[fmp_index]["vid"] = edges[factor_index][2]
            fmap[fmp_index]["x"] = self.variables[var_index]['feature_set'][elem[0]][1]  # feature_value
            fmap[fmp_index]["theta"] = self.variables[var_index]['feature_set'][elem[0]][0]  # theta
            fmp_index += 1
            factor_index += 1
            edge_index += 1
        logging.info("var-" + str(var_id) + " construct subgraph succeed")
        subgraph = weight, variable, factor, fmap, domain_mask, edges_num, var_map, feature_map_weight
        return subgraph

    def inference_subgraph(self, var_id):
        '''
        Infer the subgraph.
        :param var_id: Variable ID or set of variable IDs.
        '''
        learn = 1000
        ns = numbskull.NumbSkull(n_inference_epoch=10,
                                 n_learning_epoch=learn,
                                 quiet=True,
                                 learn_non_evidence=True,
                                 stepsize=0.0001,
                                 burn_in=10,
                                 decay=0.001 ** (1.0 / learn),
                                 regularization=1,
                                 reg_param=0.01)

        weight, variable, factor, fmap, domain_mask, edges_num, var_map, feature_map_weight = self.construct_subgraph(var_id)
        subgraph = weight, variable, factor, fmap, domain_mask, edges_num
        ns.loadFactorGraph(*subgraph)
        # Factor graph parameter learning
        ns.learning()
        logging.info("subgraph learning finished")
        # Factor graph inference
        # After parameter learning, set the isfixed attribute of weight to true
        for w in ns.factorGraphs[0].weight:
            w["isFixed"] = True
        ns.learn_non_evidence = False
        ns.inference()
        logging.info("subgraph inference finished")
        # Write probabilities back to self.variables
        if type(var_id) == set() or type(var_id) == list():
            for id in var_id:
                self.variables[var_id]['probability'] = ns.factorGraphs[0].marginals[var_map[var_id]]
        elif type(var_id) == int:
            self.variables[var_id]['probability'] = ns.factorGraphs[0].marginals[var_map[var_id]]
        else:
            print('Parameter error, var_id should be int or set and list')
        logging.info("inferenced probability recored")

    def label(self, var_id_list):
        '''
        Compare the entropy of k hidden variables and label the one with the smallest entropy.
        :param var_id_list: List of k variable IDs.
        :return: Variable ID that was labeled.
        '''
        entropy_list = list()
        if len(var_id_list) > 1:  # If the number of variables passed in is greater than 1, label the one with the smallest entropy each time
            for var_id in var_id_list:
                var_index = var_id
                self.variables[var_index]['entropy'] = gml_utils.entropy(
                    self.variables[var_index]['probability'])
                entropy_list.append([var_id, self.variables[var_index]['entropy']])
            min_var = heapq.nsmallest(1, entropy_list, key=lambda x: x[1])  # Select the variable with the smallest entropy
            var = min_var[0][0]
        else:
            var = var_id_list[0]
        var_index = var  # If only 1 variable is passed in, just label it
        self.variables[var_index]['label'] = 1 if self.variables[var_index]['probability'] >= 0.5 else 0
        self.variables[var_index]['is_evidence'] = True
        logging.info('var-' + str(var) + " labeled succeed---------------------------------------------")
        self.poential_variables_set.remove(var)
        self.observed_variables_set.add(var)
        self.labeled_variables_set.add(var)
        with open('result.txt', 'a') as f:
            f.write(str(var) + " " + str(self.variables[var_index]['label']) + ' ' + str(
                self.variables[var_index]['probability']) + '\n')
        return var

    def inference(self):
        '''
        Main process.
        '''
        update_cache = int(self.update_proportion * len(self.poential_variables_set))  # Recalculate evidential support after inferring update_cache variables
        labeled_var = 0
        labeled_count = 0
        var = 0  # Variable ID labeled each round
        update_feature_set = set()       # Store features whose evidential support has changed during one round of updates
        inferenced_variables_id = set()  # Hidden variables that have been inferred and constructed subgraphs during one round of updates
        for feature in self.features:
            update_feature_set.add(feature['feature_id'])
        self.evidential_support(update_feature_set)
        update_feature_set.clear()
        self.approximate_probability_estimation(self.poential_variables_set)
        logging.info("approximate_probability calculate finished")
        while len(self.poential_variables_set) > 0:
            # When the number of labeled variables reaches update_cache, re-regress and calculate evidential support
            if labeled_var == update_cache:
                for var_id in self.labeled_variables_set:
                    var_index = var_id
                    for feature_id in self.variables[var_index]['feature_set'].keys():
                        update_feature_set.add(feature_id)
                self.evidential_support(update_feature_set)
                self.approximate_probability_estimation(self.poential_variables_set)
                logging.info("approximate_probability calculate finished")
                labeled_var = 0
                update_feature_set.clear()
                self.labeled_variables_set.clear()
                inferenced_variables_id.clear()
            if len(self.poential_variables_set) >= self.top_m:  # If the number of hidden variables is less than topm, there is no need to select topm, and the labeled variables need to be removed from topm in real time
                m_list = self.select_top_m_by_es(self.top_m)
            else:
                m_list.remove(var)
            if len(self.poential_variables_set) >= self.top_k:  # If the number of hidden variables is less than topk, there is no need to select topk, and the labeled variables need to be removed from topk in real time
                k_list = self.select_top_k_by_entropy(m_list, self.top_k)
            else:
                k_list.remove(var)
            if self.evidence_select_method == 'interval':
            # As long as there is no update, only the newly added variables are inferred each time
                add_list = [x for x in k_list if x not in inferenced_variables_id]
                if len(add_list) > 0:
                    for var_id in add_list:
                        # if var_id not in inferenced_variables_id:
                        self.inference_subgraph(var_id)
                        # Hidden variables that have been inferred during each round of updates, no need to infer again because the parameters have not been updated.
                        inferenced_variables_id.add(var_id)
                var = self.label(k_list)
                gml_utils.write_labeled_var_to_evidence_interval(self.variables, self.features, var, self.support.evidence_interval)
            else:
                self.inference_subgraph(k_list)
                var = self.label(k_list)
                update_feature_set.clear()
                self.labeled_variables_set.clear()
                inferenced_variables_id.clear()
            if len(self.poential_variables_set) >= self.top_m:  # If the number of hidden variables is less than topm, there is no need to select topm, and the labeled variables need to be removed from topm in real time
                m_list = self.select_top_m_by_es(self.top_m)
            else:
                m_list.remove(var)
            if len(self.poential_variables_set) >= self.top_k:  # If the number of hidden variables is less than topk, there is no need to select topk, and the labeled variables need to be removed from topk in real time
                k_list = self.select_top_k_by_entropy(m_list, self.top_k)
            else:
                k_list.remove(var)
            if self.evidence_select_method == 'interval':
            # As long as there is no update, only the newly added variables are inferred each time
                add_list = [x for x in k_list if x not in inferenced_variables_id]
                if len(add_list) > 0:
                    for var_id in add_list:
                        # if var_id not in inferenced_variables_id:
                        self.inference_subgraph(var_id)
                        # Hidden variables that have been inferred during each round of updates, no need to infer again because the parameters have not been updated.
                        inferenced_variables_id.add(var_id)
                var = self.label(k_list)
                gml_utils.write_labeled_var_to_evidence_interval(self.variables, self.features, var, self.support.evidence_interval)
            else:
                self.inference_subgraph(k_list)
                var = self.label(k_list)
            labeled_var += 1
            labeled_count += 1
            logging.info("label_num=" + str(labeled_count))

    def score(self):
        # Calculate precision, recall, and f1-score of the inference results
        easys_pred_label = list()
        easys_true_label = list()
        hards_pred_label = list()
        hards_true_label = list()
        for var in self.variables:
            if var['is_easy'] == True:
                easys_true_label.append(var['true_label'])
                easys_pred_label.append(var['label'])
            else:
                hards_true_label.append(var['true_label'])
                hards_pred_label.append(var['label'])

        all_true_label = easys_true_label + hards_true_label
        all_pred_label = easys_pred_label + hards_pred_label

        print("--------------------------------------------")
        print("total:")
        print("--------------------------------------------")
        print("total precision_score: " + str(metrics.precision_score(all_true_label, all_pred_label)))
        print("total recall_score: " + str(metrics.recall_score(all_true_label, all_pred_label)))
        print("total f1_score: " + str(metrics.f1_score(all_true_label, all_pred_label)))
        print("--------------------------------------------")
        print("easys:")
        print("--------------------------------------------")
        print("easys precision_score:" + str(metrics.precision_score(easys_true_label, easys_pred_label)))
        print("easys recall_score:" + str(metrics.recall_score(easys_true_label, easys_pred_label)))
        print("easys f1_score: " + str(metrics.f1_score(easys_true_label, easys_pred_label)))
        print("--------------------------------------------")
        print("hards:")
        print("--------------------------------------------")
        print("hards precision_score: " + str(metrics.precision_score(hards_true_label, hards_pred_label)))
        print("hards recall_score: " + str(metrics.recall_score(hards_true_label, hards_pred_label)))
        print("hards f1_score: " + str(metrics.f1_score(hards_true_label, hards_pred_label)))

if __name__ == '__main__':
    warnings.filterwarnings('ignore')  # Filter out warning outputs
    begin_time = time.time()
    easys = helper.EasyInstanceLabeling.load_easy_instance_from_file('data/abtbuy_easys.csv')
    with open('data/abtbuy_variables.pkl', 'rb') as v:
        variables = pickle.load(v)
    with open('data/abtbuy_features.pkl', 'rb') as f:
        features = pickle.load(f)
    EasyInstanceLabeling(variables,features,easys).label_easy_by_file()
    graph = GML(variables, features, evidential_support_method = 'regression',approximate_probability_method = 'interval', evidence_select_method = 'interval')
    graph.inference()
    graph.score()
    end_time = time.time()
    print('Running time: %s Seconds' % (end_time - begin_time))
