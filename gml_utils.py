# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# gml_utils.py
#
# Benjamin Chaddha
#
# ===----------------------------------------------------------------------===//

import math
import numpy as np

# Store some utility functions

def separate_variables(variables):
    '''
    Separate variables into evidence variables and hidden variables.
    Modify the instance object with these two attributes: observed_variables_id, potential_variables_id.
    '''
    observed_variables_set = set()
    potential_variables_set = set()
    for variable in variables:
        if variable['is_evidence'] == True:
            observed_variables_set.add(variable['var_id'])
        else:
            potential_variables_set.add(variable['var_id'])
    return observed_variables_set, potential_variables_set

def init_evidence_interval(evidence_interval_count):
    '''
    Initialize evidence intervals.
    Output: a list containing evidence_interval_count number of intervals.
    '''
    evidence_interval = list()
    step = float(1) / evidence_interval_count
    previousleft = 0
    previousright = previousleft + step
    for intervalindex in range(0, evidence_interval_count):
        currentleft = previousright
        currentright = currentleft + step
        if intervalindex == evidence_interval_count - 1:
            currentright = 1 + 1e-3
        previousleft = currentleft
        previousright = currentright
        evidence_interval.append([currentleft, currentright])
    return evidence_interval

def init_evidence(features, evidence_interval, observed_variables_set):
    '''
    Initialize the evidence_interval attribute and evidence_count attribute for all features.
    '''
    for feature in features:
        evidence_count = 0
        intervals = [set(), set(), set(), set(), set(), set(), set(), set(), set(), set()]
        weight = feature['weight']
        feature['evidence_interval'] = intervals
        for kv in weight.items():
            if kv[0] in observed_variables_set:
                for interval_index in range(0, len(evidence_interval)):
                    if kv[1][1] >= evidence_interval[interval_index][0] and kv[1][1] < evidence_interval[interval_index][1]:
                        feature['evidence_interval'][interval_index].add(kv[0])
                        evidence_count += 1
        feature['evidence_count'] = evidence_count

def write_labeled_var_to_evidence_interval(variables, features, var_id, evidence_interval):
    '''
    Update the evidence_interval attribute for each feature after labeling a variable.
    '''
    var_index = var_id
    feature_set = variables[var_index]['feature_set']
    for kv in feature_set.items():
        for interval_index in range(0, len(evidence_interval)):
            if kv[1][1] >= evidence_interval[interval_index][0] and kv[1][1] < evidence_interval[interval_index][1]:
                features[kv[0]]['evidence_interval'][interval_index].add(var_id)
                features[kv[0]]['evidence_count'] += 1

def entropy(probability):
    '''
    Calculate entropy given a probability.
    Input: a single probability or a list of probabilities.
    Output: a single entropy or a list of entropies.
    '''
    if type(probability) == np.float64 or type(probability) == np.float32 or type(probability) == float or type(probability) == int:
        if math.isinf(probability) == True:
            return probability
        else:
            if probability <= 0 or probability >= 1:
                return 0
            else:
                return 0 - (probability * math.log(probability, 2) + (1 - probability) * math.log((1 - probability), 2))
    else:
        if type(probability) == list:
            entropy_list = []
            for each_probability in probability:
                entropy_list.append(entropy(each_probability))
            return entropy_list
        else:
            return None

def open_p(weight):
    '''
    Calculate the open probability given a weight.
    '''
    return float(1) / float(1 + math.exp(- weight))
