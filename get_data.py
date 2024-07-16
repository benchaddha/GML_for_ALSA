# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# get_data.py
#
# Benjamin Chaddha
#
# ===----------------------------------------------------------------------===//

import pickle
from copy import copy
import random


# Load data from pickle file
with open('data/all_global_parse_fg.pkl', 'rb') as f:
    all_datas = pickle.load(f)

# Initialize sets for word-type features
word_feature_set = set()    # word-type features
single_word_feature_set = set()    # word-type features with only one variable
multi_word_feature_set = set()    # word-type features with multiple variables

# Iterate through data and populate feature sets
for item in all_datas[3]:
    if item.name1 not in word_feature_set:
        word_feature_set.add(item.name1)
    else:
        multi_word_feature_set.add(item.name1)

# Filter single-word features
single_word_feature_set = [x for x in word_feature_set if x not in multi_word_feature_set]

# Create a mapping table for variable IDs
var_id_map = dict()   # maintain an ID mapping table, e.g., {'B0074703CM_102_ANONYMOUS:6:0': 0}
for id, item in enumerate(all_datas[0]):
    var_id_map[item.name] = id

# Process features
features = list()
feature = dict()
weight_elem = dict()
feature_id = 0

# Process relation-type features first
for rel_type in ['asp2asp_sequence_oppo', 'asp2asp_intrasent_simi', 'asp2asp_sequence_simi']:
    feature['feature_id'] = feature_id
    feature['feature_type'] = 'binary_feature'
    feature['feature_name'] = rel_type
    for item in all_datas[2]:
        if item.rel_type == rel_type:
            key = (var_id_map[item.name1], var_id_map[item.name2])
            weight_value = [-2.0, rel_type] if item.rel_type == 'asp2asp_sequence_oppo' else [2.0, rel_type]  # temporarily set feature value as rel_type
            weight_elem[key] = weight_value
    feature['weight'] = copy(weight_elem)
    features.append(copy(feature))
    feature_id += 1
    weight_elem.clear()
    feature.clear()

# Process word-type features that appear only once
for name in single_word_feature_set:
    for item in all_datas[3]:
        if name == item.name1:
            feature['feature_id'] = feature_id
            feature['feature_name'] = item.rel_type
            feature['feature_type'] = 'unary_feature'
            key = var_id_map[item.name2]
            weight_value = [2.0, item.name1]  # temporarily set feature value as the specific word
            weight_elem[key] = weight_value
            feature['weight'] = copy(weight_elem)
            features.append(copy(feature))
            feature_id += 1
            weight_elem.clear()
            feature.clear()

# Process word-type features that appear multiple times
for name in multi_word_feature_set:
    feature['feature_id'] = feature_id
    feature['feature_type'] = 'unary_feature'
    for item in all_datas[3]:
        if name == item.name1:
            key = var_id_map[item.name2]
            weight_value = [2.0, item.name1]  # temporarily set feature value as the word
            weight_elem[key] = weight_value
    feature['feature_name'] = item.rel_type
    feature['weight'] = copy(weight_elem)
    features.append(copy(feature))
    feature_id += 1
    weight_elem.clear()
    feature.clear()

# Organize variables
variables = list()
variable = dict()
feature_set = dict()
for id, item in enumerate(all_datas[0]):
    variable['var_id'] = id
    variable['is_evidence'] = item.isEvidence
    if item.polarity is None:
        variable['is_easy'] = False
        variable['is_evidence'] = False
        variable['label'] = random.choice((1, 0))   # if it's hard, randomly initialize the label
    else:
        variable['is_easy'] = True
        variable['is_evidence'] = True
        variable['label'] = 1 if item.polarity == 'positive' else 0  # if it's easy, set the label as the easy label
    variable['true_label'] = 1 if item.gold_polarity == 'positive' else 0
    variable['prior'] = item.prior
    for feature in features:
        for kv in feature['weight'].items():
            if type(kv[0]) == tuple and id in kv[0]:
                feature_set[feature['feature_id']] = [0, kv[1][1]]
            elif id == kv[0]:
                feature_set[feature['feature_id']] = [0, kv[1][1]]
    variable['feature_set'] = copy(feature_set)
    variables.append(copy(variable))
    variable.clear()
    feature_set.clear()

# Save variables and features to pickle files
with open('data/variables.pkl', 'wb') as v:
    pickle.dump(variables, v)
with open('data/features.pkl', 'wb') as f:
    pickle.dump(features, f)
