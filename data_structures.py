# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# data_structures.py
#
# Identification: /data_structures.py
#
# Ben Chaddha
#
# ===----------------------------------------------------------------------===//
'''
Variables
variables = list()   # Elements are of type dict: variable
variable
{
  # User-provided attributes
  'var_id': 1,                  # Variable ID, starting from 0 (int)
  'is_easy': False,            # Whether it is an Easy variable
  'is_evidence': True,          # Whether it is an evidence variable
  'true_label': 1,              # True label for subsequent accuracy calculation
  'label': 1,                   # Inferred label: 0 for negative, 1 for positive, -1 for unknown
  'feature_set':                # All features that this variable has
  { 
    feature_id1: [theta1, feature_value1],
    feature_id2: [theta2, feature_value2],
    ...
  },
  # Attributes that may be generated during code execution
  'probability': 0.99,          # Inferred probability
  'evidential_support': 0.3,     # Evidential support
  'entropy': 0.4,               # Entropy
  'approximate_weight': 0.3,     # Approximate weight
   ...
}

Features
features = list()      # Elements are of type dict: feature
feature
{
  # User-provided attributes
  'feature_id': 1,                                       # Feature ID, starting from 0 (int)
  'feature_type': unary_feature/binary_feature,         # Distinguishes whether the feature is unary or binary, currently supports unary_feature and binary_feature
  'feature_name': good,                                 # Feature name, if it is a token type, it is the specific word of the token, if it is another type of feature, it is the specific type of the feature
  'weight':                                             # Collection of all variables that have this feature
  {
    var_id1:        [weight_value1, feature_value1],     # unary_feature
   (varid3, varid4): [weight_value2, feature_value2],     # binary_feature
    ...
  }
  # Attributes that may be generated during code execution
  'tau': 0,
  'alpha': 0,
  'regression': object,
  ...
}
'''

