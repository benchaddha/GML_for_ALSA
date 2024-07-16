# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# evidence_select.py
#
# Identification: /evidence_select.py
#
# Ben Chaddha
#
# ===----------------------------------------------------------------------===//

import logging
import random



# Provide some methods for selecting evidence
class EvidenceSelect:
    def __init__(self, variables, features, interval_evidence_count=10, subgraph_limit_num=1000, k_hop=2):
        self.variables = variables
        self.features = features
        self.subgraph_limit_num = subgraph_limit_num
        self.k_hop = k_hop
        self.interval_evidence_count = interval_evidence_count

    def select_evidence_by_interval(self, var_id):
        '''
        Select a certain number of evidence variables for the specified latent variable based on the interval of feature_value, suitable for ER.
        Currently, each feature is divided into interval_evidence_count intervals, and no more than interval_evidence_count variables are selected for each interval.
        Input: 
            var_id -- latent variable id
            interval_evidence_count -- number of evidence variables to select for each interval
        Output:
            connected_var_set -- set of evidence variable ids
            connected_edge_set -- set of edges
            connected_feature_set -- set of useful features
        '''
        connected_var_set = set()
        connected_edge_set = set()
        connected_feature_set = set()  # Record which features are actually retained when building the factor graph on this latent variable
        feature_set = self.variables[var_id]['feature_set']
        
        # Add edges between evidence variables and features first
        for feature_id in feature_set.keys():
            if self.features[feature_id]['evidence_count'] > 0:  # Some features may not be connected to evidence variables, so they don't need to be added
                connected_feature_set.add(feature_id)
                evidence_interval = self.features[feature_id]['evidence_interval']
                for interval in evidence_interval:
                    # If the number of evidence variables in this interval is less than 200, add all of them
                    if len(interval) <= self.interval_evidence_count:
                        connected_var_set = connected_var_set.union(interval)
                        for id in interval:
                            connected_edge_set.add((feature_id, id))
                    else:
                        # If it is greater than 200, randomly sample 200
                        sample = random.sample(list(interval), self.interval_evidence_count)
                        connected_var_set = connected_var_set.union(sample)
                        for id in sample:
                            connected_edge_set.add((feature_id, id))
        
        # Add edges between this latent variable and features
        connected_var_set.add(var_id)
        for feature_id in connected_feature_set:
            connected_edge_set.add((feature_id, var_id))
        
        logging.info("var-" + str(var_id) + " select evidence by interval finished")
        return connected_var_set, connected_edge_set, connected_feature_set

    def select_evidence_by_realtion(self, var_id_list):
        '''Select evidence for the top_k latent variables, suitable for ALSA.
        Input:
            var_id_list --- list of k variable ids
            subgraph_limit_num -- maximum number of variables allowed in the subgraph
            k_hop -- number of hops to find adjacent variables
        Output:
            connected_var_set -- set of evidence variable ids
            connected_edge_set -- set of edges
            connected_feature_set -- set of useful features
        '''
        if type(var_id_list) == set() or type(var_id_list) == list():
            subgraph_limit_num = self.subgraph_limit_num
            k_hop = self.k_hop
            connected_var_set = set()
            connected_edge_set = set()
            connected_feature_set = set()  # Record which features are actually retained when building the factor graph on this latent variable
            connected_var_set = connected_var_set.union(set(var_id_list))
            current_var_set = connected_var_set
            next_var_set = set()
            
            # First find evidence variables that are k-hop away from relation-type features (determine whether only evidence variables are added here, excluding latent variables)
            for k in range(k_hop):
                for var_id in current_var_set:
                    feature_set = self.variables[var_id]['feature_set']
                    for feature_id in feature_set.keys():
                        if self.features[feature_id]['feature_type'] == 'binary_feature':
                            weight = self.features[feature_id]['weight']
                            for id in weight.keys():
                                if type(id) == tuple and var_id in id:
                                    another_var_id = id[0] if id[0] != var_id else id[1]
                                    if self.variables[another_var_id]['is_evidence'] == True:
                                        next_var_set.add(another_var_id)
                                        connected_feature_set.add(feature_id)
                                        connected_edge_set.add((feature_id, id))
                    connected_var_set = connected_var_set.union(next_var_set)
                    current_var_set = next_var_set
                    next_var_set.clear()
            
            # Then find variables that share word-type features with these k variables (add evidence variables first, if it does not exceed the maximum variable limit, then add latent variables)
            subgraph_capacity = subgraph_limit_num - len(connected_var_set)
            unary_connected_unlabeled_var = list()
            unary_connected_unlabeled_edge = list()
            unary_connected_unlabeled_feature = list()
            unary_connected_evidence_var = list()
            unary_connected_evidence_edge = list()
            unary_connected_evidence_feature = list()
            
            for var_id in var_id_list:
                feature_set = self.variables[var_id]['feature_set']
                for feature_id in feature_set.keys():
                    if self.features[feature_id]['feature_type'] == 'unary_feature':
                        weight = self.features[feature_id]['weight']
                        for id in weight.keys():
                            if self.variables[id]['is_evidence'] == True:
                                unary_connected_evidence_var.append(id)
                                unary_connected_evidence_feature.append(feature_id)
                                unary_connected_evidence_edge.append((feature_id, id))
                            else:
                                unary_connected_unlabeled_var.append(id)
                                unary_connected_unlabeled_feature.append(feature_id)
                                unary_connected_unlabeled_edge.append((feature_id, id))
            
            # Limit the size of the subgraph
            if(len(unary_connected_evidence_var) <= subgraph_capacity ):
                connected_var_set = connected_var_set.union(set(unary_connected_evidence_var))
                connected_feature_set = connected_feature_set.union((set(unary_connected_evidence_feature)))
                connected_edge_set = connected_edge_set.union(set(unary_connected_evidence_edge))
                if(len(unary_connected_unlabeled_var) <= (subgraph_capacity-len(unary_connected_evidence_var))):
                    connected_var_set = connected_var_set.union(set(unary_connected_unlabeled_var))
                    connected_feature_set = connected_feature_set.union((set(unary_connected_unlabeled_feature)))
                    connected_edge_set = connected_edge_set.union(set(unary_connected_unlabeled_edge))
                else:
                    connected_var_set = connected_var_set.union(set(unary_connected_unlabeled_var[:subgraph_capacity-len(unary_connected_evidence_var)]))
                    connected_feature_set = connected_feature_set.union(set(unary_connected_unlabeled_feature[:subgraph_capacity-len(unary_connected_evidence_var)]))
                    connected_edge_set = connected_edge_set.union(set(unary_connected_unlabeled_edge[:subgraph_capacity-len(unary_connected_evidence_var)]))
            else:
                connected_var_set = connected_var_set.union(set(unary_connected_evidence_var[:subgraph_capacity]))
                connected_feature_set = connected_feature_set.union((set(unary_connected_evidence_feature[:subgraph_capacity])))
                connected_edge_set = connected_edge_set.union(set(unary_connected_evidence_edge[:subgraph_capacity]))
            
            logging.info("select evidence by relation finished")
            return connected_var_set, connected_edge_set, connected_feature_set
        else:
            print('Parameter error, input should be a list of K ids')

    def select_evidence_by_custom(self, var_id):
        # Users can implement this function themselves
        connected_var_set = set()
        connected_edge_set = set()
        connected_feature_set = set()  # Record which features are actually retained when building the factor graph on this latent variable
        # Override
        return connected_var_set, connected_edge_set, connected_feature_set
