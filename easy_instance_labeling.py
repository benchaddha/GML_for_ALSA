# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# easy_instance_labeling.py
#
# Identification: /easy_instance_labeling.py
#
# Ben Chaddha
#
# ===----------------------------------------------------------------------===//

class EasyInstanceLabeling:
    '''
    Class for easy instance labeling, providing methods for marking easy instances.
    Returns easys.
    '''

    def __init__(self, variables, features, easys=None):
        self.variables = variables
        self.features = features
        self.easys = easys

    def label_easy_by_file(self):
        '''
        Mark the variables as easy based on the provided easy list.
        '''
        if self.easys is not None and isinstance(self.easys, list):
            for var in self.variables:
                var['is_easy'] = False
                var['is_evidence'] = False

            for easy in self.easys:
                var_index = easy['var_id']
                self.variables[var_index]['is_easy'] = True
                self.variables[var_index]['is_evidence'] = True

    def label_easy_by_clustering(self, easy_proportion=0.3):
        '''
        Mark easy instances using clustering.
        '''
        pass

    def label_easy_by_custom(self):
        '''
        Mark easy instances using custom logic.
        '''
        pass
