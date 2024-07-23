# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# helper.py
#
# Benjamin Chaddha
#
# ===----------------------------------------------------------------------===//



from collections import Counter
from copy import copy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
def enum(**enums):
    return type('Enum', (), enums)


class EasyInstanceLabeling:
    # Enum for label types
    labeltypes = enum(EASY0='EASY0', EASY1='EASY1')

    '''
    Clustering to obtain pairs labeled as 0 and 1
    values = Global.datasetup.pairs.values
    monotonyeffectivemetafeaturecolumnindexes = Global.datasetup.monotonyeffectivemetafeaturecolumnindexes
    '''

    @staticmethod
    def clustering(values, monotonyeffectivemetafeaturecolumnindexes):
        pairs = values
        metafeaturefilterindexlist = monotonyeffectivemetafeaturecolumnindexes
        x_input = np.array(pairs)[:, metafeaturefilterindexlist].astype(np.float32)
        y_label = np.array(pairs)[:, 1].astype(np.float32)
        random_state = 170
        km_model = KMeans(n_clusters=2, random_state=random_state).fit(x_input)
        y_pred = km_model.labels_
        cnt = Counter(y_pred)
        smallgroup = min(cnt[0], cnt[1])
        biggroup = max(cnt[0], cnt[1])
        cc = km_model.cluster_centers_
        all_point_distances = euclidean_distances(cc, x_input)
        minority = min(cnt.values())
        label_remap = dict()
        for k, v in cnt.items():
            if v == minority:
                label_remap[k] = 1
            else:
                label_remap[k] = -1
        _id_2_cluster_label = dict()
        label0id2probability = {}
        label1id2probability = {}
        pair_ids = np.array(pairs)[:, 0].astype(str)
        for i in range(0, len(pair_ids)):
            _assigned_label = label_remap.get(y_pred[i])
            _id_2_cluster_label[pair_ids[i]] = _assigned_label
            _denominator = 0
            for elem in all_point_distances:
                _denominator += elem[i]
            _numerator = all_point_distances[y_pred[i]][i]
            _cluster_pro = 1.0 - 1.0 * _numerator / _denominator
            if _assigned_label == 1:
                _match_pro = _cluster_pro
                label1id2probability[pair_ids[i]] = _match_pro
            else:
                _match_pro = 1 - _cluster_pro
                label0id2probability[pair_ids[i]] = _match_pro
        label1id2probabilitylist = sorted(label1id2probability.items(), key=lambda x: x[1], reverse=True)
        label0id2probabilitylist = sorted(label0id2probability.items(), key=lambda x: x[1], reverse=False)
        return label0id2probabilitylist, label1id2probabilitylist

    '''
    totalpairs_list = Global.totalpairs_list
    easyproportion = parasetups.easyproportion
    '''

    @staticmethod
    def easy_instance_labeling(totalpairs_list, easyproportion, values,
                               monotonyeffectivemetafeaturecolumnindexes):
        label0id2probabilitylist, label1id2probabilitylist = EasyInstanceLabeling.clustering(values,
                                                                                             monotonyeffectivemetafeaturecolumnindexes)
        totalpairslist = sorted(totalpairs_list, key=lambda x: x.similarity, reverse=False)
        easy0count = int(len(label0id2probabilitylist) * easyproportion)
        easy1count = int(len(label1id2probabilitylist) * easyproportion)
        currenteasypair = None
        pair_list = []
        label_list = []
        for index in range(len(totalpairslist) - 1, len(totalpairslist) - 1 - easy1count, -1):
            currenteasypair = totalpairslist[index]
            pair_list.append(currenteasypair.pid)
            correct, truthlabel, label = currenteasypair.tolabel(EasyInstanceLabeling.labeltypes.EASY1, None, True)
            label_list.append(label)
        for index in range(0, easy0count):
            currenteasypair = totalpairslist[index]
            pair_list.append(currenteasypair.pid)
            correct, truthlabel, label = currenteasypair.tolabel(EasyInstanceLabeling.labeltypes.EASY0, None, True)
            label_list.append(label)
        dataframe = pd.DataFrame({'id': pair_list, 'label': label_list})
        dataframe.to_csv("easy_instance.csv", index=False, sep=',')

    def tolabel(self, labeltype, probability=None, report=True):
        if self.label == None:
            originlabeltype = self.labeltype
            self.labeltype = labeltype
        if self.labeltype == EasyInstanceLabeling.labeltypes.EASY0:
            probability = 0
            self.label = 0
        else:
            if self.labeltype == EasyInstanceLabeling.labeltypes.EASY1:
                probability = 1
                self.label = 1
        return None, self.truthlabel, self.label

    '''
        Load easy instances from file
        Read a CSV file containing all easy instances
        Return the Id and label of the easy instances

        The content format of the CSV file is as follows:
        id represents the name of the easy pair, label 1 indicates a match, 0 indicates a non-match
        id,label
        "conf/sigmod/2000,219042",0
        "conf/sigmod/2001,219542",1
        "conf/sigmod/2002,219342",0
        "conf/sigmod/2003,219342",0
        "conf/sigmod/2005,219742",1
    '''

    @staticmethod
    def load_easy_instance_from_file(filename):
        easy_data = pd.read_csv(filename)
        easy_pair = {'var_id': 0, 'label': 1}
        easy_pair_list = []
        for i in range(len(easy_data)):
            easy_pair['var_id'] = easy_data['id'][i]
            easy_pair['label'] = easy_data['label'][i]
            easy_pair_list.append(copy(easy_pair))
        return easy_pair_list


'''
if __name__ == '__main__':
    easy_pair_list = EasyInstanceLabeling.load_easy_instance_from_file('varid_easy_instance.csv')
    print(len(easy_pair_list))
    print(easy_pair_list)
'''
