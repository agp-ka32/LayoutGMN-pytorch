#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:19:45 2020


from the normalized iou_dict, separate out ids, iou_norm for
    -    > 0.7 (0.7 is set as the threshold for selecting the positives.) 
    - 0.6-0.7 
    - 0.5-0.6 
    - 0.4-0.5

Also ids between ious 0.2-0.4

Note:
Many of the images will be filtered out from anchors (30-40%)
However, note that the other images in the train set can occur as negative sets...

Changed the pos threhold to >0.6.
@author: dipu
"""

import pickle
from collections import defaultdict
import random 

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)
        print('Saved to {}.'.format(fname))

def pickle_load(fname):
    with open(fname, 'rb') as pf:
         data = pickle.load(pf)
         print('Loaded {}.'.format(fname))
         return data

iou_dict = pickle_load('../../GCN_CNN_data/data/Triplets/iouValues_combined_13K.pkl')
#iou_dict = pickle_load('../../GCN_CNN_data/data/Triplets/iouValues_combined_2K.pkl')

iou_dict = dict(iou_dict)

apn_dict = defaultdict(dict)

for key in iou_dict.keys():
    temp = iou_dict[key]
    ids_b2040 = temp['ids_b40']
    ious_g40_norm = temp['ious_g40']
    max_val = max(ious_g40_norm)

    ious_g40_norm = ious_g40_norm/max_val
    ids_g40 = temp['ids_g40']
    
    sel_pos = [x for x in zip(ious_g40_norm,ids_g40) if x[0] != 1 and x[0] >= 0.6] # avoid the same image (iou_norm =1)
    sel_iou1 = [x for x in zip(ious_g40_norm,ids_g40) if x[0] == 1 ]
   
    #sel_b6070 = [x for x in zip(ious_g40_norm,ids_g40) if x[0] >= 0.6 and x[0] < 0.7] 
    sel_b5060 = [x for x in zip(ious_g40_norm,ids_g40) if x[0] >= 0.5 and x[0] < 0.6]
    sel_b4050 = [x for x in zip(ious_g40_norm,ids_g40) if x[0] >= 0.4 and x[0] < 0.5]

    sel_neg = [x for x in zip(ious_g40_norm, ids_g40) if x[0] >= 0.2 and x[0] < 0.4]

    if len(sel_pos) != 0:
        sel_pos_ious, sel_pos_ids = zip(*sel_pos)
        apn_dict[key]['ious_pos'] = list(sel_pos_ious)
        apn_dict[key]['ids_pos'] = list(sel_pos_ids)
        
        if len(sel_iou1)  != 0:
            sel_iou1_ious, sel_iou1_ids = zip(*sel_iou1)
            apn_dict[key]['ious_iou1'] = list(sel_iou1_ious)
            apn_dict[key]['ids_iou1'] = list(sel_iou1_ids)
        else:
            #sel_iou1_ious, sel_iou1_ids = zip(*sel_iou1)
            apn_dict[key]['ious_iou1'] = [] #list(sel_iou1_ious)
            apn_dict[key]['ids_iou1'] = [] #list(sel_iou1_ids)
        
#        if len(sel_b6070) != 0:
#            sel_b6070_ious, sel_b6070_ids = zip(*sel_b6070)
#            apn_dict[key]['ious_b6070'] = list(sel_b6070_ious)
#            apn_dict[key]['ids_b6070'] = list(sel_b6070_ids)
#        else:
#            apn_dict[key]['ious_b6070'] = []
#            apn_dict[key]['ids_b6070'] = []
            
        
        if len(sel_b5060) != 0:
            sel_b5060_ious, sel_b5060_ids = zip(*sel_b5060)
            apn_dict[key]['ious_b5060'] = list(sel_b5060_ious)
            apn_dict[key]['ids_b5060'] = list(sel_b5060_ids)
        else:
            apn_dict[key]['ious_b5060'] = []
            apn_dict[key]['ids_b5060'] = []
            
        
        if len(sel_b4050) != 0:
            sel_b4050_ious, sel_b4050_ids = zip(*sel_b4050)
            apn_dict[key]['ious_b4050'] = list(sel_b4050_ious)
            apn_dict[key]['ids_b4050'] = list(sel_b4050_ids)
        else:
            apn_dict[key]['ious_b4050'] = []
            apn_dict[key]['ids_b4050'] = []

        #if len(sel_neg) != 0:
            #sel_neg_ious, sel_neg_ids = zip(*sel_neg)
            #apn_dict[key]['ious_b2040'] = list(sel_neg_ious)
            #apn_dict[key]['ids_b2040'] = list(sel_neg_ids)
        #else:
            #apn_dict[key]['ious_b2040'] = []
            #apn_dict[key]['ids_b2040'] = ids_b2040
        apn_dict[key]['ids_b2040'] = ids_b2040
    else:
         continue
     
print('length of selected: {}/{}'.format(len(apn_dict), len(iou_dict)))
        
pickle_save('../../GCN_CNN_data/data/Triplets/apn_dict_13K_pthres60.pkl', apn_dict)
#pickle_save('../../GCN_CNN_data/data/Triplets/apn_dict_2K_pthres60.pkl', apn_dict)