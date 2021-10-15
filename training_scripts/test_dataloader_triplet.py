#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:05:38 2019
    - Dataloader for triplet of graph-data.
    - trainset only includes ids that has valid positive-pairs (iou > threshold.) e.g. 0.6
    - randomly sample anchors [same as previous]
    - for selected anchor, find an positive from positive set (randomly choose if multiple exits)
    - To find the negative,
        1) randomly choose any image except from the pos list
        2) only choose images whose iou is beteen some range (l_iou, h_iou) --> (0.2-0.4)
           The higher the iou the harder is the negative.
        3) Only choose hard examples just below the postive threshold, and above some iou e.g. (0.4-0.7))
@author: dipu
"""

import torch
from torch.utils.data import Dataset
import torch.utils.data as data
import os
from PIL import Image

from torchvision import transforms
import numpy as np
import random
import pickle
import torch.nn.functional as F
from collections import defaultdict
import random


def default_loader(path):
    return Image.open(path).convert('RGB')


def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)
        print('Saved to {}.'.format(fname))


def pickle_load(fname):
    with open(fname, 'rb') as pf:
        data = pickle.load(pf)
        print('Loaded {}.'.format(fname))
        return data


# %%
class test_RICO_TripletDataset(Dataset):
    def default_loader(path):
        return Image.open(path).convert('RGB')

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, if_shuffle=(split == 'train'))
        self.iterators[split] = 0

    def __init__(self, config):
        self.config = config

        self.info = pickle.load(
            open('/gruvi/usr/akshay/1-FPs/7-FP_Metric/GCN_CNN_scripts/data/FP_box_info_list.pkl', 'rb'))
        # self.Channel_img_dir = self.config.Channel25_img_dir
        # self.img_dir = self.config.img_dir

        self.sg_geometry_dir = '/gruvi/usr/akshay/1-FPs/7-FP_Metric/GCN_CNN_data/graph_data/geometry-directed/'
        print('\nLoading geometric graphs and features from {}\n'.format(self.sg_geometry_dir))

        self.batch_size = self.config.batch_size
        # self.transform = transform
        self.loader = default_loader

        # self.com2index = get_com2index()
        self.geometry_relation = True
        self.geom_feat_size = 8

        # % get the anchor-positive-negative apn_dict dictionary
        self.apn_dict = pickle_load(self.config.apn_dict_path)
        # %%
        train_uis = list(self.apn_dict.keys())

        # Separate out indexes for the train and test
        UI_data = pickle.load(open("/gruvi/usr/akshay/1-FPs/7-FP_Metric/GCN_CNN_scripts/data/FP_data.p", "rb"))
        orig_train_uis = UI_data['train_fps']
        gallery_uis = UI_data['val_fps']

        UI_test_data = pickle.load(
            open("/gruvi/usr/akshay/1-FPs/7-FP_Metric/GCN_CNN_scripts/data/FP_test_data.p", "rb"))
        query_uis = UI_test_data['query']
        # gallery_uis = UI_test_data['gallery_uis']


        # Remove '.png' extension for ease
        orig_train_uis = [x.rsplit('/', 1)[1].replace('.png', '') for x in orig_train_uis]
        query_uis = [x.rsplit('/', 1)[1].replace('.png', '') for x in query_uis]
        gallery_uis = [x.rsplit('/', 1)[1].replace('.png', '') for x in gallery_uis]

        # Donot use the images with large number of components.
        # uis_ncomponent_g100 = pickle.load(open('data/ncomponents_g100_imglist.pkl', 'rb'))
        # self.orig_train_uis = list(set(orig_train_uis) & set([x['id'] for x in self.info]))  #some img (e.g. img with no comp are removed in info)
        self.orig_train_uis = orig_train_uis
        # self.orig_train_uis = orig_train_uis
        # self.orig_train_uis = list(set(self.orig_train_uis) - set(uis_ncomponent_g100))

        # train_uis = list(set(train_uis) - set(uis_ncomponent_g100))

        # Instantiate the ix
        self.split_ix = {'train': [], 'gallery': [], 'query': []}

        # id2index: the dataset is indexed with indicies of the list info:
        self.id2index = defaultdict(dict)

        for ix in range(len(self.info)):
            img = self.info[ix]['id']
            self.id2index[img] = ix
            if img in train_uis:  # and img not in uis_ncomponent_g100 :
                self.split_ix['train'].append(ix)
            elif img in query_uis:  # and img not in uis_ncomponent_g100:
                self.split_ix['query'].append(ix)
            elif img in gallery_uis:  # and img not in uis_ncomponent_g100:
                self.split_ix['gallery'].append(ix)
                # else:
                #   raise Exception('image is not in the original list')

        self.split_ix['train'] = self.split_ix['train'][7700:]  # use only 7700 samples for training

        self.iterators = {'train': 0, 'query': 0, 'gallery': 0}

        for split in self.split_ix.keys():
            print('assigned %d images to split %s' % (len(self.split_ix[split]), split))

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train', num_workers=4)

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        #        ix = index #self.split_ix[index]

        sg_data = self.get_graph_data(index)
        # image_id = self.info[index]['id']

        '''
        if self.config.use_25_images:
            # c_img = self.get_classwise_channel_image(index) #transform/resize this later
            channel25_path = os.path.join(self.Channel_img_dir, image_id + '.npy' )
            img = np.load(channel25_path)
            img = torch.tensor(img.astype(np.float32))
        else:
            img_name = os.path.join(self.img_dir, str(image_id) +'.png' )
            img = self.loader(img_name)
            img = self.transform(img)
        '''

        # return (sg_data,img,index)
        return (sg_data, index)

    def get_pairs(self, ix):
        id_a = self.info[ix]['id']
        pos_pool = self.apn_dict[id_a]['ids_pos']

        random.seed(4)
        c_ind = random.choice(range(len(pos_pool)))
        id_p = pos_pool[c_ind]

        # iou_p = self.anc_pos_dict[id_a]['pos_ious'][c_ind]
        iou_p_norm = self.apn_dict[id_a]['ious_pos'][c_ind]

        '''
        if self.config.hardmining:
            #print('hard_negative mining')
            neg_pool = self.apn_dict[id_a]['ids_b2040']
            if len(neg_pool) == 0:
                ids_b5060 = self.apn_dict[id_a]['ids_b5060']
                ids_b4050 = self.apn_dict[id_a]['ids_b4050']
                ids_iou1 = self.apn_dict[id_a]['ids_iou1']
                # sample from any image except pos, anchor-itself, and ids with iou between [0.4-0.6]
                neg_pool = list(set(self.orig_train_uis) - set(pos_pool) - set([id_a]) - set(ids_iou1))
        else:
            ids_b5060 = self.apn_dict[id_a]['ids_b5060']
            ids_b4050 = self.apn_dict[id_a]['ids_b4050']
            ids_iou1 = self.apn_dict[id_a]['ids_iou1']
            # sample from any image except pos, anchor-itself, and ids with iou between [0.4-0.6]
            neg_pool = list(set(self.orig_train_uis) - set(pos_pool) - set([id_a]) - set(ids_b5060) - set(ids_b4050) -  set(ids_iou1))
        '''

        neg_pool = self.apn_dict[id_a]['ids_b2040'] + self.apn_dict[id_a]['ids_b4050']
        #neg_pool = self.apn_dict[id_a]['ids_b5060']
        
        if len(neg_pool) != 0:
            random.seed(9)
            random.shuffle(neg_pool)

            random.seed(8)
            cn_ind = random.choice(range(len(neg_pool)))
            id_n = neg_pool[cn_ind]
            iou_n_norm = 0  # Need to implement this, may be useful for hard negative

            # Hard negative say ids with ious between 20-50:
            #        ids_b2040 = self.apn_dict[id_a]['ids_b2040']  # Note: these are not norm_iou
            #        ids_b4050 = self.apn_dict[id_a]['ids_b4050']
            #        neg_pool = ids_b2040 + ids_b4050
            #        id_n = random.choice(neg_pool)
            #        iou_n_norm = 0


            
            triplet_text_file_path = 'triplet_id_testing.txt'
            list1 = [id_a + ', ', id_p + ', ', id_n + '\n']
            triplets = ''.join(list1)
            with open(triplet_text_file_path, 'a') as f:
                f.write(triplets)          
            
            return id_a, id_p, id_n, iou_p_norm, iou_n_norm

        else:
            return id_a, id_p, 0,  iou_p_norm, 0



    def get_graph_data(self, index):
        # self.config.use_box_feats = True
        image_id = self.info[index]['id']
        #        sg_use = np.load(self.sg_data_dir + image_id + '.npy', encoding='latin1', allow_pickle=True)[()]

        geometry_path = os.path.join(self.sg_geometry_dir, image_id + '.npy')
        rela = np.load(geometry_path, allow_pickle=True)[()]  # dict contains keys of edges and feats

        obj = self.info[index]['class_id']
        obj = np.reshape(obj, (-1, 1))

        box = self.info[index]['xywh']

        if self.config.use_box_feats:
            box_feats = self.get_box_feats(box)
            sg_data = {'obj': obj, 'box_feats': box_feats, 'rela': rela, 'box': box}
        else:
            sg_data = {'obj': obj, 'rela': rela, 'box': box}

        return sg_data

    def get_graph_data_by_id(self, image_id):
        # combines get_graph_data & getitem functions.
        geometry_path = os.path.join(self.sg_geometry_dir, image_id + '.npy')
        rela = np.load(geometry_path, allow_pickle=True)[()]  # dict contains keys of edges and feats

        index = self.id2index[image_id]
        assert (image_id == self.info[index]['id'])

        obj = self.info[index]['class_id']
        obj = np.reshape(obj, (-1, 1))
        #one_hot_encoded_obj = self.class_labels_to_one_hot(obj)
    
        box = self.info[index]['xywh']                
        
        if self.config.use_box_feats:
            box_feats = self.get_box_feats(box)
            #box_feats = np.concatenate((box_feats, one_hot_encoded_obj), axis=-1)
            sg_data = {'obj': obj, 'box_feats': box_feats, 'rela': rela, 'box':box}

        else:
            sg_data = {'obj': obj,  'rela': rela, 'box':box}

        '''
        if self.config.use_25_images:
            if self.config.use_precomputed_25Chan_imgs:
                channel25_path = os.path.join(self.Channel_img_dir, image_id + '.npy' )
                img = np.load(channel25_path)    
                img = torch.tensor(img.astype(np.float32))
            else:
                img = self.get_classwise_channel_image(index) #transform/resize this later
        else:
            img_name = os.path.join(self.img_dir, str(image_id) +'.png' )
            img = self.loader(img_name)
            img = self.transform(img)
        '''
   
        #return (sg_data,img,index)
        return (sg_data, index)



    def get_box_feats(self, box):
        boxes = np.array(box)
        x1, y1, w, h = np.hsplit(boxes, 4)
        x2, y2 = x1 + w, y1 + h

        W, H = 256, 256  # We know the height and weight for all semantic UIs are 2560 and 1400
        '''
        x_min = min([x[0] for x in x1])
        x_max = max([x[0] for x in x2])

        y_min = min([y[0] for y in y1])
        y_max = max([y[0] for y in y2])

        W = x_max - x_min
        H = y_max - y_min
        '''

        box_feats = np.hstack((0.5 * (x1 + x2) / W, 0.5 * (y1 + y2) / H, w / W, h / H, w * h / (W * H)))
        # box_feats = box_feat / np.linalg.norm(box_feats, 2, 1, keepdims=True)
        return box_feats

    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size
        sg_batch_a = []
        sg_batch_p = []
        sg_batch_n = []

        #        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')
        infos = []

        # images_a = []
        # images_p = []
        # images_n = []

        wrapped = False

        for i in range(batch_size):
            # fetch image
            # tmp_sg_a, tmp_img_a, ix_a, tmp_wrapped = self._prefetch_process[split].get()
            tmp_sg_a, ix_a, tmp_wrapped = self._prefetch_process[split].get()

            # id_p, id_n, iou_p, iou_p_norm, iou_n, iou_n_norm  = self.get_pairs(ix_a)
            id_a, id_p, id_n, iou_p_norm, iou_n_norm = self.get_pairs(ix_a)
            if id_n == 0:
                continue

            '''
            tmp_sg_p, tmp_img_p, ix_p = self.get_graph_data_by_id(id_p)
            tmp_sg_n, tmp_img_n, ix_n = self.get_graph_data_by_id(id_n)
            '''

            #tmp_sg_a_new, ix_a_new = self.get_graph_data_by_id(id_a)
            tmp_sg_p, ix_p = self.get_graph_data_by_id(id_p)
            tmp_sg_n, ix_n = self.get_graph_data_by_id(id_n)

            sg_batch_a.append(tmp_sg_a)
            # images_a.append(tmp_img_a)

            sg_batch_p.append(tmp_sg_p)
            # images_p.append(tmp_img_p)

            sg_batch_n.append(tmp_sg_n)
            # images_n.append(tmp_img_n)



            # record associated info as well
            info_dict = {}
            info_dict['ix_a'] = ix_a
            info_dict['id_a'] = self.info[ix_a]['id']
            info_dict['id_p'] = id_p
            info_dict['id_n'] = id_n
            # info_dict['iou_p'] = iou_p
            # info_dict['iou_n'] = iou_n
            info_dict['iou_p_norm'] = iou_p_norm
            info_dict['iou_n_norm'] = iou_n_norm

            infos.append(info_dict)

            if tmp_wrapped:
                wrapped = True
                break

        data = {}

        '''
        max_box_len_a = max([_['obj'].shape[0] for _ in sg_batch_a])
        max_box_len_p = max([_['obj'].shape[0] for _ in sg_batch_p])
        max_box_len_n = max([_['obj'].shape[0] for _ in sg_batch_n])
        '''

        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        data['sg_data_a'] = self.batch_sg(sg_batch_a)  # , max_box_len_a)
        data['sg_data_p'] = self.batch_sg(sg_batch_p)  # , max_box_len_p)
        data['sg_data_n'] = self.batch_sg(sg_batch_n)  # , max_box_len_n)

        # data['images_a'] = torch.stack(images_a)
        # data['images_p'] = torch.stack(images_p)
        # data['images_n'] = torch.stack(images_n)

        return data

    def batch_sg(self, sg_batch):  # , max_box_len):
        "batching object, attribute, and relationship data"
        obj_batch = [_['obj'] for _ in sg_batch]
        rela_batch = [_['rela'] for _ in sg_batch]
        #box_batch = [_['box'] for _ in sg_batch]

        sg_data = []
        for i in range(len(obj_batch)):
            sg_data.append(dict())

        '''
        # obj labels, shape: (B, No, 1)
        sg_data['obj_labels'] = []
        for i in range(len(obj_batch)):
            sg_data['obj_labels'].append(obj_batch[i])
        sg_data['obj_labels'] = np.array(sg_data['obj_labels'])


        sg_data['obj_boxes'] = []
        for i in range(len(box_batch)):
            sg_data['obj_boxes'].append(box_batch[i])
        sg_data['obj_boxes'] = np.array(sg_data['obj_boxes'])
        '''

        if self.config.use_box_feats:
            box_feats_batch = [_['box_feats'] for _ in sg_batch]
            # sg_data['box_feats'] = []
            for i in range(len(box_feats_batch)):
                sg_data[i]['box_feats'] = box_feats_batch[i]
                sg_data[i]['room_ids'] = obj_batch[i]

            for i in range(len(rela_batch)):
                sg_data[i]['rela_edges'] = rela_batch[i]['edges']
                sg_data[i]['rela_feats'] = rela_batch[i]['feats']

                # sg_data['box_feats'] = np.array(sg_data['box_feats'])
        '''
        # rela
        sg_data['rela_edges'] = []
        sg_data['rela_feats'] = []

        for i in range(len(rela_batch)):
            sg_data['rela_edges'].append(rela_batch[i]['edges'])
            sg_data['rela_feats'].append(rela_batch[i]['feats'])

        sg_data['rela_edges'] = np.array(sg_data['rela_edges'])
        sg_data['rela_feats'] = np.array(sg_data['rela_feats'])
        '''

        return sg_data


# %%
class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False, num_workers=4):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        #        self.config =config
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle
        self.num_workers = num_workers

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                                 batch_size=1,
                                                 sampler=SubsetSampler(self.dataloader.split_ix[self.split][
                                                                       self.dataloader.iterators[self.split]:]),
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers,  # 1, # 4 is usually enough
                                                 worker_init_fn=None,
                                                 collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.seed(6)
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[-1] == ix, "ix not equal"

        return tmp + [wrapped]


# %%
class SubsetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
        # return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)