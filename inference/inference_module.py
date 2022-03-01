import os
import numpy as np
import sys
import time

sys.path.insert(1,'../')
from util import get_args
from collections import defaultdict

from dataloader_graph import data_input_to_gmn
from train_TRI import load_pretrained_model
from combine_all_modules_6 import gmn_net, reshape_and_split_tensor, compute_similarity

import torch
import torch.nn as nn
import pickle

'''
class get_sg_data(object):
    def __init__(self, fp_path, config, info):
        super(get_sg_data, self).__init__()
        self.fp_path = fp_path
        self.config = config

        self.sg_geometry_dir = '../updated-geometry-directed/'

        self.info = info

        self.id2index = defaultdict(dict)
        for ix in range(len(self.info)):
            img = self.info[ix]['id']
            self.id2index[img] = ix

        #self._main()


    def get_graph_data_by_id(self, image_id):
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

        new_sg_data = {}
        new_sg_data['box_feats'] = sg_data['box_feats']
        new_sg_data['room_ids'] = sg_data['obj']
        new_sg_data['rela_edges'] = sg_data['rela']['edges']
        new_sg_data['rela_feats'] = sg_data['rela']['feats']

        return new_sg_data


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


    def _main(self):
        img_id = self.fp_path.rsplit('/', 1)[1][:-4]
        sg_data = self.get_graph_data_by_id(img_id)

        return sg_data


class compute_and_sort_fp(object):

    def __init__(self, query_list_txt_file, db_list_txt_file, config, loaded_gmn_model, device):
        '''

        :param query_list_txt_file: path of the text file containing query FPs
        :param db_list_txt_file: path of the text file containing FPs in the DB list
        :param config: config file with all the parameters
        :param loaded_gmn_model: pre-trained GMN model
        '''

        super(compute_and_sort_fp, self).__init__()

        with open(query_list_txt_file, 'r') as f:
            self.query_list = f.read().splitlines()

        #self.query_list = self.query_list[53:]

        with open(db_list_txt_file, 'r') as f:
            self.db_list = f.read().splitlines()

        self.config = config
        self.loaded_gmn_model = loaded_gmn_model
        self.device = device

        self.info = pickle.load(
            open('../box_info_list.pkl', 'rb')) # path to your pickle file


    def get_triplet_graph_data(self, query_fp, db_fp_1, db_fp_2):

        query_sg_data = []
        query_sg_data.append(get_sg_data(query_fp, self.config, self.info)._main())

        db_fp_1_sg_data = []
        db_fp_1_sg_data.append(get_sg_data(db_fp_1, self.config, self.info)._main())

        db_fp_2_sg_data = []
        db_fp_2_sg_data.append(get_sg_data(db_fp_2, self.config, self.info)._main())

        device = self.device
        Graph_Data_dict = data_input_to_gmn(self.config, device,
                                       query_sg_data, db_fp_1_sg_data, db_fp_2_sg_data).quadruples()

        return Graph_Data_dict


    def get_sim_values(self, Graph_Data_dict):

        graph_vecs = self.loaded_gmn_model(**Graph_Data_dict)

        x1, y, x2, z = reshape_and_split_tensor(graph_vecs, 4)

        sim_1 = torch.mean(compute_similarity(self.config, x1, y))
        sim_2 = torch.mean(compute_similarity(self.config, x2, z))

        return sim_1, sim_2


    def sort_sim_values(self, sim_val_list):
        indices = np.argsort(sim_val_list).tolist()

        similar_fp_list = [self.db_list[i] for i in indices]

        return similar_fp_list


    def _main(self):
        cnt = 0
        for file in self.query_list:
            print(cnt)
            cnt += 1
            sim_val_list = []
            query_fp = file

            for i in range(0, 10, 2): #len(self.db_list)-1, 2):
                #print(i)
                db_fp_1 = self.db_list[i]
                db_fp_2 = self.db_list[i+1]

                Graph_Data_dict = self.get_triplet_graph_data(query_fp, db_fp_1, db_fp_2)
                sim_1, sim_2 = self.get_sim_values(Graph_Data_dict)

                #sim_1 = sim_1.cpu().detach().numpy()
                sim_val_list.append(-(sim_1.cpu().detach().numpy()))
                sim_val_list.append(-(sim_2.cpu().detach().numpy()))

            similar_fp_list = self.sort_sim_values(sim_val_list)
            query_fp_id = query_fp.rsplit('/', 1)[1][:-4]

            np.savetxt(str(query_fp_id)+'_retrievals.txt', similar_fp_list, delimiter='\n', fmt='%s')
            print('Finished saving retrievals for {} file'.format(cnt))
'''

####################### Batched retrieval code from here ##################################################

class get_batch_sg_data(object):
    def __init__(self, query, db_list, batch_size, config, info):
        super(get_batch_sg_data, self).__init__()
        self.query = query
        self.db_list = db_list


        self.batch_size = batch_size
        self.config = config

        self.sg_geometry_dir = '../updated-geometry-directed/'

        self.info = info

        self.id2index = defaultdict(dict)
        for ix in range(len(self.info)):
            img = self.info[ix]['id']
            self.id2index[img] = ix


    def get_graph_data_by_id(self, image_id):
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
        new_sg_data = {}
        new_sg_data['box_feats'] = sg_data['box_feats']
        new_sg_data['room_ids'] = sg_data['obj']
        new_sg_data['rela_edges'] = sg_data['rela']['edges']
        new_sg_data['rela_feats'] = sg_data['rela']['feats']

        return new_sg_data
        '''

        return sg_data


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


    def get_batch(self, batch_cnt):
        sg_batch_query = []
        sg_batch_fp_1 = []
        sg_batch_fp_2 = []

        start_val = (self.batch_size*2)*batch_cnt

        assert self.batch_size % 2 == 0

        for i in range(self.batch_size):
            try:
                query_id = self.query.rsplit('/', 1)[1][:-4]
                fp_1_id = self.db_list[start_val + i*2].rsplit('/', 1)[1][:-4]
                fp_2_id = self.db_list[start_val + i*2 + 1].rsplit('/', 1)[1][:-4]

                query_sg = self.get_graph_data_by_id(query_id)
                fp1_sg = self.get_graph_data_by_id(fp_1_id)
                fp2_sg = self.get_graph_data_by_id(fp_2_id)

                sg_batch_query.append(query_sg)
                sg_batch_fp_1.append(fp1_sg)
                sg_batch_fp_2.append(fp2_sg)
            except:
                break

        data = {}
        data['sg_data_a'] = self.batch_sg(sg_batch_query)  # , max_box_len_a)
        data['sg_data_p'] = self.batch_sg(sg_batch_fp_1)  # , max_box_len_p)
        data['sg_data_n'] = self.batch_sg(sg_batch_fp_2)  # , max_box_len_n)

        return data


    def batch_sg(self, sg_batch):
        "batching object, attribute, and relationship data"
        obj_batch = [_['obj'] for _ in sg_batch]
        rela_batch = [_['rela'] for _ in sg_batch]
        # box_batch = [_['box'] for _ in sg_batch]

        sg_data = []
        for i in range(len(obj_batch)):
            sg_data.append(dict())

        if self.config.use_box_feats:
            box_feats_batch = [_['box_feats'] for _ in sg_batch]
            # sg_data['box_feats'] = []
            for i in range(len(box_feats_batch)):
                sg_data[i]['box_feats'] = box_feats_batch[i]
                sg_data[i]['room_ids'] = obj_batch[i]

            for i in range(len(rela_batch)):
                sg_data[i]['rela_edges'] = rela_batch[i]['edges']
                sg_data[i]['rela_feats'] = rela_batch[i]['feats']

        return sg_data



class batched_compute_and_sort_fp(object):

    def __init__(self, query_list_txt_file, db_list_txt_file, config, loaded_gmn_model, device):
        '''

        :param query_list_txt_file: path of the text file containing query FPs
        :param db_list_txt_file: path of the text file containing FPs in the DB list
        :param config: config file with all the parameters
        :param loaded_gmn_model: pre-trained GMN model
        :param device: whether to use cpu or gpu
        '''

        super(batched_compute_and_sort_fp, self).__init__()

        with open(query_list_txt_file, 'r') as f:
            self.query_list = f.read().splitlines()
        self.query_list = self.query_list[95:]        

        with open(db_list_txt_file, 'r') as f:
            self.db_list = f.read().splitlines()
        self.db_list = self.db_list[:-1] # making it even numbered list

        self.config = config
        self.loaded_gmn_model = loaded_gmn_model
        self.device = device

        self.info = pickle.load(
            open('../box_info_list.pkl', 'rb')) #change the path to your pickle file



    def sort_sim_values(self, sim_val_list):
        indices = np.argsort(sim_val_list).tolist()

        similar_fp_list = [self.db_list[i] for i in indices]

        return similar_fp_list


    def _main(self, retrieval_batch_size):
        cnt = 0

        for file in self.query_list[0:]:
            print(cnt)
            cnt += 1
            batch_cnt = 0
            sim_val_list_1 = []
            sim_val_list_2 = []
            query_fp = file

            max_batch_cnt = int(len(self.db_list) / (retrieval_batch_size*2)) + 1

            start_time = time.time()

            while batch_cnt < max_batch_cnt:
                #print(batch_cnt)
                data = get_batch_sg_data(query_fp, self.db_list, retrieval_batch_size,
                                         self.config, self.info).get_batch(batch_cnt)

                sg_data_a = data['sg_data_a']
                sg_data_fp_1 = data['sg_data_p']
                sg_data_fp_2 = data['sg_data_n']

                Graph_Data_dict = data_input_to_gmn(config, self.device,
                                                    sg_data_a, sg_data_fp_1, sg_data_fp_2).quadruples()
                graph_vecs = self.loaded_gmn_model(**Graph_Data_dict)
                x1, y, x2, z = reshape_and_split_tensor(graph_vecs, 4)

                sim_1 = compute_similarity(self.config, x1, y) # these are now list of tensors \n;
                # previously there was torch.mean at the start of the RHS
                sim_2 = compute_similarity(self.config, x2, z) # list of tensors

                sim_val_list_1.append(-(sim_1.cpu().detach().numpy()))
                sim_val_list_2.append(-(sim_2.cpu().detach().numpy()))

                batch_cnt += 1
            sim_val_list_1 = np.concatenate(sim_val_list_1, axis=0)
            sim_val_list_2 = np.concatenate(sim_val_list_2, axis=0)

            assert len(sim_val_list_1) == len(sim_val_list_2)

            all_sim_val_list = [None]*(len(sim_val_list_1) + len(sim_val_list_2))
            all_sim_val_list[::2] = sim_val_list_1
            all_sim_val_list[1::2] = sim_val_list_2

            similar_fp_list = self.sort_sim_values(all_sim_val_list)
            query_fp_id = query_fp.rsplit('/', 1)[1][:-4]

            save_folder = 'results/'
            os.makedirs(save_folder, exist_ok=True)

            np.savetxt(save_folder+str(query_fp_id)+'_retrievals.txt', similar_fp_list, delimiter='\n', fmt='%s')
            print('Finished saving retrievals for {} file'.format(cnt))

            end_time = time.time()
            print('Time taken for one file is {} minutes'.format((end_time- start_time)/60))

############################ End of Batched retrieval Code ########################################################




if __name__ == '__main__':

    config = get_args()
    retrieval_batch_size = 2048

    batched_retr = True

    gmn_model = gmn_net
    save_dir = '../trained_models/'
    stored_epoch = 'xx' # xx is the epoch number of the saved model

    loaded_gmn_model = load_pretrained_model(gmn_model, save_dir, stored_epoch)

    if config.cuda and torch.cuda.is_available():
        print('Using CUDA on GPU', config.gpu)
    else:
        print('Not using CUDA')

    device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else "cpu")
    print(device)

    #if torch.cuda.device_count() > 1 and config.cuda:
        #print('Using', torch.cuda.device_count(), 'GPUs!')
        #loaded_gmn_model = nn.DataParallel(gmn_model, device_ids=[0, 1])  # , output_device=device)

    #loaded_gmn_model.to(f'cuda:{loaded_gmn_model.device_ids[0]}')
    loaded_gmn_model.to(device)
    loaded_gmn_model.eval()  # set the model in evaluation mode
    torch.set_grad_enabled(False)

    query_list_txt_file = 'query_list.txt'
    #db_list_txt_file = 'database_list.txt'
    db_list_txt_file = '../filtered_fp_files.txt'

    if batched_retr:
        print('Batched Retrievals')
        my_obj = batched_compute_and_sort_fp(query_list_txt_file, db_list_txt_file, config, loaded_gmn_model, device)
        my_obj._main(retrieval_batch_size)
    else:
        print('Not batched Retrievals')
        my_obj = compute_and_sort_fp(query_list_txt_file, db_list_txt_file, config, loaded_gmn_model, device)
        my_obj._main()
