from torch.optim.lr_scheduler import StepLR

import copy
import time
from time import gmtime, strftime
from datetime import datetime

from dynamicplot import DynamicPlot
from dataloader_graph import data_input_to_gmn
from dataloader_triplet import RICO_TripletDataset
from test_dataloader_triplet import test_RICO_TripletDataset
from combine_all_modules_6 import * # this imports the util file


#####################################################
########### Some helper functions ###################
def set_lr2(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group['lr'] = group['lr'] * decay_factor
    print('\n', optimizer, '\n')


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

################ End of helper functions ##############
#######################################################

def _main(config):
    
    if config.cuda and torch.cuda.is_available():
        print('Using CUDA on GPU', config.gpu)
    else:
        print('Not using CUDA')

    device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else "cpu")
    print(device)

    print('Initializing the model..........')

    stored_epoch = '380'
    if config.load_pretrained == False:
        print('No pretrained models loaded')
        gmn_model = gmn_net

    else:
        print('Loading pretrained models from:  ')
        
        print(config.model_save_path + 'gmn_tmp_model' + stored_epoch + '.pkl')
        gmn_model = gmn_net

        gmn_model_state_dict = torch.load(
            config.model_save_path + 'gmn_tmp_model' + stored_epoch + '.pkl')

        from collections import OrderedDict

        def remove_module_fromStateDict(model_state_dict, model):
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k[0:]  # remove 'module.'
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            return model

        gmn_model = remove_module_fromStateDict(gmn_model_state_dict, gmn_model)
        print('Finished loading saved models')

    '''
    if torch.cuda.device_count() > 1 and config.cuda:
        print('Using', torch.cuda.device_count(), 'GPUs!')
        gmn_model = nn.DataParallel(gmn_model, device_ids=[0,1])  # , output_device=device)
    '''

    #gmn_model.to(f'cuda:{gmn_model.device_ids[0]}')
    gmn_model.to(device)
    #gmn_model.cuda()

    gmn_model_params = list(gmn_model.parameters())
    optimizer = torch.optim.Adam(gmn_model_params, lr=config.lr)
    #scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    header = '  Time      Epoch   Iter    GVS    s_pos   sm_neg   s_diff     Loss'
    log_template = ' '.join('{:>9s},{:>4.0f}/{:<4.0f},{:<5.0f},{:>6.4f},{:>7.5f},{:>7.5f},{:>7.5f}, {:>10.7f}'.split(','))

    iteration = 0
    epoch = 0

    gmn_model.train()
    torch.set_grad_enabled(True)
    start = time.time()
    loader = RICO_TripletDataset(config)

    
    while True:
        data =  loader.get_batch('train')#.to(device)

        sg_data_a = data['sg_data_a']
        sg_data_p = data['sg_data_p']
        sg_data_n = data['sg_data_n']

        GraphData = data_input_to_gmn(config, device, sg_data_a, sg_data_p, sg_data_n).quadruples()
        #GraphData = data_input_to_gmn(sg_data_a, sg_data_p, sg_data_n).quadruples()


        graph_vectors = gmn_model(**GraphData)#.cuda()
        # print(graph_vectors)
        x1, y, x2, z = reshape_and_split_tensor(graph_vectors, 4)
        loss = triplet_loss(x1, y, x2, z, loss_type=config.loss_type, margin=config.margin_val)
        sim_pos = torch.mean(compute_similarity(config, x1, y))
        sim_neg = torch.mean(compute_similarity(config, x2, z))
        sim_diff = sim_pos - sim_neg

        graph_vec_scale = torch.mean(graph_vectors ** 2)

        if config.graph_vec_regularizer_weight > 0:
            loss += config.graph_vec_regularizer_weight * 0.5 * graph_vec_scale

        if config.cuda:
            total_batch_loss = loss.sum().to(device)

        total_batch_loss = loss.sum().to(device)

        optimizer.zero_grad()
        total_batch_loss.backward()
        clip_gradient(optimizer, config.clip_val)
        optimizer.step()
        torch.cuda.empty_cache()
        iteration += 1

        if epoch == 0 and iteration == 1:
            print("Training Started ")
            print(header)

        if iteration % 100 == 0:
            elsp_time = (time.time() - start)
            print(log_template.format(strftime(u"%H:%M:%S", time.gmtime(time.time() - start)),
                                      epoch, config.epochs, iteration, graph_vec_scale,
                                      sim_pos, sim_neg, sim_diff, total_batch_loss.item()))

            '''
            with open(config.model_save_dir + '/log.txt', 'a') as f:
                f.write('Epoch [%02d] [%05d / %05d  ] Average_Loss: %.5f  Recon Loss: %.4f  DML Loss: %.4f\n' % (
                epoch + 1, iteration * opt.batch_size, len(loader), losses.avg, losses_recon.avg, losses_dml.avg))
                f.write('Completed {} images in {}'.format(iteration * opt.batch_size, elsp_time))
            '''
            #print('Completed {} images in {}'.format(iteration * config.batch_size, elsp_time))
            #start = time.time()


        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True
            iteration = 0


        if (epoch + 1) % config.save_network_every == 0 and epoch_done:
            os.makedirs(config.model_save_path, exist_ok=True)
            # os.makedirs(config.feature_save_path, exist_ok=True)
            try:
                if config.load_pretrained == False:
                    # save models
                    torch.save(gmn_model.state_dict(),
                               config.model_save_path + 'gmn_tmp_model' + str(epoch + 1) + '.pkl')
                    #print('temporary models saved!')
                else:
                    torch.save(gmn_model.state_dict(),
                               config.model_save_path + 'gmn_tmp_model' + str(
                                   int(stored_epoch) + epoch + 1) + '.pkl')
                    #print('temporary models saved!')

            except:
                print('failed to save temp models')
                raise

            # set model to training mode again after it had been in eval mode
            #gmn_model.train()
            #torch.set_grad_enabled(True)


        if epoch > 700:
            break



def perform_tests(test_config, loaded_gmn_model, loader_test, save_dir, ep):
    '''
    :param test_config: config file with parameters usied during test time
    :param loaded_gmn_model: loaded pre-trained network model, in your case Graph Matching Network
    :param loader_test: object carrying test dataset
    :param save_dir: path to the accuracy is saved
    :param ep: int, epoch number
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else "cpu")
     
    loaded_gmn_model.eval() # set the model in evaluation mode
    torch.set_grad_enabled(False) # no backward propagation of loss, no parameter update
   
    save_file = save_dir + 'result.txt'

    epoch_done = False
    wrongly_predicted_ids = []
    #fnames = []

    correct_cnt = 0
    c = 0
    file_cnt = 0
    
    while epoch_done == False:
        
        t1 = time.time()
        data =  loader_test.get_batch('train') # although it is 'train', the number of samples has already been split, and only the test samples have been loaded
        #fnames += [x['id'] for x in data['infos']]

        sg_data_a = data['sg_data_a']
        sg_data_p = data['sg_data_p']
        sg_data_n = data['sg_data_n']

        file_cnt += 1
        if len(sg_data_a) == 0 or len(sg_data_p) == 0 or len(sg_data_n) == 0:
            #print('Some issue with {} triplet'.format(file_cnt))
            continue

        GraphData = data_input_to_gmn(test_config, device, sg_data_a, sg_data_p, sg_data_n).quadruples()

        #gmn_model = gmn_net
        #gmn_model_params = list(gmn_model.parameters())

        graph_vectors = loaded_gmn_model(**GraphData)#.cuda()
        t2 = time.time()
        print(t2-t1)
        exit()
        # print(graph_vectors)
        x1, y, x2, z = reshape_and_split_tensor(graph_vectors, 4)
        sim_pos = torch.mean(compute_similarity(config, x1, y))
        sim_neg = torch.mean(compute_similarity(config, x2, z))
        sim_diff = sim_pos - sim_neg

        if sim_diff > 0:
            correct_cnt += 1
        else:
            wrongly_predicted_ids.append(c)


        c += 1

        if data['bounds']['wrapped']:
                #print('Extracted features from {} images from {} split'.format(c, split))
                epoch_done = True

        #print('Extracted features from {} images from {} split'.format(len(fnames), 'train, but test'))


    total_cnt = c #len(fnames)
    print('total number of images is:', total_cnt)
    print('correctly identified triplets are:', correct_cnt)

    accuracy = 100 *(correct_cnt / total_cnt)
    print('ep: {}\n'.format(ep))
    print('Accuracy = {}\n\n'.format(accuracy))
          
    with open(save_file, 'a') as f:
        f.write('ep: {}\n'.format(ep))
        f.write('Accuracy = {}\n\n'.format(accuracy))

    np.savetxt('wrongly_predicted_triplets.txt', wrongly_predicted_ids, delimiter='\n', fmt='%s')



def load_pretrained_model(gmn_model, save_dir, stored_epoch):
    '''
    :param gmn_model: network model
    :param save_dir: path of the dir where the models have been saved
    :param stored_epoch: str, ex: '8'
    '''
    print('Loading pretrained models')
    gmn_model_state_dict = torch.load(
    save_dir + 'gmn_tmp_model' + stored_epoch + '.pkl')

    from collections import OrderedDict

    def remove_module_fromStateDict(model_state_dict, model):
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[0:]  # if ran on two GPUs, remove 'module.module.'; else, no change
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return model

    loaded_gmn_model = remove_module_fromStateDict(gmn_model_state_dict, gmn_model)
    print('Finished loading checkpoint')
    return loaded_gmn_model


#os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
if __name__ == '__main__':
    config = get_args()

    assert config.train_mode != config.eval_mode
    if config.train_mode and not config.eval_mode:
        _main(config)

    if config.eval_mode and not config.train_mode:
        gmn_model = gmn_net
        #gmn_model_params = list(gmn_model.parameters())
        save_dir = 'trained_models/'

        test_config = copy.deepcopy(config)
        test_config.batch_size = 1
        loader_test = test_RICO_TripletDataset(test_config) #load Triplet dataset for testing

        #device = 'cpu'
        for epoch in range(410, 411, 10):

            string_epoch = str(epoch)
            loaded_gmn_model = load_pretrained_model(gmn_model, save_dir, string_epoch)
            if torch.cuda.is_available() and config.cuda:
                loaded_gmn_model.cuda()
           

            loaded_model_params = loaded_gmn_model.parameters()
            #evaluate_function
            perform_tests(test_config, loaded_gmn_model, loader_test, save_dir, epoch)