from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing
from torch.optim import lr_scheduler
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from utils import *
from utils.grad_batch import *
from tqdm import tqdm
from utils.get_results_for_all_score_normalizations import get_results_for_all_score_normalizations


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        self.args = args

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        
        if len(self.args.data_path_list)>0:

            data_set, data_loader = data_provider(self.args, flag, self.args.trace_list)
        else:
            data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1.e-6)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            llh_loss = []
            kl_loss_list = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, :, f_dim:]


                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())


                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            torch.save(self.model.state_dict(), path + '/' + str(epoch) + '_checkpoint.pth')
 


        best_model_path = path + '/' + str(epoch) + '_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        epoch = 0
        test_labels = []


        while (epoch+self.args.cp) < self.args.train_epochs:
            epoch += self.args.cp
            self.model.load_state_dict(torch.load(
                os.path.join('./checkpoints/' + setting, str(epoch) + '_checkpoint.pth')))

            print('Computing Influence for CP:{}/{}'.format(epoch, self.args.train_epochs))

            range_list = list(range(0, train_data.train.shape[0]-self.args.seq_len, self.args.seq_len))
            range_list = np.array(range_list)

            attens_energy = []

            for (batch_x, batch_y, index) in tqdm(test_loader):
                batch = batch_x.to(self.device)
                index = index.cpu().numpy()
                if epoch==self.args.cp:
                    test_labels.append(batch_y)
                grad_x_val = grad_batch(batch, self.args.tracin_layers, self.model, self.anomaly_criterion, reconstruct_num=1)
                grad_x_val = [torch.stack(x) for x in list(zip(*grad_x_val))]
                grad_score = 0
                for iter_i in range(1):

                    '''self influence'''
                    grad_dot = [torch.diag(torch.mm(torch.flatten(val_grad, start_dim=1),
                                                    torch.flatten(val_grad, start_dim=1).transpose(0, 1)))
                                for val_grad in grad_x_val]

                    grad_dot_product = torch.mean(torch.stack(grad_dot), dim=0).detach().cpu().numpy()
                    grad_score += grad_dot_product

                test_grad_score = np.expand_dims(grad_score, axis=1)   
                test_grad_score = np.repeat(test_grad_score, self.args.seq_len, axis=1)
                attens_energy.append(test_grad_score)
                
            attens_energy = np.concatenate(attens_energy, axis=-1)
            attens_energy = attens_energy.transpose(1,0)
 
            if epoch==self.args.cp:
                test_energy = np.array(attens_energy)
                test_energy = (test_energy-test_energy.mean())/test_energy.std()
                test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
                test_labels = np.array(test_labels)

            else:
                attens_energy = np.array(attens_energy)
                test_energy = attens_energy
                test_energy += (attens_energy - attens_energy.mean()) / attens_energy.std()

            _, df_best_normalization = get_results_for_all_score_normalizations(
                attens_energy,
                test_labels,
                eval_method='point_wise'
            )

 

        print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            df_best_normalization.T[1][1],
                df_best_normalization.T[2][1], df_best_normalization.T[0][1]))
        if (self.args.data!='SMD' and self.args.data!='SMAP' and self.args.data!='MSL'):
            f = open("result_anomaly_detection.txt", 'a')
            f.write(setting + "  \n")
            f.write("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            df_best_normalization.T[1][1],
                df_best_normalization.T[2][1], df_best_normalization.T[0][1]))
            
            f.write('\n')
            f.close()
        return df_best_normalization.T[0][1], df_best_normalization.T[1][1], df_best_normalization.T[2][1] 

    
    def softmax(self, x):
        x_row_max = x.max(axis=-1)
        x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
        x = x - x_row_max
        x_exp = np.exp(x)
        x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
        softmax = x_exp / x_exp_row_sum
        return softmax

