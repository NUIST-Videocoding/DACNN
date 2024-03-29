import torch
from scipy import stats
import numpy as np

from network import DACNN

import dataloader.dataLoder as data_loader

class IQASolver(object):
    """training and testing"""
    def __init__(self, config, path, train_idx, test_idx, train_round):
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        self.config = config
        self.train_round = train_round

        self.model = DACNN().cuda()
        self.model.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()
        self.lr = config.lr 
        self.lrratio = config.lr_ratio 
        self.weight_decay = config.weight_decay


        paras = [{'params': self.model.parameters(), 'lr': self.lr * self.lrratio}]

        self.optimizer = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset,
                                              path,
                                              train_idx,
                                              config.patch_size,
                                              config.train_patch_num,
                                              batch_size=config.batch_size,
                                              istrain=True)
        test_loader = data_loader.DataLoader(config.dataset,
                                             path,
                                             test_idx,
                                             config.patch_size,
                                             config.test_patch_num,
                                             istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0

        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for img, label in self.train_data:

                img = torch.as_tensor(img.cuda())
                label = torch.as_tensor(label.cuda())

                self.optimizer.zero_grad()

                pred = self.model(img)

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)

            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                # torch.save(self.model.state_dict(),
                #            "./models/DACNN_"+ str(self.config.dataset) + "_" + str(self.train_round) + ".pth")
                
            print('%d\t\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

         
            lr = (self.lr * 9) / pow(10, (t // 10))
            self.paras = [{'params': self.model.parameters(), 'lr': lr}]
            self.optimizer = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model.train(False)
        pred_scores = []
        gt_scores = []

        for img, label in data:
            img = torch.as_tensor(img.cuda())
            label = torch.as_tensor(label.cuda())

            pred = self.model(img)

            pred_scores.append(float(pred.item()))
            gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model.train(True)
        return test_srcc, test_plcc
