import torch
from scipy import stats
import numpy as np

from pretrain_model.LSIQAdataloader import DataLoader
from pretrain_model.EfficientNet import effNet

class Solver(object):
    def __init__(self, config, path, train_idx, test_idx):
        self.epochs = config.epochs
        self.model = effNet().cuda()
        self.model.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()
        self.lr = config.lr
        self.lrratio = config.lr_ratio#10
        self.weight_decay = config.weight_decay

        paras = [{'params': self.model.parameters(), 'lr': self.lr * self.lrratio}]

        self.optimizer = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = DataLoader(path, train_idx,
                                              config.patch_size,
                                              batch_size=config.batch_size,
                                              istrain=True)
        test_loader = DataLoader(path, test_idx,
                                             config.patch_size,
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
            for dst_img, ref_img, label in self.train_data:
                dst_img = dst_img.cuda()
                ref_img = ref_img.cuda()
                label = np.array(label)
                label = label.astype(np.float32)
                label = torch.from_numpy(label).cuda()
                # print(dst_img.size())#[128, 3, 224, 224]
                # print(ref_img.size())#[128, 3, 224, 224]
                # print(label.shape)#[16]

                self.optimizer.zero_grad()

                pred = self.model([dst_img, ref_img])
                # print(pred.size())#[16, 1]

                pred_scores = pred_scores + pred.tolist()
                gt_scores = gt_scores + label.tolist()

                # print('pred.squeeze()', pred.squeeze().size())#torch.Size([16])
                # print('label.detach()', label.float().detach().size())#torch.Size([16])

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            # print(train_srcc)

            test_srcc, test_plcc = self.test(self.test_data)
            if test_srcc > best_srcc and t + 1 >= 250:
                best_srcc = test_srcc
                best_plcc = test_plcc
                torch.save(self.model.state_dict(), "./model/EffnetPretaining.pth")

            print('%d\t\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

            # Update optimizer
            lr = self.lr / pow(10, (t // 200))
            if t > 500:
                self.lrratio = 1
            self.paras = [{'params': self.model.parameters(), 'lr': lr * self.lrratio}]
            self.optimizer = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model.train(False)
        pred_scores = []
        gt_scores = []

        for dst_img, ref_img, label in data:
            dst_img = dst_img.cuda()
            ref_img = ref_img.cuda()
            label = np.array(label)
            label = label.astype(np.float32)
            label = torch.from_numpy(label).cuda()

            pred = self.model([dst_img, ref_img])

            pred_scores.append(float(pred.item()))
            gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model.train(True)
        return test_srcc, test_plcc
