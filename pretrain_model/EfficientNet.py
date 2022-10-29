import torch
from torch import nn
from efficientnet_pytorch import EfficientNet




class effNet(nn.Module):
    def __init__(self):
        super(effNet, self).__init__()
        self.efficient = EfficientNet.from_name('efficientnet-b0')

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.eff = nn.Sequential()
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, data):
        # print('x', x.shape)#[32, 3, 224, 224]
        # print('x_ref', x_ref.shape)#[32, 3, 224, 224]

        #pretraining code
        # [x,x_ref] = data
        # endpoints_x = self.efficient.extract_endpoints(x)
        # endpoints_x_ref = self.efficient.extract_endpoints(x_ref)
        # x = endpoints_x['reduction_5']
        # x_ref = endpoints_x_ref['reduction_5']
        # temp = x - x_ref
        # temp = self.avgpool(temp)
        # temp = torch.flatten(temp, 1)
        # q = torch.nn.functional.relu(self.fc1(temp))
        # # q = torch.nn.functional.dropout(q)
        # q = torch.nn.functional.relu(self.fc2(q))
        # q = self.fc3(q)
        # return q


        endpoints_x = self.efficient.extract_endpoints(data)
        feature_red2 = endpoints_x['reduction_2']
        feature_red3 = endpoints_x['reduction_3']
        feature_red4 = endpoints_x['reduction_4']
        feature_red5 = endpoints_x['reduction_5']
        return [feature_red2, feature_red3, feature_red4, feature_red5]



