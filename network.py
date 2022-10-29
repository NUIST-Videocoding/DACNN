import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from pretrain_model.EfficientNet import effNet


#Multi-Scale Feature Blocks
class MSFB(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(MSFB, self).__init__()

        self.conv1_MSFB = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=2, padding=0)
        self.conv3_MSFB_s1 = nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1)
        self.conv3_MSFB_s2_1 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2, padding=1)
        self.conv3_MSFB_s2_2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2,
                                       padding=1)

    def forward(self, x):
        x1 = self.conv1_MSFB(x)
        x3 = self.conv3_MSFB_s2_1(x)
        x5 = self.conv3_MSFB_s1(self.conv3_MSFB_s2_2(x))

        output = torch.cat((x1, x3, x5), 1)
        return output


#Weight Generation Block
class WGB(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(WGB, self).__init__()

        self.conv3_s1_1 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1)
        self.conv3_s1_2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x_con_1 = self.conv3_s1_1(x)
        x_con_2 = self.conv3_s1_2(x)
        x1_mul = torch.mul(x1, torch.sigmoid(x_con_1))
        x2_mul = torch.mul(x2, torch.sigmoid(x_con_2))

        output = torch.cat((x1_mul, x2_mul), 1)
        return output





#Distortion-Aware Convolution Neural Network
class DACNN(nn.Module):
    def __init__(self):
        super(DACNN, self).__init__()

        self.ADANet = EfficientNet.from_pretrained('efficientnet-b0')
        self.ADANet.eval()

        self.SDANet = effNet().cuda()
        self.SDANet.load_state_dict(torch.load('./pretrain_model/model/EffnetPretraining.pth'))
        # EfficientNet.parameters()
        self.SDANet.eval()
        for param in self.ADANet.parameters():
            param.requires_grad = False

        for param in self.SDANet.parameters():
            param.requires_grad = False


        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.msfb11 = MSFB(inchannel=24,outchannel=24)
        self.msfb21 = MSFB(inchannel=40,outchannel=24)
        self.msfb31 = MSFB(inchannel=112,outchannel=24)
        self.msfb41 = MSFB(inchannel=1280,outchannel=24)

        self.msfb12 = MSFB(inchannel=24, outchannel=24)
        self.msfb22 = MSFB(inchannel=40, outchannel=24)
        self.msfb32 = MSFB(inchannel=112, outchannel=24)
        self.msfb42 = MSFB(inchannel=1280, outchannel=24)

        self.msfb_1 = MSFB(inchannel=144,outchannel=24)
        self.msfb_2 = MSFB(inchannel=144, outchannel=24)
        self.msfb_3 = MSFB(inchannel=144, outchannel=24)
        self.msfb_4 = MSFB(inchannel=144, outchannel=24)


        self.wgb1 = WGB(inchannel=144,outchannel=72)
        self.wgb2 = WGB(inchannel=144, outchannel=72)
        self.wgb3 = WGB(inchannel=144, outchannel=72)
        self.wgb4 = WGB(inchannel=144, outchannel=72)

        self.fc1 = nn.Linear(288, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

        # parameter initialization
        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)
        nn.init.kaiming_normal_(self.fc3.weight.data)



    def forward(self,input):
        input = input.view(-1, input.size(-3), input.size(-2), input.size(-1))
        # print('input', input.size())#[16, 3, 224, 224]

        # Authentic Distortion Features Extraction
        #a = authentic distortion
        endpoints_a = self.ADANet.extract_endpoints(input)
        a1 = endpoints_a['reduction_2']#[1, 24, 56, 56]
        a2 = endpoints_a['reduction_3']#[1, 40, 28, 28]
        a3 = endpoints_a['reduction_4']#[1, 112, 14, 14]
        a4 = endpoints_a['reduction_5']#[1, 1280, 7, 7]


        #Synthetic Distortion Features Extraction
        #s = synthetic distortion
        [s1, s2, s3, s4] = self.SDANet(input)
        # s1 = endpoints_s['reduction_2']#[1, 24, 56, 56]
        # s2 = endpoints_s['reduction_3']#[1, 40, 28, 28]
        # s3 = endpoints_s['reduction_4']#[1, 112, 14, 14]
        # s4 = endpoints_s['reduction_5']#[1, 1280, 7, 7]


        #Distortion Fusion Module
        a1_msfb = self.msfb11(a1)#[1,72,28,28]
        s1_msfb = self.msfb12(s1)#[1,72,28,28]
        as1_wgb = self.wgb1(a1_msfb,s1_msfb)#[1,144,28,28]
        as1 = self.msfb_1(as1_wgb)#[1,72,14,14]
        c1 = self.gap(as1)#[1,72,1,1]
        # print('c1',c1.size())

        a2_msfb = self.msfb21(a2)#[1, 72, 14, 14]
        s2_msfb = self.msfb22(s2)#[1, 72, 14, 14]
        as2_wgb = self.wgb2(a2_msfb, s2_msfb)#[1, 144, 14, 14]
        as2 = self.msfb_2(as2_wgb)#[1, 72, 7, 7]
        c2 = self.gap(as2)#[1, 72, 1, 1]
        # print('c2',c2.size())

        a3_msfb = self.msfb31(a3)#[1, 72, 7, 7]
        s3_msfb = self.msfb32(s3)#[1, 72, 7, 7]
        as3_wgb = self.wgb3(a3_msfb, s3_msfb)#[1, 144, 7, 7]
        as3 = self.msfb_3(as3_wgb)#[1, 72, 4, 4]
        c3 = self.gap(as3)#[1, 72, 1, 1]
        # print('c3',c3.size())

        a4_msfb = self.msfb41(a4)#[1, 72, 4, 4]
        s4_msfb = self.msfb42(s4)#[1, 72, 4, 4]
        as4_wgb = self.wgb4(a4_msfb, s4_msfb)#[1, 144, 4, 4]
        as4 = self.msfb_4(as4_wgb)#1, 72, 2, 2]
        c4 = self.gap(as4)#[1, 72, 1, 1]
        # print('c4',c4.size())

        #score prediction module
        full=torch.cat((c1,c2,c3,c4),1)
        full = full.squeeze(3).squeeze(2)
        q = self.prelu1(self.fc1(full))
        q = F.dropout(q)
        q = self.prelu2(self.fc2(q))
        q = self.fc3(q)

        return q


#network test
if __name__ == '__main__':
    net = DACNN().cuda()
    input = torch.tensor(torch.randn((1, 3, 224, 224))).cuda()
    output = net(input)
    # print(output)
