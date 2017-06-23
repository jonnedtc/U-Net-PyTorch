import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# implementation of U-Net as described in the paper
# & padding to keep input and output sizes the same

class UNet(nn.Module):

    def __init__(self, dice=False):

        super(UNet, self).__init__()

        self.conv1_input =      nn.Conv2d(1, 64, 3, padding=1)
        self.conv1 =            nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_input =      nn.Conv2d(64, 128, 3, padding=1)
        self.conv2 =            nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_input =      nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 =            nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_input =      nn.Conv2d(256, 512, 3, padding=1)
        self.conv4 =            nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_input =      nn.Conv2d(512, 1024, 3, padding=1)
        self.conv5 =            nn.Conv2d(1024, 1024, 3, padding=1)

        self.conv6_up =         nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6_input =      nn.Conv2d(1024, 512, 3, padding=1)
        self.conv6 =            nn.Conv2d(512, 512, 3, padding=1)
        self.conv7_up =         nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7_input =      nn.Conv2d(512, 256, 3, padding=1)
        self.conv7 =            nn.Conv2d(256, 256, 3, padding=1)
        self.conv8_up =         nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8_input =      nn.Conv2d(256, 128, 3, padding=1)
        self.conv8 =            nn.Conv2d(128, 128, 3, padding=1)
        self.conv9_up =         nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9_input =      nn.Conv2d(128, 64, 3, padding=1)
        self.conv9 =            nn.Conv2d(64, 64, 3, padding=1)
        self.conv9_output =     nn.Conv2d(64, 2, 1)
        
        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def switch(self, dice):

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def forward(self, x):

        layer1 = F.relu(self.conv1_input(x))
        layer1 = F.relu(self.conv1(layer1))

        layer2 = F.max_pool2d(layer1, 2)
        layer2 = F.relu(self.conv2_input(layer2))
        layer2 = F.relu(self.conv2(layer2))

        layer3 = F.max_pool2d(layer2, 2)
        layer3 = F.relu(self.conv3_input(layer3))
        layer3 = F.relu(self.conv3(layer3))

        layer4 = F.max_pool2d(layer3, 2)
        layer4 = F.relu(self.conv4_input(layer4))
        layer4 = F.relu(self.conv4(layer4))

        layer5 = F.max_pool2d(layer4, 2)
        layer5 = F.relu(self.conv5_input(layer5))
        layer5 = F.relu(self.conv5(layer5))

        layer6 = F.relu(self.conv6_up(layer5))
        layer6 = torch.cat((layer4, layer6), 1)
        layer6 = F.relu(self.conv6_input(layer6))
        layer6 = F.relu(self.conv6(layer6))

        layer7 = F.relu(self.conv7_up(layer6))
        layer7 = torch.cat((layer3, layer7), 1)
        layer7 = F.relu(self.conv7_input(layer7))
        layer7 = F.relu(self.conv7(layer7))

        layer8 = F.relu(self.conv8_up(layer7))
        layer8 = torch.cat((layer2, layer8), 1)
        layer8 = F.relu(self.conv8_input(layer8))
        layer8 = F.relu(self.conv8(layer8))

        layer9 = F.relu(self.conv9_up(layer8))
        layer9 = torch.cat((layer1, layer9), 1)
        layer9 = F.relu(self.conv9_input(layer9))
        layer9 = F.relu(self.conv9(layer9))
        layer9 = self.final(self.conv9_output(layer9))

        return layer9


# 2D variation of VNet - similar to UNet
# added residual functions to each block
# & down convolutions instead of pooling

class VNet(nn.Module):

    def __init__(self, dice=False):

        super(VNet, self).__init__()

        self.conv1 =        nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.conv1 =        nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.conv1_down =   nn.Conv2d(16, 32, 2, stride=2, padding=0)

        self.conv2a =       nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv2b =       nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv2_down =   nn.Conv2d(32, 64, 2, stride=2, padding=0)

        self.conv3a =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv3b =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv3c =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv3_down =   nn.Conv2d(64, 128, 2, stride=2, padding=0)

        self.conv4a =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv4b =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv4c =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv4_down =   nn.Conv2d(128, 256, 2, stride=2, padding=0)

        self.conv5a =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv5b =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv5c =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv5_up =     nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)

        self.conv6a =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv6b =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv6c =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv6_up =     nn.ConvTranspose2d(256, 64, 2, stride=2, padding=0)

        self.conv7a =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv7b =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv7c =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv7_up =     nn.ConvTranspose2d(128, 32, 2, stride=2, padding=0)

        self.conv8a =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv8b =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv8_up =     nn.ConvTranspose2d(64, 16, 2, stride=2, padding=0)

        self.conv9 =        nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv9_1x1 =    nn.Conv2d(32, 2, 1, stride=1, padding=0)

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def switch(self, dice):

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def forward(self, x):

        layer1 = F.relu(self.conv1(x))
        layer1 = torch.add(layer1, torch.cat([x]*16,1))

        conv1 = F.relu(self.conv1_down(layer1))

        layer2 = F.relu(self.conv2a(conv1))
        layer2 = F.relu(self.conv2b(layer2))
        layer2 = torch.add(layer2, conv1)

        conv2 = F.relu(self.conv2_down(layer2))

        layer3 = F.relu(self.conv3a(conv2))
        layer3 = F.relu(self.conv3b(layer3))
        layer3 = F.relu(self.conv3c(layer3))
        layer3 = torch.add(layer3, conv2)

        conv3 = F.relu(self.conv3_down(layer3))

        layer4 = F.relu(self.conv4a(conv3))
        layer4 = F.relu(self.conv4b(layer4))
        layer4 = F.relu(self.conv4c(layer4))
        layer4 = torch.add(layer4, conv3)

        conv4 = F.relu(self.conv4_down(layer4))

        layer5 = F.relu(self.conv5a(conv4))
        layer5 = F.relu(self.conv5b(layer5))
        layer5 = F.relu(self.conv5c(layer5))
        layer5 = torch.add(layer5, conv4)

        conv5 = F.relu(self.conv5_up(layer5))

        cat6 = torch.cat((conv5, layer4), 1)

        layer6 = F.relu(self.conv6a(cat6))
        layer6 = F.relu(self.conv6b(layer6))
        layer6 = F.relu(self.conv6c(layer6))
        layer6 = torch.add(layer6, cat6)

        conv6 = F.relu(self.conv6_up(layer6))

        cat7 = torch.cat((conv6, layer3), 1)

        layer7 = F.relu(self.conv7a(cat7))
        layer7 = F.relu(self.conv7b(layer7))
        layer7 = F.relu(self.conv7c(layer7))
        layer7 = torch.add(layer7, cat7)

        conv7 = F.relu(self.conv7_up(layer7))

        cat8 = torch.cat((conv7, layer2), 1)

        layer8 = F.relu(self.conv8a(cat8))
        layer8 = F.relu(self.conv8b(layer8))
        layer8 = torch.add(layer8, cat8)

        conv8 = F.relu(self.conv8_up(layer8))

        cat9 = torch.cat((conv8, layer1), 1)

        layer9 = F.relu(self.conv9(cat9))
        layer9 = torch.add(layer9, cat9)
        layer9 = self.final(self.conv9_1x1(layer9))

        return layer9


# 2D variation of VNet - similar to UNet
# added residual functions to each block
# & down convolutions instead of pooling
# & batch normalization for convolutions
# & drop out before every upsample layer
# & context parameter to make it 2.5 dim

class VNet_Xtra(nn.Module):

    def __init__(self, dice=False, dropout=False, context=0):

        super(VNet_Xtra, self).__init__()

        self.dropout = dropout
        if self.dropout:
            self.do6 = nn.Dropout2d()
            self.do7 = nn.Dropout2d()
            self.do8 = nn.Dropout2d()
            self.do9 = nn.Dropout2d()

        self.conv1 =        nn.Conv2d(1 + context * 2, 16, 5, stride=1, padding=2)
        self.bn1 =          nn.BatchNorm2d(16)
        self.conv1_down =   nn.Conv2d(16, 32, 2, stride=2, padding=0)
        self.bn1_down =     nn.BatchNorm2d(32)

        self.conv2a =       nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.bn2a =         nn.BatchNorm2d(32)
        self.conv2b =       nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.bn2b =         nn.BatchNorm2d(32)
        self.conv2_down =   nn.Conv2d(32, 64, 2, stride=2, padding=0)
        self.bn2_down =     nn.BatchNorm2d(64)

        self.conv3a =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn3a =         nn.BatchNorm2d(64)
        self.conv3b =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn3b =         nn.BatchNorm2d(64)
        self.conv3c =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn3c =         nn.BatchNorm2d(64)
        self.conv3_down =   nn.Conv2d(64, 128, 2, stride=2, padding=0)
        self.bn3_down =     nn.BatchNorm2d(128)

        self.conv4a =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.bn4a =         nn.BatchNorm2d(128)
        self.conv4b =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.bn4b =         nn.BatchNorm2d(128)
        self.conv4c =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.bn4c =         nn.BatchNorm2d(128)
        self.conv4_down =   nn.Conv2d(128, 256, 2, stride=2, padding=0)
        self.bn4_down =     nn.BatchNorm2d(256)

        self.conv5a =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.bn5a =         nn.BatchNorm2d(256)
        self.conv5b =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.bn5b =         nn.BatchNorm2d(256)
        self.conv5c =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.bn5c =         nn.BatchNorm2d(256)
        self.conv5_up =     nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
        self.bn5_up =       nn.BatchNorm2d(128)

        self.conv6a =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.bn6a =         nn.BatchNorm2d(256)
        self.conv6b =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.bn6b =         nn.BatchNorm2d(256)
        self.conv6c =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.bn6c =         nn.BatchNorm2d(256)
        self.conv6_up =     nn.ConvTranspose2d(256, 64, 2, stride=2, padding=0)
        self.bn6_up =       nn.BatchNorm2d(64)

        self.conv7a =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.bn7a =         nn.BatchNorm2d(128)
        self.conv7b =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.bn7b =         nn.BatchNorm2d(128)
        self.conv7c =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.bn7c =         nn.BatchNorm2d(128)
        self.conv7_up =     nn.ConvTranspose2d(128, 32, 2, stride=2, padding=0)
        self.bn7_up =       nn.BatchNorm2d(32)

        self.conv8a =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn8a =         nn.BatchNorm2d(64)
        self.conv8b =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn8b =         nn.BatchNorm2d(64)
        self.conv8_up =     nn.ConvTranspose2d(64, 16, 2, stride=2, padding=0)
        self.bn8_up =       nn.BatchNorm2d(16)

        self.conv9 =        nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.bn9 =          nn.BatchNorm2d(32)
        self.conv9_1x1 =    nn.Conv2d(32, 2, 1, stride=1, padding=0)
        self.bn9_1x1 =      nn.BatchNorm2d(2)

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def switch(self, dice):

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def forward(self, x):

        layer1 = F.relu(self.bn1(self.conv1(x)))
        layer1 = torch.add(layer1, torch.cat([x[:,0:1,:,:]]*16,1))

        conv1 = F.relu(self.bn1_down(self.conv1_down(layer1)))

        layer2 = F.relu(self.bn2a(self.conv2a(conv1)))
        layer2 = F.relu(self.bn2b(self.conv2b(layer2)))
        layer2 = torch.add(layer2, conv1)

        conv2 = F.relu(self.bn2_down(self.conv2_down(layer2)))

        layer3 = F.relu(self.bn3a(self.conv3a(conv2)))
        layer3 = F.relu(self.bn3b(self.conv3b(layer3)))
        layer3 = F.relu(self.bn3c(self.conv3c(layer3)))
        layer3 = torch.add(layer3, conv2)

        conv3 = F.relu(self.bn3_down(self.conv3_down(layer3)))

        layer4 = F.relu(self.bn4a(self.conv4a(conv3)))
        layer4 = F.relu(self.bn4b(self.conv4b(layer4)))
        layer4 = F.relu(self.bn4c(self.conv4c(layer4)))
        layer4 = torch.add(layer4, conv3)

        conv4 = F.relu(self.bn4_down(self.conv4_down(layer4)))

        layer5 = F.relu(self.bn5a(self.conv5a(conv4)))
        layer5 = F.relu(self.bn5b(self.conv5b(layer5)))
        layer5 = F.relu(self.bn5c(self.conv5c(layer5)))
        layer5 = torch.add(layer5, conv4)

        conv5 = F.relu(self.bn5_up(self.conv5_up(layer5)))

        cat6 = torch.cat((conv5, layer4), 1)

        if self.dropout: cat6 = self.do6(cat6)

        layer6 = F.relu(self.bn6a(self.conv6a(cat6)))
        layer6 = F.relu(self.bn6b(self.conv6b(layer6)))
        layer6 = F.relu(self.bn6c(self.conv6c(layer6)))
        layer6 = torch.add(layer6, cat6)

        conv6 = F.relu(self.bn6_up(self.conv6_up(layer6)))

        cat7 = torch.cat((conv6, layer3), 1)

        if self.dropout: cat7 = self.do7(cat7)

        layer7 = F.relu(self.bn7a(self.conv7a(cat7)))
        layer7 = F.relu(self.bn7b(self.conv7b(layer7)))
        layer7 = F.relu(self.bn7c(self.conv7c(layer7)))
        layer7 = torch.add(layer7, cat7)

        conv7 = F.relu(self.bn7_up(self.conv7_up(layer7)))

        cat8 = torch.cat((conv7, layer2), 1)

        if self.dropout: cat8 = self.do8(cat8)

        layer8 = F.relu(self.bn8a(self.conv8a(cat8)))
        layer8 = F.relu(self.bn8b(self.conv8b(layer8)))
        layer8 = torch.add(layer8, cat8)

        conv8 = F.relu(self.bn8_up(self.conv8_up(layer8)))

        cat9 = torch.cat((conv8, layer1), 1)

        if self.dropout: cat9 = self.do9(cat9)

        layer9 = F.relu(self.bn9(self.conv9(cat9)))
        layer9 = torch.add(layer9, cat9)
        layer9 = self.final(self.bn9_1x1(self.conv9_1x1(layer9)))

        return layer9


# a smaller version of UNet
# used for testing purposes

class UNetSmall(nn.Module):

    def __init__(self, dice=False):

        super(UNetSmall, self).__init__()

        self.conv1_input =      nn.Conv2d(1, 64/2, 3, padding=1)
        self.conv1 =            nn.Conv2d(64/2, 64/2, 3, padding=1)
        self.conv2_input =      nn.Conv2d(64/2, 128/2, 3, padding=1)
        self.conv2 =            nn.Conv2d(128/2, 128/2, 3, padding=1)
        self.conv3_input =      nn.Conv2d(128/2, 256/2, 3, padding=1)
        self.conv3 =            nn.Conv2d(256/2, 256/2, 3, padding=1)
        self.conv4_input =      nn.Conv2d(256/2, 512/2, 3, padding=1)
        self.conv4 =            nn.Conv2d(512/2, 512/2, 3, padding=1)

        self.conv7_up =         nn.ConvTranspose2d(512/2, 256/2, 2, 2)
        self.conv7_input =      nn.Conv2d(512/2, 256/2, 3, padding=1)
        self.conv7 =            nn.Conv2d(256/2, 256/2, 3, padding=1)
        self.conv8_up =         nn.ConvTranspose2d(256/2, 128/2, 2, 2)
        self.conv8_input =      nn.Conv2d(256/2, 128/2, 3, padding=1)
        self.conv8 =            nn.Conv2d(128/2, 128/2, 3, padding=1)
        self.conv9_up =         nn.ConvTranspose2d(128/2, 64/2, 2, 2)
        self.conv9_input =      nn.Conv2d(128/2, 64/2, 3, padding=1)
        self.conv9 =            nn.Conv2d(64/2, 64/2, 3, padding=1)
        self.conv9_output =     nn.Conv2d(64/2, 2, 1)

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def switch(self, dice):

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def forward(self, x):

        layer1 = F.relu(self.conv1_input(x))
        layer1 = F.relu(self.conv1(layer1))

        layer2 = F.max_pool2d(layer1, 2)
        layer2 = F.relu(self.conv2_input(layer2))
        layer2 = F.relu(self.conv2(layer2))

        layer3 = F.max_pool2d(layer2, 2)
        layer3 = F.relu(self.conv3_input(layer3))
        layer3 = F.relu(self.conv3(layer3))

        layer4 = F.max_pool2d(layer3, 2)
        layer4 = F.relu(self.conv4_input(layer4))
        layer4 = F.relu(self.conv4(layer4))

        layer7 = F.relu(self.conv7_up(layer4))
        layer7 = torch.cat((layer3, layer7), 1)
        layer7 = F.relu(self.conv7_input(layer7))
        layer7 = F.relu(self.conv7(layer7))

        layer8 = F.relu(self.conv8_up(layer7))
        layer8 = torch.cat((layer2, layer8), 1)
        layer8 = F.relu(self.conv8_input(layer8))
        layer8 = F.relu(self.conv8(layer8))

        layer9 = F.relu(self.conv9_up(layer8))
        layer9 = torch.cat((layer1, layer9), 1)
        layer9 = F.relu(self.conv9_input(layer9))
        layer9 = F.relu(self.conv9(layer9))
        layer9 = self.final(self.conv9_output(layer9))

        return layer9