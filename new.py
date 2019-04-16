import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt

class videoNetwork(nn.Module):

    def __init__(self):
        super(videoNetwork,self).__init__()

        self.conv1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(224),
            nn.Conv2d(in_channels=224, out_channels=1536,kernel_size=5,stride=1,padding=2)
            )

        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(1536),
            nn.Conv2d(in_channels=1536, out_channels=1536,kernel_size=5,stride=1,padding=2)
            )

        self.conv3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(1536),
            nn.Conv2d(in_channels=1536, out_channels=1536,kernel_size=5,stride=1,padding=2)
            )

        self.conv4 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(1536),
            nn.Conv2d(in_channels=1536, out_channels=1536,kernel_size=5,stride=1,padding=2)
            )

        self.conv5 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(1536),
            nn.Conv2d(in_channels=1536, out_channels=1536,kernel_size=5,stride=1,padding=2)
            )

        self.conv6 = nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm2d(1536),
                    nn.Conv2d(in_channels=1536, out_channels=1536,kernel_size=5,stride=1,padding=2)
                    )

        self.conv7 = nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm2d(1536),
                    nn.Conv2d(in_channels=1536, out_channels=1536,kernel_size=5,stride=1,padding=2)
                    )

        self.conv8 = nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm2d(1536),
                    nn.Conv2d(in_channels=1536, out_channels=1536,kernel_size=5,stride=1,padding=2)
                    )

        self.conv9 = nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm2d(1536),
                    nn.Conv2d(in_channels=1536, out_channels=1536,kernel_size=5,stride=1,padding=2)
                    )

        self.conv10 = nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm2d(1536),
                    nn.Conv2d(in_channels=1536, out_channels=1536,kernel_size=5,stride=1,padding=2)
                    )


    def forward(self, out):
        out = self.conv1(out)
        print(11111)
        out = self.conv2(out)
        print(22222)
        out = self.conv3(out)
        print(33333)
        out = self.conv4(out)
        print(44444)
        out = self.conv5(out)
        print(55555)
        out = self.conv6(out)
        print(66666)
        out = self.conv7(out)
        print(77777)
        out = self.conv8(out)
        print(88888)
        out = self.conv9(out)
        print(99999)
        out = self.conv10(out)
        print(1010101010)

        print("the shape of out is: {}".format(out.shape))
        out = out.view(out.size(0),224,-1)
        print("reshaped out = {}".format(out.shape))
        return out




class audioNetwork(nn.Module):

    def __init__(self):
        super(audioNetwork,self).__init__()

        self.conv1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=1024,kernel_size=5,stride=1,padding=2)
            )

        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(in_channels=1024, out_channels=1024,kernel_size=5,stride=2)
            )

        self.conv3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(in_channels=1024, out_channels=1024,kernel_size=5,stride=1,padding=2)
            )

        self.conv4 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(in_channels=1024, out_channels=1024,kernel_size=5,stride=2)
            )

        self.conv5 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(in_channels=1024, out_channels=2,kernel_size=5,stride=1,padding=2)
            )

    def forward(self, out):
        out = self.conv1(out)
        print(1111)
        out = self.conv2(out)
        print(2222)
        out = self.conv3(out)
        print(3333)
        out = self.conv4(out)
        print(4444)
        out = self.conv5(out)
        print(5555)

        print("the shape of out is: {}".format(out.shape))
        out = out.view(out.size(0),-1)
        print("reshaped out = {}".format(out.shape))
        return out














if __name__ == "__main__":
    test_video = torch.randn(1,224,224,1)
    print(test_video)
    video_net = videoNetwork()

    test_audio = torch.randn(1,2,321,297)
    print(test_audio)
    audio_net = audioNetwork()

    out = video_net(test_video)
    out_audio = audio_net(test_audio)

    print(out)
    print(out.shape)

    print(out_audio)
    print(out_audio.shape)