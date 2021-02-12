import torch
import torch.nn as nn
from collections import OrderedDict
class FCN (nn.Module):
    def __init__(self, input_vertex=16, output_vertex=2, hidden_vertex=32, num_layer=2):
        super(FCN,self).__init__()
        self.iv = input_vertex
        self.ov = output_vertex
        self.hv = hidden_vertex
        self.nl = num_layer
        input_layer = [
            nn.Linear(self.iv,self.hv),
            nn.ReLU(),
            nn.BatchNorm1d(self.hv),
        ]
        output_layer = [
            nn.Linear(self.hv,self.ov),
        ]
        self.input_layer = nn.Sequential(*input_layer)
        self.output_layer = nn.Sequential(*output_layer)
        self.hidden_layer = nn.Sequential()
        self.hidden_layer = self.add_hidden_layer()

    def add_hidden_layer(self):
        for i in range(self.nl):
            self.hidden_layer.add_module('hidden_layer{}'.format(i+1),nn.Linear(self.hv,self.hv))
            self.hidden_layer.add_module('ReLU{}'.format(i+1),nn.ReLU())
            self.hidden_layer.add_module('BatchNor{}'.format(i + 1), nn.BatchNorm1d(self.hv))
        return self.hidden_layer


    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

if __name__=='__main__':
    tensor = torch.rand([2,18])
    label = torch.Tensor([1,1]).long()
    net = FCN(18, 2, 18, 1)
    opt = torch.optim.SGD(net.parameters(),lr=0.1)
    for i in range(100):
        pred = net(tensor)
        loss = nn.CrossEntropyLoss()(pred, label)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(loss)
        print(pred)
