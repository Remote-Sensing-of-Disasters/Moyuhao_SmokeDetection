import torch
import torch.nn as nn
from collections import OrderedDict
class FCN (nn.Module):
    def __init__(self, input_vertex=16, output_vertex=2, hidden_vertex=32, num_layer=2,BN=False,Drop=0):
        super(FCN,self).__init__()
        self.iv = input_vertex
        self.ov = output_vertex
        self.hv = hidden_vertex
        self.nl = num_layer
        input_layer = [
                nn.Linear(self.iv, self.hv),
                nn.ReLU(),
            ]


        output_layer = [
            #nn.Dropout(0.5),
            nn.Linear(self.hv,self.ov),
            #nn.Dropout(0.5),
             ####################
        ]
        self.input_layer = nn.Sequential(*input_layer)
        self.output_layer = nn.Sequential(*output_layer)
        self.hidden_layer = nn.Sequential()
        self.hidden_layer = self.add_hidden_layer(BN,Drop)

        if BN:
            self.input_layer.add_module('BatchNorInput',nn.BatchNorm1d(self.hv))
        if Drop:
            self.input_layer.add_module('DropoutInput',nn.Dropout(Drop))
    def add_hidden_layer(self,BN,Drop):
        for i in range(self.nl):
            #self.hidden_layer.add_module('Dropout{}'.format(i + 1), nn.Dropout(0.5))  #######20210505
            self.hidden_layer.add_module('hidden_layer{}'.format(i+1),nn.Linear(self.hv,self.hv))
            #self.hidden_layer.add_module('BatchNor{}'.format(i + 1), nn.BatchNorm1d(self.hv))
            #self.hidden_layer.add_module('Dropout{}'.format(i + 1), nn.Dropout(0.5))
            self.hidden_layer.add_module('ReLU{}'.format(i+1),nn.ReLU())
            if BN:
                self.hidden_layer.add_module('BatchNorHid{}'.format(i + 1), nn.BatchNorm1d(self.hv))
            if Drop:
                self.hidden_layer.add_module('DropoutHid{}'.format(i + 1), nn.Dropout(Drop))
        return self.hidden_layer


    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x
