import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
from torch.nn import init
import math
from typing import *
from opendelta.basemodel import DeltaBase

class Victim(nn.Module):
    def __init__(self):
        super(Victim, self).__init__()

    def forward(self, inputs):
        pass
    
    def process(self, batch):
        pass
