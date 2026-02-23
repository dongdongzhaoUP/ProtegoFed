import math
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from opendelta.basemodel import DeltaBase
from torch.nn import init
from torch.nn.parameter import Parameter


class Victim(nn.Module):
    def __init__(self):
        super(Victim, self).__init__()

    def forward(self, inputs):
        pass

    def process(self, batch):
        pass
