#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import torch
import torch.nn as nn


class Flatten(nn.Module):
    """Convenience class that flattens feature maps to vectors."""
    def forward(self, x):
        # Required for pretrained weights for some reason?
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)
