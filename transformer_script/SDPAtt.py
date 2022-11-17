#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import math

def ScaledDotProductAttention(k,v,q, mask=None):
    d_k=q.size()[-1]
    attn_logits=torch.matmul(q, k.transpose(-2,-1))
    attn_logits=attn_logits/math.sqrt(d_k)
    
    if mask is not None:
        attn_logits=attn_logits.masked_fill(mask==0, -9e15)
    attention=F.softmax(attn_logits, dim=-1)
    values=torch.matmul(attention, v)
    
    return values, attention

