#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 20:45:40 2022

@author: joesh
"""
import torch
import torch.nn as nn
import SDPAtt

class MultiHeadAttentionCla(nn.Module):
    
    def __init__(self, input_dim, embed_dim, n_heads):
        super().__init__()    #inherit methods nn modules
        assert embed_dim%n_heads==0
        
        self.n_heads=n_heads
        self.embed_dim=embed_dim
        self.input_dim=input_dim
        self.kqv_proj=nn.Linear(input_dim, n_heads*embed_dim)
        self.o_proj=nn.Linear(n_heads*input_dim, embed_dim)
        
        self._reset_parameter()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.kqv_proj.weight)
        self.kqv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        
    def feed_forward(self, x, mask=None, return_attention=False):
        batch_size, seq_len=x.size()
        qkv=self.kqv_proj(x)
        qkv=qkv.permute(0,2,1,3)
        q,k,v=qkv.chunk(3,dim=-1) #splits tensor into specified chunks
        
        
        values, attention = SDPAtt.ScaledDotProductAttention(k, v, q)
        values=values.permute(0,2,1,3)
        values=values.reshape(batch_size, seq_len, self.embed_dim)
        
        o=self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o