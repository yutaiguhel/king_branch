# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 07:56:42 2018

@author: yuta
"""
from Bayes_function13 import*

class Rabi(Bayes_Function):
    def __init__(self):
        Bayes_Function.__init__(self)
        self.ParamC["tw"]=0
                   
    def init_C(self):
        """
        実験設計の初期値代入
        """
        self.C=[self.V1, self.V2, self.phi, self.t, self.MWf]
        
    
    