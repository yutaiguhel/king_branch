# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 06:16:19 2018

@author: yuta
"""

"""
============このクラスはQ_module_new02をimport後,importして下さい。===================
"""
import numpy as np
from Q_module_new02 import*
class Q_H:
    pi=np.pi #円周率
    D0=2870 #ゼロ磁場分裂[MHz]
    QN=-4.945  #核四重極子分裂[MHz]
    AN=-2.2 #電子スピン-窒素核スピン間超微細相互作用[MHz]
    h=6.62606957*1e-34/(2*pi) #ディラック定数[m**2*kg/s]
    ge=2.002319 #電子スピンのg因子
    mu=1.5e-7 #真空の透磁率[m**3*s**4*A**2/kg]
    muB=927.401*1e-20/h #ボーア磁子[J/T]
    muN=5.05078324*1e-27/h #核磁子[J/T]
    Be=4.5*1e-5 #地磁気[T]
    Bo=-0.450*1e-4 #外部磁場[T]
    Ac_list=[-3.265] #電子スピン-炭素同位体核スピン1間超微細相互作用[MHz]
    #真のハミルトニアンの定義
    x0=[D0,QN,AN,Bo+Be]
    for i in range(len(Ac_list)):
        x0.append(Ac_list[i])
    
    def __init__(self):
        self.C_mat=II
        self.H0=III
        self.Vd=III
        self.Hf=III
        self.pulse=0 #MW:0, RF:1
        self.rho0=tensor(S0,III/3)
    
    def C_matrix(self):
        C_list=[]
        for i in range(len(Q_H.Ac_list)):
            C_list.append(2)
        self.C_mat=Qobj(qeye(2**len(Q_H.Ac_list)),dims=[C_list,C_list])/2**len(Q_H.Ac_list)
        return self.C_mat
        
    def H(self,x):
        C_list=[]
        Hint_ec=[]
        self.H0=III
        if len(Q_H.Ac_list) != 0:
            for i in range(len(Q_H.Ac_list)): #i番目の炭素の超微細相互作用
                C_z=tensor(Sz,III)
                for j in range(i):
                    C_z=tensor(C_z,II)
                C_z=tensor(C_z,sigz)
                for k in range(len(Q_H.Ac_list)-i-1):
                    C_z=tensor(C_z,II)
                Hint_ec.append(Q_H.Ac_list[i]*C_z)
                C_list.append(2)
            He=x[0]*tensor(Sz*Sz,III,self.C_mat)
            Hn=x[1]*tensor(III,Sz*Sz,self.C_mat)
            Hint_en=x[2]*tensor(Sz,Sz,self.C_mat)
            HB=Q_H.ge*(x[3])*Q_H.muB/10.0*tensor(Sz,III,self.C_mat)
            self.H0=He+Hn+Hint_en+HB
            for i in range(len(Q_H.Ac_list)):
                self.H0=self.H0+Hint_ec[i]
        else:
            He=x[0]*tensor(Sz*Sz,III)
            Hn=x[1]*tensor(III,Sz*Sz)
            Hint_en=x[2]*tensor(Sz,Sz)
            HB=ge*(x[3])*muB/10.0*tensor(Sz,III)
            self.H0=He+Hn+Hint_en+HB
            return H0

    def Vdrive(self,omega): #ドライブハミルトニアンを生成する関数
        mw=III
        rf=III
        if self.pulse==1:
            rf=Sx
        else:
            mw=Sx
        self.Vd=omega/2.0*tensor(mw,rf,self.C_mat)
        return self.Vd
    
    def H_rot(self,omega): #回転座標系に乗せるための行列を生成する関数
        mw=III
        rf=III
        if self.pulse==1:
            rf=Sz*Sz
        else:
            mw=Sz*Sz
        self.Hf=omega*tensor(mw,rf,self.C_mat)
        return self.Hf

    
    def Tevo(self,width): #量子状態の時間発展を計算する関数
        rho0=tensor(S0,III/3,self.C_mat)
        HI=self.Vd+self.H0-self.Hf
        U=(-2*pi*1j*HI*width).expm()
        dens=U*rho0*U.dag()
        return dens
    
    def exp(self,rhof): #量子状態の期待値を計算する関数
        #a:density matrix, b:projector of electron, c:Nuclear, d:13C1, f:13C2, g:13C3 
        e=(rhof*tensor(S0,III,self.C_mat)).tr() #expected value
        e=e.real
        return e
    
        