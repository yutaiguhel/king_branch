# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 06:16:19 2018

@author: yuta
"""

"""
============このクラスはQ_module_new02をimport後にimportして下さい。===================
"""
import numpy as np
from Q_module_new02 import*
class Q_H:
    """
    クロスワイヤーに対応したクラスです。
    NVに関するパラメータの構造体、ハミルトニアンの生成、時間発展演算子を計算するメソッドを持っています。
    """
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
    a1=500 #ワイヤ1のΩ-V変換式の係数1
    b1=0.6 #ワイヤ1Ω-V変換式の係数2
    a2=156 #ワイヤ2のΩ-V変換式の係数1
    b2=0.8 #ワイヤ2Ω-V変換式の係数2
    w_theta=90*pi/180 #ワイヤ1とワイヤ2の角度[rad]
    
    def __init__(self):
        self.C_mat=II
        self.H0=III #NVのハミルトニアン
        self.Vd=III #ドライブハミルトニアン
        self.Hf=III #回転座標系に乗るためのハミルトニアン
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
                C_z=tensor(Sz,III) #電子、窒素
                for j in range(i):
                    C_z=tensor(C_z,II)
                C_z=tensor(C_z,sigz)
                for k in range(len(Q_H.Ac_list)-i-1):
                    C_z=tensor(C_z,II)
                Hint_ec.append(Q_H.Ac_list[i]*C_z)
                C_list.append(2)
            He=x[5]*tensor(Sz*Sz,III,self.C_mat) #x[5]:D0
            Hn=x[7]*tensor(III,Sz*Sz,self.C_mat) #x[7]:An
            Hint_en=x[6]*tensor(Sz,Sz,self.C_mat) #x[6]:Qn
            HB=Q_H.ge*(x[8])*Q_H.muB/10.0*tensor(Sz,III,self.C_mat) #x[8]:Bzs
            self.H0=He+Hn+Hint_en+HB
            for i in range(len(Q_H.Ac_list)):
                self.H0=self.H0+Hint_ec[i]
        else:
            He=x[5]*tensor(Sz*Sz,III)
            Hn=x[7]*tensor(III,Sz*Sz)
            Hint_en=x[6]*tensor(Sz,Sz)
            HB=ge*(x[8])*muB/10.0*tensor(Sz,III)
            self.H0=He+Hn+Hint_en+HB
    
    def R_V_func(self,a,b,V):
        if b > V:
            Ome=a*(b*V - (V**2)/2.0)
        else:
            Ome=1/2.0 * a *b**2
        return Ome
    
    def Vdrive_all(self,V1,V2,phi):
        if self.wire==0:
            Ur=(-1j*phi*Sp-1j*phi*Sm).expm()
            Ome1=self.R_V_func(self.a1,self.b1,V1)
            Ome2=self.R_V_func(self.a2,self.b2,V2)
            self.Vd = (1/2.0)*(Ome1*Sx+Ome2*Ur*(Sx*cos(Q_H.w_theta)+Sy*sin(Q_H.w_theta))*Ur.dag())
        elif self.wire==1:
            Ome1=self.R_V_func(self.a1,self.b1,V1)
            self.Vd=Ome1/2.0*Sx
        elif self.wire==2:
            Ome2=self.R_V_func(self.a2,self.b2,V2)
            self.Vd=Ome2/2.0*Sx
        self.Vd=tensor(self.Vd,III,self.C_mat)
        
    def H_rot(self,omega): #回転座標系に乗せるための行列を生成する関数
        mw=III
        rf=III
        if self.pulse==1:
            rf=Sz*Sz
        else:
            mw=Sz*Sz
        self.Hf=omega*tensor(mw,rf,self.C_mat)

    def Tevo(self,width): #量子状態の時間発展を計算する関数
        rho0=tensor(S0,III/3,self.C_mat/(2**len(self.Ac_list)))
        HI=self.Vd+self.H0-self.Hf
        U=(-2*pi*1j*HI*width).expm()
        dens=U*rho0*U.dag()
        return dens
    
    def exp(self,rhof): #量子状態の期待値を計算する関数
        #a:density matrix, b:projector of electron, c:Nuclear, d:13C1, f:13C2, g:13C3 
        e=(rhof*tensor(S0,III/3,self.C_mat/(2**len(self.Ac_list)))).tr() #expected value
        e=e.real
        return e
    
        