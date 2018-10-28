# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:08:39 2018

@author: yuta
"""
import time
from Q_H01 import*
import itertools
class Bayes_Function(Q_H):
    def __init__(self):
        Q_H.__init__(self)
        self.ex=50 #試行回数
        self.n=10 #推定基底毎のパーティクルの数
        self.g=10 #量子操作において変更するパラメータの分割数
        self.d=100 #一度の推定に使用する実験データの数
        self.a=0.75 #パーティクルの再配分における移動強度
        self.resample_threshold=0.5 #パーティクルの再配分を行う判断をする閾値
        self.approx_ratio=0.98 #不要なパーティクルを削除する際の残す割合
        self.mode=1
        self.state=0
        self.Q=0
        self.U=0
        self.Utility="bayes_risk"
        #真のハミルトニアンs
        self.x0=[self.D0, self.AN, self.QN, self.Be+self.Bo]
        #結果格納配列
        self.D=np.empty([self.d,1])
        #パーティクル
        self.w=0 #現在のパーティクルの重みs
        self.x=[self.D0+3, self.AN, self.QN, self.Be+self.Bo] #現在のパーティクル
        self.ParamH=[1,0,0,0,0] #変更するパーティクルのパラメータ
        self.RangeH=[10] #変更する範囲
        #実験設計
        self.OMEGA=10 #Rabi周波数の中心[MHz]
        self.t=0.05 #MWパルス長[us]
        self.MWf=2870 #MW周波数の中心[MHz]
        self.ParamC=[1,0,0] #変更する量子操作のパラメータ
        self.RangeC=[1,0,4.0] #変更する範囲
        self.C_best=[]
        #GRAPE
        self.omega_j=[]
        self.U_grape=[]
        
    def n_particles(self):
        """
        パーティクルの数
        """
        return self.n ** sum(self.ParamH)
    
    def n_exp(self):
        """
        実験設計の数
        """
        return self.g ** sum(self.ParamC)
        
    def Mean(self,w,x): #重み付き平均を計算する関数
        i=0
        n=len(w)
        m=len(x[0])
        mu=np.zeros([1,m])
        for i in range(n):
            mu=mu+w[i][0]*x[i]
        return mu
        
    def init_C(self):
        self.C=[self.OMEGA, self.t, self.MWf]
        return self.C
    
    def init_w(self):
        n_p=self.n_particles()
        self.w=np.ones([n_p,1])
        return self.w
        
    def init_U(self):
        n_C=self.n_exp()
        self.U=np.ones([n_C,1])
        return self.U
    
    def init_x(self,x):
        for Ac in self.Ac_list:
            x.append(Ac)
        return x
    
    def resample(self,w,x): #パーティクルの移動と重みの再配分を行う関数
        i=0
        n=len(w)
        m=len(x[0])
        mu=self.Mean(w,x)
        mui=np.zeros([1,m])
        for i in range(n):
            if w[i]<1.0/n*self.a:
                mui=self.a*x[i]+(1-self.a)*mu
                x[i]=mui
            w[i][0]=1.0/n
        print ("resample")
        return w,x
    
    def reapprox(self,w,x): #不要となったパーティクルを削除する関数
        n=len(w)
        ws=sorted(w)
        m=floor(n*(1.0-self.approx_ratio))
        if m<1:
            m=0
        j=0
        delist=[]
        while j!=m:
            i=0
            for i in range(n):
                if w[i]==ws[j] and n!=0:
                    delist.append(i)
                    j=j+1
                if j==m:
                    break
        w=np.delete(w,delist,0)
        w=w/sum(w)
        x=np.delete(x,delist,0)
        return w,x,delist
    
    def Particlemaker(self,x,n,Param,Range): #パーティクルを生成する関数
        #itertools.product与えられた変数軍([x1,x2,x3],[y1,y2])の総重複なし組み合わせを配列として出力
        N=len(x)
        temp=[]
        for i in range(N):
            if(Param[i]==1):
                temp.append(np.linspace(x[i]-Range[i]/2,x[i]+Range[i]/2,n))
            else:
                temp.append([x[i]])
        return(np.array(list(itertools.product(*temp))))
    
    def Expsim(self,x,C): #実験と同様のシーケンスを行いデータの生成を行う関数
        H0=self.H(x)
        HfMW=self.H_rot(C[2]) #C[2]:MW周波数
        VdMW=self.Vdrive(C[0]) #C[0]:ラビ周波数
        rhof=self.Tevo(C[1]) #MWwidth
        expect0=self.exp(rhof) #ms=0で測定
        if self.mode==0:
            if rand()>=expect0:
                mes=1 #ms=+1,-1
            else:
                mes=0 #ms=0
        else:
            if self.state==0:
                mes=expect0
            else:
                mes=1.0-expect0
                if mes<0:
                    mes=0
        return mes
        
    def Update(self,C): #ベイズ推定を行う関数
        i=0
        n=len(self.w)
        nw=np.zeros([n,1])
        d1=len(self.D)
        d2=sum(self.D)
        d3=d1-d2
        print("実験回数:%d, ms=±1にいた回数:%f, ms=0にいた回数:%f" %(d1, d2, d3))
        self.mode=1
        for i in range(n):
            p=self.Expsim(self.x[i],C) #ms=0の確率
            nw[i][0]=self.w[i][0]*pow(p,d3)*pow(1-p,d2) #更新した重み
        nw=nw/sum(nw)
        return nw
    
    def weighting_matrix(self):
        self.Q=np.zeros([len(self.x0), len(self.x0)])
        for i in range(len(self.ParamH)):
            p=self.ParamH[i]
            self.Q[i][i]=p
        return self.Q
    
    def UtilIG(self,Ci): #るCの量子操作の効用を計算する関数
        j=0
        d=2
        p=np.zeros(d)
        m=argmax(self.w)
        self.mode=1
        for j in range(d):
            p[j]=self.Expsim(rho0,x[m],Ci)
        In=self.Hd(p)
        return In
    
    def UtilIG_bayes_risk(self,Ci): #あるCについてベイズリスクを計算する関数
        i=0
        Util=0
        m=np.argmax(self.w)
        self.mode=0
        for j in range(self.d):
            self.D[j]=self.Expsim(self.x[m],Ci) #シミュレーションによるデータ取得
        w_new=self.Update(Ci)
        x_infer=self.Mean(self.w,self.x)
        Util=np.trace(self.Q*np.dot((self.x - x_infer[0]).T,(self.x - x_infer[0])))
        return Util
            
    def GuEx(self): #最適な量子操作を特定する関数
        n=len(self.C)
        Ud=np.zeros([n,1])
        i=0
        for i in range(n):
            if self.Utility=="entropy":
                Ud[i]=self.UtilIG(self.C[i])*self.U[i]
            if self.Utility == "bayes_risk":
                Ud[i]=self.UtilIG_bayes_risk(self.C[i])*self.U[i]
        Ud=Ud/sum(Ud)
        return Ud
    
    #=================================GRAPE====================================
    def Tevo_operator(self,U_init,width):
        HI=Vd+H0-HfMW
        U=(-2*pi*1j*HI*width).expm()
        Ud=U*U_init
        return Ud
    
    def GRAPEpulse(self):
        m=np.argmax(self.w)
        C_known=[]
        for i in range(len(self.Ac_list)):
            if self.Ac_list[i] != 0:
                C_known.append(self.Ac_list[i])
        grape=GRAPE(self.x[m][0],self.x[m][1],0,C_known,C_inhomo,Be,theta,phi1,phi2,state_init,
                state_goal,weighting_value,permax,pulse_time,t_div,target_list)
    
        phi_array, self.omega_array=grape.optimize()
        return self.omega_array
    
    def GRAPE_operator(self,x,omega_j):
        H0=self.H(x)
        HfMW=self.H_rot(C[2])
        C_list=[]
        for i in range(len(self.Ac_list)):
            C_list.append(2)
        U0=tensor(III,III,self.C_mat)
        U=U0
        for i in range(len(omega_j)):
            VdMW=self.Vdrive(omega_j[i])
            U=self.Tevo_operator(H0,VdMW,HfMW,U,t_div)
        return U