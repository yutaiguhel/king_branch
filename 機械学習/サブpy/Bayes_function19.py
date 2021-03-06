# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:08:39 2018

@author: yuta
"""
import time
import os
import datetime
from Q_H07 import*
import itertools
from scipy.stats import binom
from numpy.random import*
import matplotlib.pyplot as plt
import multiprocessing
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
"""
このクラスはベイズ推定のためのクラスです.
クロスワイヤのドライブハミルトニアンを生成できます.
効用はベイズリスク、エントロピーが選べます。
確率のルックアップテーブルは十字に計算するか、全て計算するか選ぶことが出来ます。
"""

def wrapper_as_class(arg, **kwarg):
        # メソッドfをクラスメソッドとして呼び出す関数
        return Bayes_Function.wrapper_Expsim(*arg, **kwarg)
    
class Bayes_Function(Q_H):
    """
    ベイズ推定のメソッドをまとめたクラス
    """
    def __init__(self):
        """
        変数にデフォルト値を代入
        """
        Q_H.__init__(self)
        self.ex=100 #試行回数
        self.i=0 #現在の試行回数
        self.n={"a1":5,"b1":5,"a2":5,"b2":5,"w_theta":5,"D0":5,"AN":5,"QN":5,"Bz":5} #推定基底毎のパーティクルの数 a1,b1,a2,b2,w_theta,D0,An,Qn,Bz
        self.g={"V1":5,"V2":5,"phi":30,"MWwidth":5,"MWfreq":5} #量子操作において変更するパラメータの分割数 V1,V2,phi,MWwidth,MWfreq
        self.d=1000 #一度の推定に使用する実験データの数
        self.a=0.75 #パーティクルの再配分における移動強度
        self.resample_threshold=0.5 #パーティクルの再配分を行う判断をする閾値
        self.approx_ratio=0.98 #不要なパーティクルを削除する際の残す割合
        self.bayes_threshold=1 #推定を終えるベイズリスクの閾値
        self.mode=1 #Expsimの測定モード(1:確率,0:射影測定)
        self.flag1=False #パーティクルの数が変化したらTrue
        self.flag2=False #実験設計の数が変化したらTrue
        self.exp_flag="rabi"
        self.exp_select="all" #"rabi", "ramsey", "all"
        self.Q=0 #ベイズリスクの重み行列
        self.U=0 #効用
        self.B_R=0 #ベイズリスク
        self.Utility="binomial"
        self.ptable_mode="all" #all or cross
        self.p_exp=0 #真値におけるms=0にいた確率
        self.dir_num=0 #作成したテキストファイルの番号
        self.Data_num=0 #読み取ったテキストファイルの番号
        self.Data=0 #実験データ 一個目:操作後の発光量、2つ目:ms=0の発光量
        
        #結果格納配列
        self.i_list=[]
        self.ptable=[]
        self.ptable_best=0
        self.risk=[]
        self.exp_list=[]
        
        #信用区間
        self.xout_in_region_max={"a1":[],"b1":[],"a2":[],"b2":[],"w_theta":[],"D0":[],"AN":[],"QN":[],"Bz":[]}
        self.xout_in_region_min={"a1":[],"b1":[],"a2":[],"b2":[],"w_theta":[],"D0":[],"AN":[],"QN":[],"Bz":[]}
        
        #パーティクル
        self.w=0 #現在のパーティクルの重み
        self.ParamH={"a1":0,"b1":0,"a2":0,"b2":0,"w_theta":0,"D0":1,"AN":0,"QN":0,"Bz":0} #変更するパーティクルのパラメータ
        self.RangeH={"a1":5,"b1":3,"a2":10,"b2":5,"w_theta":2*np.pi,"D0":10,"AN":0,"QN":0,"Bz":0} #変更する範囲
        
        #実験設計
        self.V1=1 #ワイヤ1の電圧[V]
        self.V2=1 #ワイヤ2の電圧[V]
        self.phi=180*pi/180 #ワイヤ間の位相差[rad]
        self.t=0.05 #MWパルス長[us]
        self.tw=1.0 #ラムゼー干渉の待機時間[us]
        self.MWf=2870 #MW周波数の中心[MHz]
        self.ParamC={"V1":0,"V2":0,"phi":1,"MWwidth":1,"MWfreq":1,"tw":1} #V1,V2,phi,MWwidth,MWfreq #変更する量子操作のパラメータ
        self.RangeC={"V1":1,"V2":1,"phi":360,"MWwidth":0.05,"MWfreq":10,"tw":0.5} #変更する範囲
        self.Limit={"V1max":2.25,"V1min":0.001,"V2max":2.25,"V2min":0.001,"phimax":2*np.pi,"phimin":0,\
                    "MWwidthmax":5,"MWwidthmin":0.01,\
                    "MWfreqmax":2875,"MWfreqmin":2865,"twmax":5,"twmin":0} #実験パラメータの上限と下限s
        self.C_best=0 #効用が最大の実験
        self.C_best_i=0 #効用が最大の実験のインデックス
        
        #GRAPE
        self.omega_j=[] #GRAPEの結果を格納する配列
        self.U_grape=[] #GRAPEの真値に対する時間発展演算子
        
    def params(self):
        """
        インスタンス変数に依存する変数を初期化
        """
        #ラビ振動パラメータリスト
        self.params_rabi_list=["V1", "V2", "phi", "MWwidth", "MWfreq"]
        
        #パラメータリスト
        self.params_list=["a1","b1","a2","b2","w_theta","D0","AN","QN","Bz"]
        
        #真のハミルトニアン(辞書型)
        self.x0_dict={"a1":self.a1,"b1":self.b1,"a2":self.a2,"b2":self.b2,"w_theta":self.w_theta-np.pi/10
                 ,"D0":self.D0,"AN":self.AN,"QN":self.QN,"Bz":self.Be+self.Bo-1} #真のハミルトニアン
        
        #真のハミルトニアン
        self.x0=[self.x0_dict["a1"],self.x0_dict["b1"],self.x0_dict["a2"],self.x0_dict["b2"],self.x0_dict["w_theta"]
                ,self.x0_dict["D0"],self.x0_dict["AN"],self.x0_dict["QN"],self.x0_dict["Bz"]]
        
        #パーティクルの中心値(辞書型)
        self.x_dict={"a1":self.a1,"b1":self.b1,"a2":self.a2,"b2":self.b2,"w_theta":self.w_theta
                ,"D0":self.D0,"AN":self.AN,"QN":self.QN,"Bz":self.Be+self.Bo} #現在のパーティクル
        
        #パーティクルの中心値
        self.x=[self.x_dict["a1"],self.x_dict["b1"],self.x_dict["a2"],self.x_dict["b2"],self.x_dict["w_theta"]
                ,self.x_dict["D0"],self.x_dict["AN"],self.x_dict["QN"],self.x_dict["Bz"]]
        self.x_first=self.x_dict
        
    def Exp_limit(self,j):
        #実験パラメータを範囲内に収める
        for i,p in enumerate(self.params_rabi_list):
            #実験パラメータが範囲外の実験設計のインデックスを取得
            id_max=self.C[j].T[i] > self.Limit[p+"max"]
            
            #範囲外の実験パラメータを範囲内に収める
            self.C[j].T[i][id_max]=self.Limit[p+"max"]
            
            #実験パラメータが範囲外の実験設計のインデックスを取得
            id_min=self.C[j].T[i] < self.Limit[p+"min"]
            
            #範囲外の実験パラメータを範囲内に収める
            self.C[j].T[i][id_min]=self.Limit[p+"min"]

    def n_particles(self):
        """
        パーティクルの数を返す
        """
        n_p=1
        if self.i==0:
            for p in self.n:
                if self.ParamH[p]==1:
                    n_p=n_p * self.n[p] 
        else:
            n_p=len(self.w)
        return n_p
    
    def n_exp(self,exp):
        """
        実験設計の数を返す
        """
        n_C=1
        
        #全実験設計の数を返す
        if self.i==0:
            for c in self.g:
                if self.ParamC[c]==1:
                    n_C=n_C*self.g[c]
            
            if exp=="ramsey":
                return n_C
            elif exp=="rabi":
                if self.ParamC["tw"]==1:
                    return int(n_C/self.g["tw"])
                else:
                    return n_C
                
        #選択した実験設計の数を返す
        else:
            if self.exp_select=="all":
                if exp=="ramsey":
                    n_C=len(self.U[1])
                elif exp=="rabi":
                    n_C=len(self.U[0])
                else:
                    n_C=len(self.U[0])+len(self.U[1])
            else:
                n_C=len(self.U)
            return n_C
        
    def Mean(self,w,x): #重み付き平均を計算する関数
        """
        w:重み
        x:パラメータ
        wで重みづけされたxの平均を返す
        """
        mu=w*x
        
        #パーティクル数方向に和を取る
        mu=np.sum(mu,axis=0)
        
        return mu
        
    def init_C(self):
        """
        実験設計の初期値代入
        """
        self.C=[self.V1, self.V2, self.phi, self.t, self.MWf, self.tw]
    
    def init_w(self):
        """
        重みを一様分布に初期化
        """
        n_p=self.n_particles()
        self.w=np.ones([n_p,1])/n_p
        
    def init_U(self):
        """
        効用を一様分布に初期化
        """
        n_C_rabi=self.n_exp("rabi")
        n_C_ramsey=self.n_exp("ramsey")
        
        #全実験設計を選択
        if self.exp_select=="all":
            self.U=[np.ones([n_C_rabi,1])/(n_C_rabi+n_C_ramsey),np.ones([n_C_ramsey,1])/(n_C_rabi+n_C_ramsey)]
        
        #ラビ振動のみ行う
        elif self.exp_select=="rabi":
            self.U=np.ones([n_C_rabi,1])/(n_C_rabi)
            
        #ラムゼー干渉のみ行う
        elif self.exp_select=="ramsey":
            self.U=np.ones([n_C_ramsey,1])/(n_C_ramsey)
    
    def init_x(self):
        """
        パーティクルの中心を生成
        ここでは炭素の超微細相互作用を追加
        """
        for Ac in self.Ac_list:
            self.x.append(Ac)
       
    
    def init_x0(self):
        """
        パーティクルの真値を生成
        ここでは炭素の超微細相互作用を追加
        """
        N=len(self.Ac_list)
        for i in range(N):
            self.params_list.append("Ac"+str(i))
            self.x0_dict["Ac"+str(i)]=self.Ac_list[i]
            self.x0.append(self.Ac_list[i])
    
    def weighting_matrix(self):
        """
        ベイズリスクの重み行列を作成
        ParamHの要素が1ならば対応する重み行列の要素も1
        つまり、ベイズリスクを考慮する
        """
        self.Q=np.zeros([len(self.x0), len(self.x0)])
        
        #広げたパラメータのみ考慮する
        for i,p in enumerate(self.ParamH):
            if self.ParamH[p]==1:
                px=1
            else:
                px=0
            self.Q[i][i]=px
    
    def resample(self,w,x): #パーティクルの移動と重みの再配分を行う関数
        """
        a:resample強度
        各パーティクルをx=a*x+(1-a)*x_averageに移動させる
        つまり、各パーティクルを強度aで分布の中心に寄せる
        """
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
    
    def reapprox(self,w,x,mode): #不要となったパーティクルを削除する関数
        """
        m:残すパーティクルの数
        ws:昇順に並び替えた重み
        wsの(m+1)番目の要素よりも大きい重みのパーティクルは残す
        """
        n=len(w)
        ws=sorted(w)
        m=floor(n*(1.0-self.approx_ratio))
        if m<1:
            m=0
        j=0
        delist=[]
        
        #delistにm個たまるまで継続
        while j!=m:
            i=0
            for i in range(n):
                if w[i]==ws[j] and n!=0:
                    delist.append(i)
                    j=j+1
                if j==m:
                    break
        #重み、パーティクルから不要なパーティクルを削除
        w=np.delete(w,delist,0)
        x=np.delete(x,delist,0)
        
        w=w/sum(w)
        
        if mode=="par":
            if n != len(w):
                self.flag1=True
                print("reapprox_par")
            else:
                self.flag1=False
        elif mode=="exp":
            if n != len(w):
                self.flag2=True
                print("reapprox_exp")
            else:
                self.flag2=False
        return w,x
            
    
    def Particlemaker(self,x,n,Param,Range): #パーティクルを生成する関数
        #itertools.product与えられた変数軍([x1,x2,x3],[y1,y2])の総重複なし組み合わせを配列として出力
        """
        パーティクルを生成する関数
        最小値:x-Range/2
        最大値:x+Range/2
        """
        N=len(x)
        temp=[]
        
        #ParamHが1のパーティクルのみ全幅RangeHで広げる
        for i,p in enumerate(Param):
            if(Param[p]==1):
                temp.append(np.linspace(x[i]-Range[p]/2,x[i]+Range[p]/2,n[p]))
            else:
                temp.append([x[i]])
        return(np.array(list(itertools.product(*temp))))
    
    def Expmaker(self):
        """
        実験設計の組み合わせを作成する関数
        """
        temp=[]
        temp_rabi=[]
        
        #ParamCが1のパラメータのみ全幅RangeCで広げる
        for i,p in enumerate(self.ParamC):
            if(self.ParamC[p]==1):
                temp.append(np.linspace(self.C[i]-self.RangeC[p]/2,self.C[i]+self.RangeC[p]/2,self.g[p]))
            else:
                temp.append([self.C[i]])
            if p != "tw":
                if(self.ParamC[p]==1):
                    temp_rabi.append(np.linspace(self.C[i]-self.RangeC[p]/2,self.C[i]+self.RangeC[p]/2,self.g[p]))
                else:
                    temp_rabi.append([self.C[i]])
    
        if self.exp_select=="all":
            return([np.array(list(itertools.product(*temp_rabi))),np.array(list(itertools.product(*temp)))])
        elif self.exp_select=="rabi":
            return [np.array(list(itertools.product(*temp_rabi)))]
        elif self.exp_select=="ramsey":
            return [np.array(list(itertools.product(*temp)))]
    
    def Expsim(self,x,C): #実験と同様のシーケンスを行いデータの生成を行う関数
        """
        パーティクルxに実験Cで実験シミュレーションを行う関数
        """
        #量子状態の初期化
        self.rho_init()
        
        #System Hamiltonian
        self.H_0(x)
        
        #ラビ振動の時間発展
        if self.exp_flag=="rabi":
            #回転座標系に乗るためのハミルトニアン
            self.H_rot(C[4]) #C[4]:MW周波数
            
            #回転座標系に乗った時のドライブハミルトニアン
            self.Vdrive_all(x,C[0],C[1],C[2]) #C[0]:V1, C[1]:V2, C[2]:ワイヤ間の位相差phi
            
            #ドライブハミルトニアンで時間発展
            self.Tevo(C[3]) #C[3]:MWwidth
            
        #ラムゼー干渉の時間発展
        elif self.exp_flag=="ramsey":
            #回転座標系に乗るためのハミルトニアン
            self.H_rot(C[4]) #C[4]:MW周波数
            
            #回転座標系に乗った時のドライブハミルトニアン
            self.Vdrive_all(x,C[0],C[1],C[2]) #C[0]:V1, C[1]:V2, C[2]:ワイヤ間の位相差phi
            
            #ドライブハミルトニアンで時間発展
            self.Tevo(C[3]) #C[3]:MWwidth
            
            #自由時間発展
            self.Tevo_free(C[5]) #C[5]:wait
            
            #ドライブハミルトニアンで時間発展
            self.Tevo(C[3]) #C[3]:MWwidth
            
        #ms=0で測定
        expect0=self.exp(self.rho) #ms=0で測定
        if expect0 > 1.0:
            print("Probability Error")
            print(expect0)
            expect0=1
        return expect0
                    
    def Prob_Lookup(self):
        #ラビ振動のテーブルを作成
        ptable_rabi=np.zeros([self.n_particles(),self.n_exp("rabi")])
        
        #ラムゼー干渉のテーブルを作成
        ptable_ramsey=np.zeros([self.n_particles(),self.n_exp("ramsey")])
        
        #テーブル全体を作成
        if self.exp_select=="all":
            repeat=2
            self.exp_flag="rabi"
            self.ptable=[ptable_rabi,ptable_ramsey]
        else:
            repeat=1
            self.exp_flag=self.exp_select
            if self.exp_select=="rabi":
                self.ptable=[ptable_rabi]
            elif self.exp_select=="ramsey":
                self.ptable=[ptable_ramsey]
                
        #テーブルを埋める
        for i in range(repeat):
            if i==1 and repeat==2:
                self.exp_flag="ramsey"
            for k in range(self.n_exp(self.exp_flag)):
                for j in range(self.n_particles()):
                    self.ptable[i][j][k]=self.Expsim(self.x[j],self.C[i][k])
                    
    def Exp_random(self):
        """
        実験をランダムに選ぶ関数
        """
        
        if self.exp_select=="all":
            i=np.round(np.random.random(1)[0])
            if i==0:
                self.exp_flag="rabi"
                self.exp_list.append(0)
            else:
                self.exp_flag="ramsey"
                self.exp_list.append(1)
            self.C_best_i=int(np.round(self.n_exp(exp)*(np.random.random(1)[0])))
            self.C_best=self.C[i][self.C_best_i]
            
        else:
            self.exp_flag=self.exp_select
            if self.exp_select=="rabi":
                self.exp_list.append(0)
            elif self.exp_select=="ramsey":
                self.exp_list.append(1)
            self.C_best_i=int(np.round(self.n_exp(self.exp_select)*(np.random.random(1)[0])))
            self.C_best=self.C[0][self.C_best_i]
            
        #全パーティクルについてシミュレーション
        self.ptable_best=np.zeros(self.n_particles(),)
        for j in range(self.n_particles()):
            self.ptable_best[j]=self.Expsim(self.x[j],self.C_best)

    def UtilIG_bayes_risk_one(self):
        self.exp_flag=self.exp_select
        for k in range(self.n_exp(self.exp_flag)):
            #最もらしいms=0にいた回数
            num=self.d*np.transpose(self.ptable[0],(1,0))[k].reshape(self.x.shape[0],1)
            num=np.round(self.Mean(self.w,num))
            
            #実験設計Ckで実験したms=0の確率
            pjk=np.transpose(self.ptable[0],(1,0))[k].reshape(self.x.shape[0],1)
            
            #尤度の算出、重みの更新
            L=binom.pmf(num,n=self.d,p=pjk)
            w_new=L*self.w
            
            #ベイズリスクの算出
            x_infer=self.Mean(self.w,self.x)
            x_infer_new=self.Mean(w_new,self.x)
            self.U[k]=self.U[k]*np.trace(self.Q*np.dot((x_infer_new - x_infer).T,(x_infer_new - x_infer)))
        
        self.U=self.U/np.sum(self.U)
        if self.exp_select=="rabi":
            self.exp_list.append(0)
        else:
            self.exp_list.append(1)
        self.C_best_i=np.argmin(self.U)
        self.C_best=self.C[0][self.C_best_i]
        self.ptable_best=np.transpose(self.ptable[0],(1,0))[self.C_best_i]
        
    def UtilIG_bayes_risk_all(self):
        self.exp_flag="rabi"
        
        #各実験操作について平均効用を計算
        for i in range(2):
            if i==1:
                self.exp_flag="ramsey"
                
            #各実験設計について効用を計算
            for k in range(self.n_exp(self.exp_flag)):

                """
                #ある結果を仮定して効用を計算
                Ud=np.zeros(self.d+1,)
                for data in range(self.d+1):
                    L=binom.pmf(data,n=self.d,p=np.transpose(self.ptable[i],(1,0))[k]).reshape(np.transpose(self.ptable[i],(1,0))[k].shape[0],1)
                    w_new=L*self.w
                    x_infer=self.Mean(self.w,self.x)
                    x_infer_new=self.Mean(w_new,self.x)
                    Ud[data]=np.trace(self.Q*np.dot((x_infer_new[0] - x_infer[0]).T,(x_infer_new[0] - x_infer[0])))
                self.U[i][k]=self.U[i][k]*np.sum(Ud)
                """
                
                #事前分布からあり得る結果を求める
                #結果を乱数にする場合
                #num=binomial(self.d,np.transpose(self.ptable[i],(1,0))[k])
                
                #結果を乱数にしない場合
                num=self.d*np.transpose(self.ptable[i],(1,0))[k].reshape(self.x.shape[0],1)
                num=np.round(self.Mean(self.w,num))
                
                #ルックアップテーブルのある列を抜き出す
                pjk=np.transpose(self.ptable[i],(1,0))[k].reshape(self.x.shape[0],1)
                
                #尤度を求める
                L=binom.pmf(num,n=self.d,p=pjk)
                
                #重みを仮更新する
                w_new=L*self.w
                
                #現在の推定値、仮の推定値を求める
                x_infer=self.Mean(self.w,self.x)
                x_infer_new=self.Mean(w_new,self.x)
                
                #k番目の実験設計のベイズリスクを計算する
                self.U[i][k]=self.U[i][k]*np.trace(self.Q*np.dot((x_infer_new - x_infer).T,(x_infer_new - x_infer)))
                
        #効用を規格化
        for i in range(2):
            self.U[i]=self.U[i]/(np.sum(self.U[0])+np.sum(self.U[1]))
        U_min=[np.min(self.U[0]),np.min(self.U[1])]
        
        #ベイズリスクが最小となる実験設計を選ぶ
        if U_min[0] < U_min[1]:
            self.exp_flag="rabi"
            self.exp_list.append(0)
            self.C_best_i=np.argmin(self.U[0])
            self.C_best=self.C[0][self.C_best_i]
            self.ptable_best=np.transpose(self.ptable[0],(1,0))[self.C_best_i]
        else:
            self.exp_flag="ramsey"
            self.exp_list.append(1)
            self.C_best_i=np.argmin(self.U[1])
            self.C_best=self.C[1][self.C_best_i]
            self.ptable_best=np.transpose(self.ptable[1],(1,0))[self.C_best_i]
            
    def Update(self): #ベイズ推定を行う関数 引数(self,Ci)
        """
        パーティクルの重みを更新する関数
        """
        #実験を行う
        self.mode=1
        self.p_exp=self.Expsim(self.x0,self.C_best)#真値におけるms0の確率
        
        #結果を乱数にする場合
        #num=binomial(self.d, self.p_exp) #実験をd回行いｍs=0であった回数
        
        #結果を乱数にしない場合
        num=np.round(self.d*self.p_exp) #実験をd回行いｍs=0であった回数
        
        #尤度を計算
        temp=binom.pmf(num,n=self.d,p=self.ptable_best)#各パーティクルでの実験でms=0にいた確率
        temp=temp.reshape([len(temp),1])
        
        #重みの更新
        self.w=temp*self.w 
        
        #重みの全要素が0だった場合の例外処理
        if np.sum(self.w)==0:
            self.w=np.ones([self.n_particles(),1])/self.n_particles()
        else:
            self.w=self.w/np.sum(self.w) #重みの規格化
    
    def Bayes_risk(self): #ベイズリスクを計算する関数
        """
        ベイズリスクを計算する関数
        """
        #現在の推定値を計算
        x_infer=self.Mean(self.w,self.x) 
        self.risk.append(np.trace(self.Q*np.dot((self.x - x_infer).T,(self.x - x_infer))))
    
    #=============================結果を描画する関数=============================
    def generate_cmap(self,colors):
        """自分で定義したカラーマップを返す"""
        values = range(len(colors))
        vmax = np.ceil(np.max(values))
        color_list = []
        for v, c in zip(values, colors):
            color_list.append( ( v/ vmax, c) )
        return LinearSegmentedColormap.from_list('custom_cmap', color_list)
    
    def Estimate_credible_region(self,level):
        #重みを降順に並べなおす
        id_sorted=np.argsort(self.w,axis=0)[::-1]
        w_sorted=np.sort(self.w,axis=0)[::-1]
        
        #累積確率を計算する
        cumsum_weights=np.cumsum(w_sorted)
        id_cred=cumsum_weights<=level
        
        #パーティクルが一つになった場合の例外処理
        if((id_cred==False).all()):
            x_range=self.x[id_sorted[0]]
        else:
            x_range_temp=self.x[id_sorted][id_cred]
            x_range=np.reshape(x_range_temp,[len(x_range_temp),len(self.x[0])])
        
        #信用区間内のパーティクルを戻り値
        return x_range
    
    def Region_edge(self,level,param):
        """
        各推定における信用区間に含まれるパーティクルの最大もしくは最小値のarrayを得る関数
        paramで推定パラメーターを指定
        """
        x_region=self.Estimate_credible_region(level)
        for i,p in enumerate(self.ParamH):
            if self.ParamH[p]==1 and p==param:
                temp=[]
                for j in range(x_region.shape[0]):
                    temp.append(x_region[j][i])
                self.xout_in_region_max[p].append(max(temp))
                self.xout_in_region_min[p].append(min(temp))
                
    def Region_edge_output(self,param,flag):
        """
        flagで戻り値が最大か最小かを指定(1:max,0:min)
        """
        for i,p in enumerate(self.ParamH):
            if self.ParamH[p]==1 and p==param:
                if flag==1:
                    #print("xout_in_region_max",self.xout_in_region_max) #debug用
                    return self.xout_in_region_max[p]
                else:
                    #print("xout_in_region_min",self.xout_in_region_min) #debug用
                    return self.xout_in_region_min[p]
        
    def Show_result(self):
        #グラフのx軸を作成
        wi=np.linspace(1,self.n_particles(),self.n_particles())
        Ui_rabi=np.linspace(1,self.n_exp("rabi"),self.n_exp("rabi"))
        Ui_ramsey=np.linspace(1,self.n_exp("ramsey"),self.n_exp("ramsey"))
        plt.figure(figsize=(12,8))
        
        #重みの表示
        plt.subplot(3,2,1)
        plt.xlabel("Particle number",fontsize=20)
        plt.ylabel("probability",fontsize=20)
        plt.title("Weight",fontsize=20)
        plt.plot(wi,self.w)
        
        #ラビ振動の効用を表示
        if self.exp_select != "ramsey":
            plt.subplot(3,2,2)
            plt.xlabel("Rabi number",fontsize=20)
            plt.ylabel("Utility [a.u.]",fontsize=20)
            plt.title("Utility_rabi",fontsize=20)
            if self.exp_select=="all":
                plt.plot(Ui_rabi, self.U[0])
            else:
                plt.plot(Ui_rabi, self.U)
        
        #ラムゼー干渉の効用を表示
        if self.exp_select != "rabi": 
            plt.subplot(3,2,3)
            plt.xlabel("Ramsey number",fontsize=20)
            plt.ylabel("Utility [a.u.]",fontsize=20)
            plt.title("Utility_ramsey",fontsize=20)
            if self.exp_select=="all":
                plt.plot(Ui_ramsey, self.U[1])
            else:
                plt.plot(Ui_ramsey, self.U)
        
        #ベイズリスクを表示
        plt.subplot(3,2,4)
        plt.xlabel("iteration number",fontsize=20)
        plt.ylabel("Bayes_risk",fontsize=20)
        plt.title("Bayes_risk after estimation",fontsize=20)
        plt.plot(self.i_list, self.risk)
        
        #選ばれた実験を表示
        plt.subplot(3,2,5)
        plt.xlabel("iteration number",fontsize=20)
        plt.ylabel("Experiment",fontsize=20)
        plt.title("Experiment (0:rabi, 1:ramsey)",fontsize=20)
        plt.plot(self.i_list, self.exp_list)
        
        plt.tight_layout()
        plt.show()
        
    def show_hyper_parameter(self):
        """
        ハイパーパラメータを出力する関数
        """
        print("============================ハイパーパラメータの表示======================\n")
        print("実験回数:%d" %(self.i))
        print("リサンプリング強度 %f" %(self.a))
        print("リサンプリング閾値 %f" %(self.resample_threshold))
        print("ベイズリスクの閾値 %f" %(self.bayes_threshold))
        print("それぞれのハミルトニアンパラメータの分割数")
        print(self.n)
        print("それぞれの実験パラメータの分割数")
        print(self.g)
        print("パーティクルの真の値")
        print(self.x0)
        print("始めのパーティクルの中心")
        print(self.x_first)
        print("推定したパラメータD0, AN, QN, Bz, Ac_list")
        print(self.ParamH)
        print(self.RangeH)
        print("変化させた実験設計とその範囲")
        print(self.ParamC)
        print(self.RangeC)
        print("現在のパーティクルの数:%d" %(self.n_particles()))
    
    def storage_data(self):
        #ディレクトリの移動,作成
        cd=os.getcwd()
        cd="\\\\UNICORN\\data&prog\\機械学習用テストフォルダ"
        os.chdir(cd)
        while self.i==0 and os.path.isdir(cd+"\\"+'HL_'+str(self.dir_num))==True:
            self.dir_num=self.dir_num+1
        print(self.dir_num)
        if self.i==0:
            os.mkdir(cd+"\\"+'HL_'+str(self.dir_num))
        os.chdir(cd+"\\"+'HL_'+str(self.dir_num))
        
        #共有テキストファイル作成
        l=list(self.C_best)
        if self.exp_flag=="rabi":
            l.append(0)
        else:
            l.append(1)
        np.savetxt("Setting"+str(self.i)+".csv",l,newline="\r\n",delimiter=",")

        #ディレクトリを元に戻す
        os.chdir("../")
        
    def Read_data(self):
        #ディレクトリの移動,作成
        cd=os.getcwd()
        cd="\\\\UNICORN\\data&prog\\機械学習用テストフォルダ"+"\\"+"HL_"+str(self.dir_num)
        os.chdir(cd)
        A=False
        while A==False:
            A=os.path.exists(cd+"\\"+'Data'+str(self.Data_num)+".csv")
            time.sleep(1)
        
        D=np.loadtxt("Data"+str(self.Data_num)+".csv",delimiter=",")
        self.Data_num=self.Data_num+1
        self.p_exp=D[0]/D[1]
        print(self.p_exp)
        
        if self.p_exp > 1:
            self.p_exp=0.99999999999999999

        #ディレクトリを元に戻す
        os.chdir("../")
        
    def END_file(self):
        cd=os.getcwd()
        cd="\\\\UNICORN\\data&prog\\機械学習用テストフォルダ"
        os.chdir(cd)
        dir_num_end=0
        while self.i==0 and os.path.isdir(cd+"\\"+'HL_'+str(dir_num_end))==True:
            dir_num_end=dir_num_end+1
        print(dir_num_end)
        os.chdir(cd+"\\"+'HL_'+str(self.dir_num))
        
        #共有テキストファイル作成
        l=[1,2,3]
        np.savetxt("END.txt",l)

        #ディレクトリを元に戻す
        os.chdir("../")
        
        
class Bayes_parallel(Bayes_Function):
    def __init__(self):
        Bayes_Function.__init__(self)
        
    def wrapper_Expsim(self,tuple_data):
        return tuple_data[0](tuple_data[1],tuple_data[2])
    
    def Prob_Lookup_parallel(self):
        #プロセス作成
        p = multiprocessing.Pool(4)
        
        #ルックアップテーブルを作成
        if self.exp_select=="all":
            #ラビ振動についてテーブル作成
            self.exp_flag="rabi"
            data_rabi = [(self.Expsim,self.x[i],self.C[0][j]) for i in range(self.x.shape[0]) for j in range(self.C[0].shape[0])]
            ptable_rabi=p.map(self.wrapper_Expsim, data_rabi)
            ptable_rabi=np.array(ptable_rabi).reshape(1,len(ptable_rabi)).reshape(self.x.shape[0],self.n_exp("rabi")) #テーブルに変換
            
            #ラムゼー干渉についてテーブル作成
            self.exp_flag="ramsey"
            data_ramsey = [(self.Expsim,self.x[i],self.C[1][j]) for i in range(self.x.shape[0]) for j in range(self.C[1].shape[0])]
            ptable_ramsey=p.map(self.wrapper_Expsim, data_ramsey)
            ptable_ramsey=np.array(ptable_ramsey).reshape(1,len(ptable_ramsey)).reshape(self.x.shape[0],self.n_exp("ramsey")) #テーブルに変換
            
            #各テーブルをまとめる
            self.ptable=[ptable_rabi,ptable_ramsey]
        
        else:
            data=[(self.Expsim,self.x[i],self.C[0][j]) for i in range(self.x.shape[0]) for j in range(self.C[0].shape[0])]
            self.exp_flag=self.exp_select
            self.ptable=p.map(self.wrapper_Expsim,data)
            self.ptable=[np.array(self.ptable).reshape(1,len(self.ptable)).reshape(self.x.shape[0],self.n_exp(self.exp_select))] #テーブルに変換
        
        #プロセスの停止
        p.close()
        
        #プロセスの終了
        p.terminate()