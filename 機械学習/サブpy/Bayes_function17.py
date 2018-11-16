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
        self.w=0 #現在のパーティクルの重みs
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
        self.C_best=0
        self.C_best_i=0
        
        #GRAPE
        self.omega_j=[] #GRAPEの結果を格納する配列
        self.U_grape=[] #GRAPEの真値に対する時間発展演算子
        
    def params(self):
        """
        インスタンス変数に依存する変数を初期化
        """
        self.params_list=["a1","b1","a2","b2","w_theta","D0","AN","QN","Bz"]
        self.x0_dict={"a1":self.a1,"b1":self.b1,"a2":self.a2,"b2":self.b2,"w_theta":self.w_theta
                 ,"D0":self.D0,"AN":self.AN,"QN":self.QN,"Bz":self.Be+self.Bo} #真のハミルトニアン
        self.x0=[self.x0_dict["a1"],self.x0_dict["b1"],self.x0_dict["a2"],self.x0_dict["b2"],self.x0_dict["w_theta"]
                ,self.x0_dict["D0"],self.x0_dict["AN"],self.x0_dict["QN"],self.x0_dict["Bz"]]
        self.x_dict={"a1":self.a1+self.a1/5,"b1":self.b1+self.b1/5,"a2":self.a2-self.a2/10,"b2":self.b2+self.b2/5,"w_theta":self.w_theta
                ,"D0":self.D0,"AN":self.AN,"QN":self.QN,"Bz":self.Be+self.Bo} #現在のパーティクル
        self.x=[self.x_dict["a1"],self.x_dict["b1"],self.x_dict["a2"],self.x_dict["b2"],self.x_dict["w_theta"]
                ,self.x_dict["D0"],self.x_dict["AN"],self.x_dict["QN"],self.x_dict["Bz"]]
        self.x_first=self.x_dict
        
        
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
        
        i=0
        n=len(w)
        m=len(x[0])
        mu=np.zeros([1,m])
        for i in range(n):
            mu=mu+w[i][0]*x[i]
        
        #mu=w*x
        
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
        self.w=np.ones([n_p,1])
        
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
        for i,p in enumerate(self.ParamH):
            if self.ParamH[p]==1:
                px=1
                #px=1/np.var(self.x.T[i]) #各パラメータの初期分散
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
        self.rho_init() #量子状態の初期化
        self.H_0(x)
        if self.exp_flag=="rabi":
            self.H_rot(C[4]) #C[4]:MW周波数
            self.Vdrive_all(x,C[0],C[1],C[2]) #C[0]:V1, C[1]:V2, C[2]:ワイヤ間の位相差phi
            self.Tevo(C[3]) #C[3]:MWwidth
        elif self.exp_flag=="ramsey":
            self.H_rot(C[4]) #C[4]:MW周波数
            self.Vdrive_all(x,C[0],C[1],C[2]) #C[0]:V1, C[1]:V2, C[2]:ワイヤ間の位相差phi
            self.Tevo(C[3]) #C[3]:MWwidth
            self.Tevo_free(C[5]) #C[5]:wait
            self.Tevo(C[3]) #C[3]:MWwidth
        expect0=self.exp(self.rho) #ms=0で測定
        if expect0 > 1.0:
            print("Probability Error")
            print(expect0)
            expect0=1
        return expect0
                    
    def Prob_Lookup(self):
        ptable_rabi=np.zeros([self.n_particles(),self.n_exp("rabi")])
        ptable_ramsey=np.zeros([self.n_particles(),self.n_exp("ramsey")])
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
                
        #テーブル作成
        for i in range(repeat):
            if i==1 and repeat==2:
                self.exp_flag="ramsey"
            for k in range(self.n_exp(self.exp_flag)):
                for j in range(self.n_particles()):
                    self.ptable[i][j][k]=self.Expsim(self.x[j],self.C[i][k])

    def UtilIG_bayes_risk_one(self):
        self.exp_flag=self.exp_select
        for k in range(self.n_exp(self.exp_flag)):
            num=binomial(self.d,np.transpose(self.ptable[0],(1,0))[k])
            num=self.Mean(self.w,num.reshape(num.shape[0],1))
            L=binom.pmf(num,n=self.d,p=np.transpose(self.ptable[0],(1,0))[k]).reshape(np.transpose(self.ptable[0],(1,0))[k].shape[0],1)
            w_new=L*self.w
            x_infer=self.Mean(self.w,self.x)
            x_infer_new=self.Mean(w_new,self.x)
            self.U[k]=self.U[k]*np.trace(self.Q*np.dot((x_infer_new[0] - x_infer[0]).T,(x_infer_new[0] - x_infer[0])))
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
        for i in range(2):
            if i==1:
                self.exp_flag="ramsey"
            for k in range(self.n_exp(self.exp_flag)):
                num=binomial(self.d,np.transpose(self.ptable[i],(1,0))[k])
                num=self.Mean(self.w,num.reshape(num.shape[0],1))
                L=binom.pmf(num,n=self.d,p=np.transpose(self.ptable[i],(1,0))[k]).reshape(np.transpose(self.ptable[i],(1,0))[k].shape[0],1)
                w_new=L*self.w
                x_infer=self.Mean(self.w,self.x)
                x_infer_new=self.Mean(w_new,self.x)
                self.U[i][k]=self.U[i][k]*np.trace(self.Q*np.dot((x_infer_new[0] - x_infer[0]).T,(x_infer_new[0] - x_infer[0])))
        for i in range(2):
            self.U[i]=self.U[i]/(np.sum(self.U[0])+np.sum(self.U[1]))
        U_min=[np.min(self.U[0]),np.min(self.U[1])]
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
        #ラビ振動とラムゼー干渉の実験を行う
        self.mode=1
        self.p_exp=self.Expsim(self.x0,self.C_best)#真値におけるms0の確立
        num=binomial(self.d, self.p_exp)#実験をd回行いｍs=0であった回数
        temp=binom.pmf(num,n=self.d,p=self.ptable_best)#各パーティクルでの実験でms=0にいた確率
        self.w=self.w*temp.reshape([len(temp),1]) #重みの更新
        self.w=self.w/np.sum(self.w) #重みの規格化
    
    def Bayes_risk(self): #ベイズリスクを計算する関数
        """
        ベイズリスクを計算する関数
        """
        x_infer=self.Mean(self.w,self.x) 
        self.risk.append(np.trace(self.Q*np.dot((self.x - x_infer[0]).T,(self.x - x_infer[0]))))
    
    #=============================結果を描画する関数=============================
    def Estimate_credible_region(self,level):
        id_sorted=np.argsort(self.w,axis=0)[::-1]
        w_sorted=np.sort(self.w,axis=0)[::-1]
        cumsum_weights=np.cumsum(w_sorted)
        id_cred=cumsum_weights<=level
        if((id_cred==False).all()):
            x_range=self.x[id_sorted[0]]
        else:
            x_range_temp=self.x[id_sorted][id_cred]
            x_range=np.reshape(x_range_temp,[len(x_range_temp),len(self.x[0])])
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
        now=datetime.datetime.now()
        if self.i==0:
            os.mkdir(cd+"\\"+'HL_{0:%Y%m%d}'.format(now)+"_ExpSelect_"+self.exp_select)
        os.chdir(cd+"\\"+'HL_{0:%Y%m%d}'.format(now)+"_ExpSelect_"+self.exp_select)
        
        #共有テキストファイル作成
        l=list(self.C_best)
        if self.exp_flag=="rabi":
            l.append(0)
        else:
            l.append(1)
        np.savetxt("Property"+str(self.i)+".txt",l,newline="\r\n")

        #ディレクトリを元に戻す
        os.chdir("../")
        
class Bayes_parallel(Bayes_Function):
    def __init__(self):
        Bayes_Function.__init__(self)
        
    def wrapper_Expsim(self,tuple_data):
        return tuple_data[0](tuple_data[1],tuple_data[2])
    
    def Prob_Lookup_parallel(self):
        p = multiprocessing.Pool()
        if self.exp_select=="all":
            #ラビ振動についてテーブル作成
            self.exp_flag="rabi"
            data_rabi = [(self.Expsim,self.x[i],self.C[0][j]) for i in range(self.x.shape[0]) for j in range(self.C[0].shape[0])]
            ptable_rabi=p.map(self.wrapper_Expsim, data_rabi)
            ptable_rabi=np.array(ptable_rabi).reshape(1,len(ptable_rabi)).reshape(self.n_particles(),self.n_exp("rabi")) #テーブルに変換
            
            #ラムゼー干渉についてテーブル作成
            self.exp_flag="ramsey"
            data_ramsey = [(self.Expsim,self.x[i],self.C[1][j]) for i in range(self.x.shape[0]) for j in range(self.C[1].shape[0])]
            ptable_ramsey=p.map(self.wrapper_Expsim, data_ramsey)
            ptable_ramsey=np.array(ptable_ramsey).reshape(1,len(ptable_ramsey)).reshape(self.n_particles(),self.n_exp("ramsey")) #テーブルに変換
            
            #各テーブルをまとめる
            self.ptable=[ptable_rabi,ptable_ramsey]
        
        else:
            data=[(self.Expsim,self.x[i],self.C[0][j]) for i in range(self.x.shape[0]) for j in range(self.C[0].shape[0])]
            self.exp_flag=self.exp_select
            self.ptable=p.map(self.wrapper_Expsim,data)
            self.ptable=[np.array(self.ptable).reshape(1,len(self.ptable)).reshape(self.n_particles(),self.n_exp(self.exp_select))] #テーブルに変換
        p.close()
        p.terminate()