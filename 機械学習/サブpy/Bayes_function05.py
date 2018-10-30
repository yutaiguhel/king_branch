# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:08:39 2018

@author: yuta
"""
import time
from Q_H01 import*
import itertools
from scipy.stats import binom
from numpy.random import*
import matplotlib.pyplot as plt
"""
このクラスはベイズ推定のためのクラスです.
効用はベイズリスク、エントロピーが選べます。
確率のルックアップテーブルは十字に計算するか、全て計算するか選ぶことが出来ます。
"""
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
        self.n=100 #推定基底毎のパーティクルの数
        self.g=100 #量子操作において変更するパラメータの分割数
        self.d=1000 #一度の推定に使用する実験データの数
        self.a=0.75 #パーティクルの再配分における移動強度
        self.resample_threshold=0.5 #パーティクルの再配分を行う判断をする閾値
        self.approx_ratio=0.98 #不要なパーティクルを削除する際の残す割合
        self.bayes_threshold=1 #推定を終えるベイズリスクの閾値
        self.mode=1 #Expsimの測定モード(1:確率,0:射影測定)
        self.state=0 #0:ms=0で射影測定,1:ms=±1で射影測定
        self.flag1=False #パーティクルの数が変化したらTrue
        self.flag2=False #実験設計の数が変化したらTrue
        self.Q=0
        self.U=0
        self.B_R=0 #ベイズリスク
        self.Utility="binomial"
        self.ptable_mode="all" #all or cross
        #結果格納配列
        self.i_list=[]
        self.ptable_C=[]
        self.ptable_x=[]
        self.ptable=[]
        self.risk=[]
        #パーティクル
        self.w=0 #現在のパーティクルの重みs
        self.ParamH=[1,0,0,0,0] #変更するパーティクルのパラメータ
        self.RangeH=[10] #変更する範囲
        #実験設計
        self.OMEGA=10 #Rabi周波数の中心[MHz]
        self.t=0.05 #MWパルス長[us]
        self.MWf=2870 #MW周波数の中心[MHz]
        self.ParamC=[1,0,0] #変更する量子操作のパラメータ
        self.RangeC=[5,0,0] #変更する範囲
        self.C_best=0
        self.C_best_i=0
        #GRAPE
        self.omega_j=[]
        self.U_grape=[]
        
    def params(self):
        self.x0=[self.D0, self.AN, self.QN, self.Be+self.Bo] #真のハミルトニアン
        self.D=np.empty([self.d,1])
        self.x=[self.D0-3, self.AN-0.05, self.QN, self.Be+self.Bo] #現在のパーティクル
        self.x_first=self.x
        
        
    def n_particles(self):
        """
        パーティクルの数を返す
        """
        if self.i==0:
            n_p=self.n ** sum(self.ParamH)
        else:
            n_p=len(self.w)
        return n_p
    
    def n_exp(self):
        """
        実験設計の数を返す
        """
        if self.i==0:
            n_C=self.g ** sum(self.ParamC)
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
        return mu
        
    def init_C(self):
        """
        実験設計の初期値代入
        """
        self.C=[self.OMEGA, self.t, self.MWf]
    
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
        n_C=self.n_exp()
        self.U=np.ones([n_C,1])
    
    def init_R(self):
        """
        ベイズリスクを一様分布に初期化
        """
        n_C=self.n_exp()
        self.B_R=np.ones([n_C,1])
    
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
        for Ac in self.Ac_list:
            self.x0.append(Ac)
    
    def weighting_matrix(self):
        """
        ベイズリスクの重み行列を作成
        ParamHの要素が1ならば対応する重み行列の要素も1
        つまり、ベイズリスクを考慮する
        """
        self.Q=np.zeros([len(self.x0), len(self.x0)])
        for i in range(len(self.ParamH)):
            p=self.ParamH[i]
            self.Q[i][i]=p
    
    def resample(self,w,x): #パーティクルの移動と重みの再配分を行う関数
        """
        a:resample強度
        各パーティクルをx=a*x+(1-a)*x_averageに移動させる
        つまり、各パーティクルを分布の中心に寄せる
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
    
    def reapprox_par(self): #不要となったパーティクルを削除する関数
        """
        m:残すパーティクルの数
        ws:昇順に並び替えた重み
        wsの(m+1)番目の要素よりも大きい重みのパーティクルは残す
        """
        n=len(self.w)
        ws=sorted(self.w)
        m=floor(n*(1.0-self.approx_ratio))
        if m<1:
            m=0
        j=0
        delist=[]
        while j!=m:
            i=0
            for i in range(n):
                if self.w[i]==ws[j] and n!=0:
                    delist.append(i)
                    j=j+1
                if j==m:
                    break
        self.w=np.delete(self.w,delist,0)
        self.w=self.w/sum(self.w)
        self.x=np.delete(self.x,delist,0)
        if n != len(self.w):
            self.flag1=True
            print("reapprox_exp")
        else:
            self.flag1=False
    
    def reapprox_exp(self): #不要となったパーティクルを削除する関数
        """
        m:残すパーティクルの数
        ws:昇順に並び替えた重み
        wsの(m+1)番目の要素よりも大きい重みのパーティクルは残す
        """
        n=len(self.U)
        Us=sorted(self.U)
        m=floor(n*(1.0-self.approx_ratio))
        if m<1:
            m=0
        j=0
        delist=[]
        while j!=m:
            i=0
            for i in range(n):
                if self.U[i]==Us[j] and n!=0:
                    delist.append(i)
                    j=j+1
                if j==m:
                    break
        self.U=np.delete(self.U,delist,0)
        self.U=self.U/sum(self.U)
        self.U=np.delete(self.U,delist,0)
        if n != len(self.U):
            self.flag2=True
            print("reapprox_exp")
        else:
            self.flag2=False
    
    def Particlemaker(self,x,n,Param,Range): #パーティクルを生成する関数
        #itertools.product与えられた変数軍([x1,x2,x3],[y1,y2])の総重複なし組み合わせを配列として出力
        """
        パーティクルを生成する関数
        最小値:x-Range/2
        最大値:x+Range/2
        """
        N=len(x)
        temp=[]
        for i in range(N):
            if(Param[i]==1):
                temp.append(np.linspace(x[i]-Range[i]/2,x[i]+Range[i]/2,n))
            else:
                temp.append([x[i]])
        return(np.array(list(itertools.product(*temp))))
    
    def Expsim(self,x,C): #実験と同様のシーケンスを行いデータの生成を行う関数
        """
        パーティクルxに実験Cで実験シミュレーションを行う関数
        """
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
    
    def Prob_Lookup(self,x,C): #確率のルックアップテーブルを作る。もっと軽量化したい
        """
        確率のルックアップテーブルを作成する
        allの場合,ルックアップテーブルの形は(パーティクル数 * 実験設計数)
        """
        self.mode=1 #Expsimでms=0の確率を算出
        if self.ptable_mode=="cross":
            if len(x) == self.n_particles():#各パーティクルについてある実験Cで実験を行う
                self.ptable_x=np.zeros([self.n_particles(),1])
                for i in range(self.n_particles()):
                    self.ptable_x[i]=self.Expsim(self.x[i],C)  
            else: #あるパーティクルについて全実験設計で実験を行う
                self.ptable_C=np.zeros([self.n_exp(),1])
                for i in range(self.n_exp()):
                    self.ptable_C[i]=self.Expsim(x,self.C[i])
        
        elif (self.ptable_mode=="all"):
            self.ptable=np.zeros([self.n_particles(),self.n_exp()])
            for i in range(self.n_exp()):
                for j in range(self.n_particles()):
                    self.ptable[j,i]=self.Expsim(self.x[j],self.C[i])
                

    def UtilIG(self): #量子操作の効用を計算する関数
        """
        各パーティクルについてエントロピーを計算する。
        エントロピーをかけて効用の分布を更新する
        """
        if self.ptable_mode=="cross":
            ent_table = -self.ptable_C*np.log2(self.ptable_C) #エントロピーテーブル（各要素は各々の実験でのエントロピー)
            ent_array = (np.reshape(ent_table,[len(ent_table),1])) #2D配列に変換
        elif self.ptable_mode=="all":
            w_ptable = self.ptable*self.w #重み付き確率テーブル
            ent_table = -w_ptable*np.log2(w_ptable) #エントロピーテーブル
            ent_array = np.sum(ent_table,axis=0) #各C行使時における情報量(各パーティクルについてエントロピーの和を取る)
            ent_array = (np.reshape(ent_array,[len(ent_array),1])) #2D配列に変換
        self.U=ent_array*self.U
        self.U=self.U/np.sum(self.U)
        self.C_best_i=np.argmax(self.U)
        self.C_best=self.C[self.C_best_i]

    def Update(self): #ベイズ推定を行う関数 引数(self,Ci)
        """
        パーティクルの重みを更新する関数
        """
        self.mode=1
        p_exp=self.Expsim(self.x0,self.C_best)#真値におけるms0の確立
        num=binomial(self.d, p_exp)#実験をd回行いｍs=0であった回数
        if self.ptable_mode=="cross":
            temp=binom.pmf(num,n=self.d,p=self.ptable_x)#各パーティクルでの実験でms=0にいた確率
            self.w=self.w*temp.reshape([len(temp),1]) #重みの更新
        elif self.ptable_mode=="all":
            temp=binom.pmf(num,n=self.d,p=self.ptable)[:,self.C_best_i]#各パーティクルでの実験でms=0にいた確率
            self.w=self.w*temp #重みの更新
        self.w=self.w/np.sum(self.w) #重みの規格化
    
    def UtilIG_bayes_risk(self): #ベイズリスクを計算する関数
        """
        ベイズリスクを計算する関数
        """
        x_infer=self.Mean(self.w,self.x) 
        self.risk.append(np.trace(self.Q*np.dot((self.x - x_infer[0]).T,(self.x - x_infer[0]))))
    
    #=============================結果を描画する関数=============================
    def show_w(self):
        wi=np.linspace(1,self.n_particles(),self.n_particles())
        plt.plot(wi,self.w)
        plt.xlabel("particle")
        plt.ylabel("weight (a.u.)")
        plt.title("weight", fontsize=24)
        plt.show()
        
    def show_U(self):
        Ui=np.linspace(1,self.n_exp(),self.n_exp())
        plt.plot(Ui,self.U)
        plt.xlabel("experiment")
        plt.ylabel("Utility (a.u.)")
        plt.title("Utility", fontsize=24)
        plt.show()
        
    def show_r(self):
        plt.plot(self.i_list,self.risk)
        plt.xlabel("experiment")
        plt.ylabel("Bayes_risk ")
        plt.yscale("log")
        plt.title("Bayes_risk", fontsize=24)
        plt.show()
        
    def show_hyper_parameter(self):
        print("============================ハイパーパラメータの表示======================\n")
        print("実験回数:%d" %(self.i))
        print("リサンプリング強度 %f" %(self.a))
        print("リサンプリング閾値 %f" %(self.resample_threshold))
        print("ベイズリスクの閾値 %f" %(self.bayes_threshold))
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
        print("ルックアップテーブルの表式 %s" %(self.ptable_mode))
        
    
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