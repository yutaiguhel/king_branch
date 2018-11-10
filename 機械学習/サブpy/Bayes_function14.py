# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:08:39 2018

@author: yuta
"""
import time
from Q_H07 import*
import itertools
from scipy.stats import binom
from numpy.random import*
import matplotlib.pyplot as plt
"""
このクラスはベイズ推定のためのクラスです.
クロスワイヤのドライブハミルトニアンを生成できます.
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
        self.n={"a1":5,"b1":5,"a2":5,"b2":5,"w_theta":5,"D0":5,"AN":5,"QN":5,"Bz":5} #推定基底毎のパーティクルの数 a1,b1,a2,b2,w_theta,D0,An,Qn,Bz
        self.g={"V1":5,"V2":5,"phi":30,"MWwidth":5,"MWfreq":5} #量子操作において変更するパラメータの分割数 V1,V2,phi,MWwidth,MWfreq
        self.d=1000 #一度の推定に使用する実験データの数
        self.a=0.75 #パーティクルの再配分における移動強度
        self.resample_threshold=0.5 #パーティクルの再配分を行う判断をする閾値
        self.approx_ratio=0.98 #不要なパーティクルを削除する際の残す割合
        self.bayes_threshold=1 #推定を終えるベイズリスクの閾値
        self.mode=1 #Expsimの測定モード(1:確率,0:射影測定)
        self.state=0 #0:ms=0で射影測定,1:ms=±1で射影測定
        self.flag1=False #パーティクルの数が変化したらTrue
        self.flag2=False #実験設計の数が変化したらTrue
        self.exp_flag="rabi"
        self.Q=0 #ベイズリスクの重み行列
        self.U=0 #効用
        self.B_R=0 #ベイズリスク
        self.Utility="binomial"
        self.ptable_mode="all" #all or cross
        self.num=0 #ms=0にいた回数
        self.p_exp=0 #真値におけるms=0にいた確率
        #結果格納配列
        self.i_list=[]
        self.ptable_C=[]
        self.ptable_x=[]
        self.ptable=[]
        self.risk=[]
        self.exp_list=[0]
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
        self.params_list=["a1","b1","a2","b2","w_theta","D0","AN","QN","Bz"]
        self.x0_dict={"a1":self.a1,"b1":self.b1,"a2":self.a2,"b2":self.b2,"w_theta":self.w_theta
                 ,"D0":self.D0,"AN":self.AN,"QN":self.QN,"Bz":self.Be+self.Bo} #真のハミルトニアン
        self.x0=[self.x0_dict["a1"],self.x0_dict["b1"],self.x0_dict["a2"],self.x0_dict["b2"],self.x0_dict["w_theta"]
                ,self.x0_dict["D0"],self.x0_dict["AN"],self.x0_dict["QN"],self.x0_dict["Bz"]]
        self.D=np.empty([self.d,1])
        self.x_dict={"a1":self.a1+float(np.random.random(1)*self.a1/4)-self.a1/8,"b1":self.b1+float(np.random.random(1)*self.b1/8)-self.b1/16,"a2":self.a2-self.a2/10,"b2":self.b2+self.b2/5,"w_theta":self.w_theta
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
                return int(n_C/self.g["tw"])
        
        else:
            if exp=="ramsey":
                n_C=len(self.U[1])
            elif exp=="rabi":
                n_C=len(self.U[0])
            else:
                n_C=len(self.U[0])+len(self.U[1])
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
        self.U=[np.ones([n_C_rabi,1])/(n_C_rabi+n_C_ramsey),np.ones([n_C_ramsey,1])/(n_C_rabi+n_C_ramsey)]
    
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
                px=1/np.var(self.x.T[i]) #各パラメータの初期分散
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
            print("reapprox_par")
        else:
            self.flag1=False
    
    def reapprox_exp(self): #不要となったパーティクルを削除する関数
        """
        m:残すパーティクルの数
        ws:昇順に並び替えた重み
        wsの(m+1)番目の要素よりも大きい重みのパーティクルは残す
        """
        for i in range(2):
            n=len(self.U[i])
            Us=sorted(self.U[i])
            m=floor(n*(1.0-self.approx_ratio))
            if m<1:
                m=0
            j=0
            delist=[]
            while j!=m:
                for l in range(n):
                    if self.U[i][l]==Us[j] and n!=0:
                        delist.append(l)
                        j=j+1
                    if j==m:
                        break
            self.U[i]=np.delete(self.U[i],delist,0)
            self.U[i]=self.U[i]/sum(self.U[i])
            self.U[i]=np.delete(self.U[i],delist,0)
            if n != len(self.U[i]):
                self.flag2=True
                print("reapprox_exp"+str(i))
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
        for i,p in enumerate(Param):
            if(Param[p]==1):
                temp.append(np.linspace(x[i]-Range[p]/2,x[i]+Range[p]/2,n[p]))
            else:
                temp.append([x[i]])
        return(np.array(list(itertools.product(*temp))))
    
    def Expmaker(self):
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
        return([np.array(list(itertools.product(*temp_rabi))),np.array(list(itertools.product(*temp)))])
    
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
    
    def Prob_Lookup_x(self): #確率のルックアップテーブルを作る。もっと軽量化したい
        """
        確率のルックアップテーブルを作成する
        allの場合,ルックアップテーブルの形は(パーティクル数 * 実験設計数)
        """
        self.mode=1 #Expsimでms=0の確率を算出
        self.ptable_x=np.zeros([self.n_particles(),1])
        for i in range(self.n_particles()):
            if self.exp_flag=="rabi":
                self.ptable_x[i]=self.Expsim(self.x[i],self.C_best[0])
            elif self.exp_flag=="ramsey":
                self.ptable_x[i]=self.Expsim(self.x[i],self.C_best[1])
    
    def Prob_Lookup_C(self):
        self.exp_flag="rabi"
        self.ptable_C=[np.zeros([self.n_exp("rabi"),1]),np.zeros([self.n_exp("ramsey"),1])]
        for i in range(2):
            if i==1:
                self.exp_flag="ramsey"
            if self.i==0:
                self.C_best_i=int(self.n_exp(self.exp_flag)*np.random.random())
                self.C_best=self.C[i][self.C_best_i]
            else:
                for j in range(self.n_exp(self.exp_flag)):
                    self.ptable_C[i][j]=self.Expsim(self.x[np.argmax(self.w)],self.C[i][j])
                    
    def Entropy(self,p):
        """
        p(確率が格納された配列)の各要素について平均情報量を計算する関数
        """
        if 0 in p:
            p[p.index(0)]=0.00000000001
        if 1 in p:
            p[p.index(1)]=0.99999999999
        return -p*np.log2(p)-(1-p)*np.log2(1-p)

    def UtilIG(self):
        """
        各パーティクルについてエントロピーを計算する。
        エントロピーをかけて効用の分布を更新する
        """
        ent_table = self.Entropy(self.ptable_C)#-self.ptable_C*np.log2(self.ptable_C) #エントロピーテーブル（各要素は各々の実験でのエントロピー)
        ent_array = (np.reshape(ent_table,[len(ent_table),1])) #2D配列に変換
        self.U=ent_array*self.U
        self.U=self.U/np.sum(self.U)
        self.C_best_i=np.argmax(self.U)
        self.C_best=self.C[self.C_best_i]
        
    def UtilIG_bayes_risk(self):
        """
        効用としてベイズリスクを計算する関数
        推定1回目はランダムに実験を選ぶ
        """
        if self.i==0:
            self.C_best_i=[np.argmin(self.U[0]),np.argmin(self.U[1])]
            self.C_best=[self.C[0][self.C_best_i[0]], self.C[1][self.C_best_i[1]]]
        else:
            m=np.argmax(self.w)
            L_w=np.ones(self.n_particles())
            dU=[np.zeros([self.n_exp("rabi"),1]),np.zeros([self.n_exp("ramsey"),1])]
            exp="rabi"
            #それぞれの実験で効用を計算
            for j in range(2):
                if j==1:
                    exp="ramsey"
                for i in range(self.n_exp(exp)):
                    L_w[m]=binom.pmf(self.num[j],n=self.d,p=self.ptable_C[j][i])#各パーティクルでの実験でms=0にいた確率
                    w_new=self.w*L_w.reshape([len(L_w),1]) #重みの更新
                    x_infer=self.Mean(w_new,self.x)
                    dU[j][i]=np.trace(self.Q*np.dot((self.x - x_infer[0]).T,(self.x - x_infer[0]))) #実験C[i]でのベイズリスク
                self.U[j]=dU[j]
            U_min=[] #効用が最小となる指標、効用の最小値を格納する配列
            for j in range(2):
                self.U[j]=self.U[j]/(np.sum(self.U[0])+np.sum(self.U[1]))
                U_min.append([np.min(self.U[j]),np.argmin(self.U[j])])
            print(U_min[0][0],U_min[1][0])
            #ラビ振動の方が良い場合
            if U_min[0][0] < U_min[1][0]:
                self.exp_flag="rabi"
                self.exp_list.append(0)
            #ラムゼー干渉の方が良い場合
            else:
                self.exp_flag="ramsey"
                self.exp_list.append(1)
            self.C_best_i=[U_min[0][1],U_min[1][1]]
            self.C_best=[self.C[0][self.C_best_i[0]], self.C[1][self.C_best_i[1]]]
            
    def Update(self): #ベイズ推定を行う関数 引数(self,Ci)
        """
        パーティクルの重みを更新する関数
        """
        #ラビ振動とラムゼー干渉の実験を行う
        exp_flag_best=self.exp_flag
        self.mode=1
        self.p_exp=[]
        self.exp_flag="rabi"
        self.p_exp.append(self.Expsim(self.x0,self.C_best[0]))#真値におけるms0の確立
        self.exp_flag="ramsey"
        self.p_exp.append(self.Expsim(self.x0,self.C_best[1]))#真値におけるms0の確立
        self.num=binomial(self.d, self.p_exp)#実験をd回行いｍs=0であった回数
        self.exp_flag=exp_flag_best
        if exp_flag_best=="rabi":
            _num=self.num[0]
        elif exp_flag_best=="ramsey":
            _num=self.num[1]
        temp=binom.pmf(_num,n=self.d,p=self.ptable_x)#各パーティクルでの実験でms=0にいた確率
        self.w=self.w*temp.reshape([len(temp),1]) #重みの更新
        self.w=self.w/np.sum(self.w) #重みの規格化
    
    def Bayes_risk(self): #ベイズリスクを計算する関数
        """
        ベイズリスクを計算する関数
        """
        x_infer=self.Mean(self.w,self.x) 
        self.risk.append(np.trace(self.Q*np.dot((self.x - x_infer[0]).T,(self.x - x_infer[0]))))
    
    #=============================結果を描画する関数=============================
    def show_w(self):
        """
        現在の重みを描画する関数
        """
        wi=np.linspace(1,self.n_particles(),self.n_particles())
        plt.plot(wi,self.w)
        plt.xlabel("particle")
        plt.ylabel("weight (a.u.)")
        plt.title("weight", fontsize=24)
        plt.show()
        
    def show_U_rabi(self):
        """
        現在の効用を描画する関数
        """
        Ui=np.linspace(1,self.n_exp("rabi"),self.n_exp("rabi"))
        plt.plot(Ui,self.U[0])
        plt.xlabel("experiment")
        plt.ylabel("Utility (a.u.)")
        plt.title("Utility Rabi", fontsize=24)
        plt.show()
        
    def show_U_ramsey(self):
        """
        現在の効用を描画する関数
        """
        Ui=np.linspace(1,self.n_exp("ramsey"),self.n_exp("ramsey"))
        plt.plot(Ui,self.U[1])
        plt.xlabel("experiment")
        plt.ylabel("Utility (a.u.)")
        plt.title("Utility Ramsey", fontsize=24)
        plt.show()
        
    def show_r(self):
        """
        ベイズリスクの推移を描画する関数
        """
        plt.plot(self.i_list,self.risk)
        plt.xlabel("experiment")
        plt.ylabel("Bayes_risk ")
        plt.yscale("log") #y軸をlogスケールに
        plt.title("Bayes_risk", fontsize=24)
        plt.show()
        
    def show_exp(self):
        plt.plot(self.i_list,self.exp_list)
        plt.xlabel("iteration#")
        plt.ylabel("exp")
        plt.title("Experiment", fontsize=24)
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