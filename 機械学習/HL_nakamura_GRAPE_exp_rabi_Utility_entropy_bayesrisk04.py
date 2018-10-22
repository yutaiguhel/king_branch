# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:01:39 2018

@author: yuta
"""

#============================ライブラリの読み込み=============================
import sys
sys.path.append("C:/Users/yuta/.ipython/profile_default/GRAPE/サブpy")
from GRAPE_single_class04 import *
from numpy.random import*
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['figure.autolayout']=True
mpl.rcParams['axes.grid']=True

#=================================関数の定義==================================


def Mean(w,x): #重み付き平均を計算する関数
    i=0
    n=len(w)
    m=len(x[0])
    mu=np.zeros([1,m])
    for i in range(n):
        mu=mu+w[i][0]*x[i]
    return mu
    
def Hd(p):#情報量から平均エントロピーを計算する関数
    n=len(p)
    i=0
    SUM=0.0
    for i in range(n):
        if p[i]==0:
            p[i]=0.00001
        SUM=SUM-p[i]*np.log2(p[i])
    return SUM

def resample(w,x,a): #パーティクルの移動と重みの再配分を行う関数
    i=0
    n=len(w)
    m=len(x[0])
    mu=Mean(w,x)
    mui=np.zeros([1,m])
    for i in range(n):
        if w[i]<1.0/n*a:
            mui=a*x[i]+(1-a)*mu
            x[i]=mui
        w[i][0]=1.0/n
    print ("resample")
    return w,x

def reapprox(w,x,approx_ratio): #不要となったパーティクルを削除する関数
    n=len(w)
    ws=sorted(w)
    m=floor(n*(1.0-approx_ratio))
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

def Particlemaker(x,n,m,Param,Range): #パーティクルを生成する関数
    l=pow(n,len(Param))
    if m<(len(Param)-1):
        x=Particlemaker(x,n,m+1,Param,Range)
    i=0
    j=0
    for i in range(l):
        x[i][Param[m]]=x[i][Param[m]]+Range[m]*(float(j)/float(n)-1.0/2.0)
        if (i+1)%pow(n,m)==0:
            j=j+1
        if j==n:
            j=0
    return x

def Vd(pulse,omega,C_mat): #ドライブハミルトニアンを生成する関数
    mw=III
    rf=III
    if pulse==1:
        rf=Sx
    else:
        mw=Sx
    vd=omega/2.0*tensor(mw,rf,C_mat)
    return vd

def Hf(pulse,omega,C_mat): #回転座標系に乗せるための行列を生成する関数
    mw=III
    rf=III
    if pulse==1:
        rf=Sz*Sz
    else:
        mw=Sz*Sz
    hf=omega*tensor(mw,rf,C_mat)
    return hf

def Tevo(H0,Vd,HfMW,rho,width): #量子状態の時間発展を計算する関数
    HI=Vd+H0-HfMW
    U=(-2*pi*1j*HI*width).expm()
    dens=U*rho*U.dag()
    return dens

def Tevo_operator(H0,Vd,HfMW,U_init,width):
    HI=Vd+H0-HfMW
    U=(-2*pi*1j*HI*width).expm()
    Ud=U*U_init
    return Ud

def expcal(a,b,c,C_mat): #量子状態の期待値を計算する関数
    #a:density matrix, b:projector of electron, c:Nuclear, d:13C1, f:13C2, g:13C3 
    e=(a*tensor(b,c,C_mat)).tr() #expected value
    e=e.real
    return e

def C_matrix(Ac_list):
    C_list=[]
    for i in range(len(Ac_list)):
        C_list.append(2)
    C_matrix=Qobj(qeye(2**len(Ac_list)),dims=[C_list,C_list])
    return C_matrix

def H(x,Ac_list,C_mat):
    C_list=[]
    Hint_ec=[]
    if len(Ac_list) != 0:
        for i in range(len(Ac_list)): #i番目の炭素の超微細相互作用
            C_z=tensor(Sz,III)
            for j in range(i):
                C_z=tensor(C_z,II)
            C_z=tensor(C_z,sigz)
            for k in range(len(Ac_list)-i-1):
                C_z=tensor(C_z,II)
            Hint_ec.append(Ac_list[i]*C_z)
            C_list.append(2)
        He=x[0]*tensor(Sz*Sz,III,C_mat)
        Hn=x[1]*tensor(III,Sz*Sz,C_mat)
        Hint_en=x[2]*tensor(Sz,Sz,C_mat)
        HB=ge*(x[3])*muB/10.0*tensor(Sz,III,C_mat)
        H0=He+Hn+Hint_en+HB
        for i in range(len(Ac_list)):
            H0=H0+Hint_ec[i]
    else:
        He=x[0]*tensor(Sz*Sz,III)
        Hn=x[1]*tensor(III,Sz*Sz)
        Hint_en=x[2]*tensor(Sz,Sz)
        HB=ge*(x[3])*muB/10.0*tensor(Sz,III)
        H0=He+Hn+Hint_en+HB
    return H0
    

def Expsim(rho0,x,C,C_mat,mode,state,exp,omega_j,U_grape_true): #実験と同様のシーケンスを行いデータの生成を行う関数
    H0=H(x,Ac_list,C_mat)
    HfMW=Hf(0,C[2],C_mat)
    if exp == "GRAPE_rabi_exp":
        rhof=U_grape_true*rho0*U_grape_true.dag()
    if exp == "GRAPE_rabi":
        U0=tensor(III,III,C_mat)
        U=U0
        for i in range(len(omega_j)):
            VdMW=Vd(0,omega_j[i],C_mat)
            U=Tevo_operator(H0,VdMW,HfMW,U,t_div)
        rhof=U*rho0*U.dag()
    if exp == "rabi":
        VdMW=Vd(0,C[0],C_mat)
        rhof=Tevo(H0,VdMW,HfMW,rho0,C[1])
        
    expect0=expcal(rhof,S0,III,C_mat) #ms=0で測定
    if mode==0:
        if rand()>=expect0:
            mes=1 #ms=+1,-1
        else:
            mes=0 #ms=0
    else:
        if state==0:
            mes=expect0
        else:
            mes=1.0-expect0
            if mes<0:
                mes=0

    return mes

def GRAPEpulse(w,xd):
    m=np.argmax(w)
    C_known=[]
    for i in range(len(Ac_list)):
        if Ac_list[i] != 0:
            C_known.append(Ac_list[i])
    grape=GRAPE(xd[m][0],xd[m][1],0,C_known,C_inhomo,Be,theta,phi1,phi2,state_init,
            state_goal,weighting_value,permax,pulse_time,t_div,target_list)

    phi_array, omega_array=grape.optimize()
    return omega_array

def GRAPE_operator(x,omega_j,C_mat):
    H0=H(x,Ac_list,C_mat)
    HfMW=Hf(0,C[2],C_mat)
    C_list=[]
    for i in range(len(Ac_list)):
        C_list.append(2)
    U0=tensor(III,III,C_mat)
    U=U0
    for i in range(len(omega_j)):
        VdMW=Vd(0,omega_j[i],C_mat)
        U=Tevo_operator(H0,VdMW,HfMW,U,t_div)
    return U
    

def Update(rho0,w,x,D,C,C_mat,exp,omega_j,tim0): #ベイズ推定を行う関数
    i=0
    n=len(w)
    nw=np.zeros([n,1])
    d1=len(D)
    d2=sum(D)
    d3=d1-d2
    print("実験回数:%d, ms=±1にいた回数:%f, ms=0にいた回数:%f" %(d1, d2, d3))
    for i in range(n):
        p=Expsim(rho0,x[i],C,C_mat,1,0,"GRAPE_rabi",omega_j,U_grape_true) #ms=0の確率
        nw[i][0]=w[i][0]*pow(p,d3)*pow(1-p,d2) #更新した重み
    #重み更新にかかった時間
    tim1=time.time()
    nw=nw/sum(nw)
    return nw,tim1

def weighting_matrix(x,param):
    Q=np.zeros([len(x0), len(x0)])
    for i in range(len(param)):
        p=param[i]
        Q[p][p]=1
    return Q

def UtilIG(rho0,w,x,C,exp,omega_j): #量子操作の効用を計算する関数
    j=0
    d=2
    p=np.zeros(d)
    pd=np.zeros(d)
    m=argmax(w)
    for j in range(d):
        p[j]=Expsim(rho0,x[m],C,C_mat,1,j,"rabi",omega_j,U_grape_true)
        pd[j]=w[m][0]*p[j]
    In=Hd(p)
    return In

def UtilIG_bayes_risk(x,w,C,exp,Q):
    i=0
    Util=0
    m=np.argmax(w)
    for j in range(d):
        D[j]=Expsim(rho0,x[m],C,C_mat,0,0,exp,omega_j,U_grape_true) #シミュレーションによるデータ取得
    w_new=Update(rho0,w,x,D,C,C_mat,exp,omega_j)
    x_infer=Mean(w_new,x)
    Util=np.trace(Q*np.dot((x - x_infer[0]).T,(x - x_infer[0])))
    return Util
        
def GuEx(rho0,w,x,U,C,Param,Range,exp,omega_j,Utility,Q): #最適な量子操作を特定する関数
    n=len(C)
    Ud=np.zeros([n,1])
    i=0
    for i in range(n):
        if Utility=="entropy":
            Ud[i]=UtilIG(rho0,w,x,C[i],exp,omega_j)*U[i]
        if Utility == "bayes_risk":
            Ud[i]=UtilIG_bayes_risk(x,w,C[i],exp,Q)*U[i]
    Ud=Ud/sum(Ud)
    return Ud

#==========================プログラム起動時刻の記録===========================
start=time.asctime()
print("start:",start)

#===============================物理系の定義==================================
pi=np.pi #円周率
D0=2870 #ゼロ磁場分裂[MHz]
Q=-4.945  #核四重極子分裂[MHz]
An=-2.2 #電子スピン-窒素核スピン間超微細相互作用[MHz]
h=6.62606957*1e-34/(2*pi) #ディラック定数[m**2*kg/s]
ge=2.002319 #電子スピンのg因子
mu=1.5e-7 #真空の透磁率[m**3*s**4*A**2/kg]
muB=927.401*1e-20/h #ボーア磁子[J/T]
muN=5.05078324*1e-27/h #核磁子[J/T]
Be=4.5*1e-5 #地磁気[T]
Bo=-0.450*1e-4 #外部磁場[T]
Ac_list=[-3.265] #電子スピン-炭素同位体核スピン1間超微細相互作用[MHz]
#真のハミルトニアンの定義
x0=[D0,Q,An,Bo+Be]
for i in range(len(Ac_list)):
    x0.append(Ac_list[i])
    
#==========================GRAPEパラメータ定義====================================
theta = (63.4)*pi/180.0 #ワイヤー角度
phi1=0
phi2=0
pulse_time = 1000/1000.0
t_div = 5/1000.0
permax = 4#1000/150/2.0　ラビ周波数の上限
#=========================始状態,終状態設=======================================
C_inhomo=[] #不均一幅の定義
#=============================初期量子状態の定義==============================
psi3e=S0
psi2c1=II/2.0
psi2c2=II/2.0
psi2c3=II/2.0
psi3n=III/3.0
psi3o=S0
C_mat=C_matrix(Ac_list)
rho0=tensor(psi3e,III,C_mat)
#=============================実験パラメータの定義============================
ex=50 #試行回数
n=200 #推定基底毎のパーティクルの数
g=10 #量子操作において変更するパラメータの分割数
d=100 #一度の推定に使用する実験データの数
a=0.75 #パーティクルの再配分における移動強度
resample_threshold=0.5 #パーティクルの再配分を行う判断をする閾値
approx_ratio=0.98 #不要なパーティクルを削除する際の残す割合

D=np.zeros(d)
OMEGA=10 #Rabi周波数の中心[MHz]
t=0.05 #MWパルス長[us]
MWf=2870 #MW周波数の中心[MHz]

C=[OMEGA,t,MWf] #量子操作の候補群の中心
ParamC=[0] #変更する量子操作のパラメータ
RangeC=[1,0,4.0] #変更する範囲

x=[D0+2.0,Q,An,Be+Bo] #パーティクルの中心
for i in range(len(Ac_list)):
    x.append(Ac_list[i])
ParamH=[0] #変更するパーティクルのパラメータ
RangeH=[10] #変更する範囲
Q=weighting_matrix(x0,ParamH)

exp="GRAPE_rabi" #実験手法       
"""
param_all=Parameter(D0,An,Q,C_known,C_inhomo,Be,theta,phi1,phi2,
                                  state_init,state_goal,weighting_value,permax,
                                  pulse_time,t_div,target_list)
"""

"""
===Utility===
entropy
bayes_risk
"""
"""
if exp == "GRAPE_rabi":
    Utility="GRAPE"
else:
    print("効用を以下から選んでください\n")
    print("entropy, bayes_risk")
    Utility=input()
"""
Utility="GRAPE" 
#=======================パーティクルと量子操作群の生成========================
n1=pow(n,len(ParamH))
g1=pow(g,len(ParamC))
w=np.ones([n1,1])
w=w*1.0/float(n1) #一様分布
U=np.ones([g1,1])
U=U*1.0/float(g1) #一様分布
x1=np.zeros([n1,len(x)])
Cg=np.zeros([g1,len(C)])
C1=[]
omega_j=[]
U_grape_true=[]

Xd=[]
i=0
for i in range(n1):
    Xd.append(i)

i=0
for i in range(n1):
    x1[i]=x
x=Particlemaker(x1,n,0,ParamH,RangeH)

i=0
if exp == "rabi":
    for i in range(g1):
        Cg[i]=C
    C=Particlemaker(Cg,g,0,ParamC,RangeC)

index=[]
Xi=[]
X_infer=[]
x_var=[]
X=[]
Y=[]
Z=[]
time.sleep(0.1)
#===========================実験シミュレーション開始==========================
tim0=time.time()
i=0
for i in range(ex):
    tim1=time.time()
    print (tim1-tim0,"sec","\n\n","experiment#",i)
    tim0=tim1
    index.append(i)

    if approx_ratio!=1 and i!=0: #不要なパーティクルの削除
        w,x,de=reapprox(w,x,approx_ratio)
        Xd=np.delete(Xd,de,0)
        if exp == "rabi":
            U,C,nnn=reapprox(U,C,approx_ratio)
            #パーティクルの削除にかかった時間
            tim1=time.time()
            print (tim1-tim0,"sec","\n\n","パーティクルの削除にかかった時間")
            tim0=tim1
        g1=len(U)
    n1=len(w)
    
    if 1.0/sum(w*w)<n1*resample_threshold: #パーティクルの再配分
        w,x=resample(w,x,a)
        #リサンプリングにかかった時間
        tim1=time.time()
        print (tim1-tim0,"sec","\n\n","リサンプリングにかかった時間")
        tim0=tim1
    
    if exp == "rabi":
        if 1.0/sum(U*U)<g1*resample_threshold: #量子操作群の再配分
            U,C=resample(U,C,a)
        U=GuEx(rho0,w,x,U,C,ParamC,RangeC,exp,omega_j,Utility,Q) #最適な量子操作の特定
        ibest=np.argmax(U)
        C1=C[ibest]
        print(C1)
        j=0
        for j in range(d):
            D[j]=Expsim(rho0,x0,C1,C_mat,0,0,"rabi",omega_j,U_grape_true) #実験によるデータ取得
            
    if exp == "GRAPE_rabi":
        list_size = 10
        state_init = list(zeros(list_size))
        state_goal = list(zeros(list_size))
        weighting_value = list(zeros(list_size))
        target_list = [0]
        state_init[0] = tensor(S0,III/3.0,II/2.0)#,II/2.0)
        state_goal[0] = tensor(Sp,III/3.0,II/2.0)/2.0+tensor(Sm,III/3.0,II/2.0)/2.0#ket2dm(tensor(plus1,zero,up)+tensor(minus1,zero,down))#,II/2.0)
        weighting_value[0] = 1
        state_init[1] = tensor(S0,III/3.0,II/2.0)#,II/2.0)
        state_goal[1] = tensor(SH,III/3.0,II/2.0)#,II/2.0)
        weighting_value[1] = 0               
        state_init[2] = tensor(SV,III/3.0,II/2.0)#,II/2.0)
        state_goal[2] = tensor(SV,III/3.0,II/2.0)#,II/2.0)
        weighting_value[2] = 0   
        omega_j=GRAPEpulse(w,x)
        #GRAPE波形取得にかかった時間
        tim1=time.time()
        print (tim1-tim0,"sec","\n\n","GRAPE波形取得にかかった時間")
        tim0=tim1
        x_mean=Mean(w,x)
        print("現在推定したハミルトニアン\n")
        print(x_mean[0])
        for j in range(d):
            if j == 0:
                U_grape_true=GRAPE_operator(x_mean[0],omega_j,C_mat)
                #GRAPE演算子取得にかかった時間
                tim1=time.time()
                print (tim1-tim0,"sec","\n\n","GRAPE演算子取得にかかった時間")
                tim0=tim1
            D[j]=Expsim(rho0,x0,C,C_mat,0,0,"GRAPE_rabi_exp",omega_j,U_grape_true) #実験によるデータ取得
        #実験データ生成にかかった時間
        tim1=time.time()
        print (tim1-tim0,"sec","\n\n","実験データ生成にかかった時間")
        tim0=tim1
        C1=C

    w,tim1=Update(rho0,w,x,D,C1,C_mat,exp,omega_j,tim0) #得られた実験データを用いてベイズ推定と重みの更新
    #重み更新にかかった時間
    print (tim1-tim0,"sec","\n\n","重みの更新")
    tim0=tim1
    
    j=0
    for j in range(n1):
        X.append(Xd[j])
        Xi.append(x[j][0]) #各試行でのパーティクル
        Y.append(i) #試行回数
        Z.append(w[j][0]) #重みの配列
    xout=Mean(w,x)
    print(xout[0]) #推定したハミルトニアンを出力
    var=np.var(Xi,keepdims=True) #パーティクルの分散の計算
    x_var.append(var) 
    print("分散: %f" %var) #分散の表示
    X_infer.append(xout[0][0]) #推定したハミルトニアン
    #データ解析にかかった時間
    tim1=time.time()
    print (tim1-tim0,"sec","\n\n","データ解析")
    tim0=tim1
    
    #重みを表示
    wi=np.linspace(1,n1,n1)
    plt.plot(wi,w)
    plt.xlabel("particle")
    plt.ylabel("weight (a.u.)")
    plt.title("weight", fontsize=24)
    plt.show()
    #====================パーティクルの分散を表示===================================
    fig1=plt.figure()
    ax1=plt.axes()
    ax1.set_xlim(0, max(index))
    ax1.set_ylim(0, max(x_var))
    ax1.set_xlabel("iteration#")
    ax1.set_ylabel("variance")
    plt.plot(index, x_var)
    plt.title("Variance")
    plt.show()
    #====================パーティクルの分散を表示===================================
    fig2=plt.figure()
    ax2=plt.axes()
    ax2.set_xlim(0, max(index))
    ax2.set_ylim(min(X_infer), max(X_infer))
    ax2.set_xlabel("iteration#")
    ax2.set_ylabel("D0 (MHz)")
    plt.axhline(y=D0,color="r")
    plt.plot(index, X_infer)
    plt.title("Infered_Particle")
    plt.show()
    
    #データ表示にかかった時間
    tim1=time.time()
    print (tim1-tim0,"sec","\n\n","データの表示")
    tim0=tim1
print(x[np.argmax(w)]) #最終的に最も確からしいと考えられたハミルトニアンを出力

print("============================ハイパーパラメータの表示======================\n")
print("実験回数:%d" %ex)
print("実験で使ったデータ数:%d" %d)
print("使用した効用:%s" %Utility)
print("最初に用意したパーティクルの数:%d" %n)
print("現在のパーティクルの数:%d" %len(w))
#===========================推定終了時刻を表示================================
finish=time.asctime()
print("finish:",finish)
