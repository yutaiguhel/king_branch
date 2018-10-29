# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:01:39 2018

@author: yuta
"""

#============================ライブラリの読み込み=============================
import sys
sys.path.append("C:/Users/yuta/.ipython/profile_default/GRAPE/サブpy")
sys.path.append("C:/Users/yuta/.ipython/profile_default/機械学習/サブpy")
from Q_H01 import*
from GRAPE_single_class04 import *
from Bayes_function02 import*
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['figure.autolayout']=True
mpl.rcParams['axes.grid']=True
#==========================プログラム起動時刻の記録===========================
start=time.asctime()
print("start:",start)
#===============================インスタンス生成===================================
m=Bayes_Function()
#===============================パラメータの変更===================================
m.d=1000
#==============================重み、パーティクルの初期化============================
m.init_x0() #真のハミルトニアンに炭素追加
m.init_x() #パーティクルに炭素追加
m.init_C() #実験設計の初期化
m.weighting_matrix() #ベイズリスクを計算する際の重み行列を初期化
m.init_w() #重みの初期化
m.init_U() #効用の初期化
#===========================パーティクルと量子操作群の生成===========================
#パーティクルの作成
m.x=m.Particlemaker(m.x,m.n,m.ParamH,m.RangeH)
#実験候補の作成
m.C=m.Particlemaker(m.C,m.g,m.ParamC,m.RangeC)
Xd=[]
for i in range(m.n_particles()):
    Xd.append(i)
index=[]
Xi=[]
X_infer=[]
x_var=[]
X=[]
Y=[]
Z=[]
time.sleep(0.1)
#===============================実験シミュレーション開始=============================
tim0=time.time()
for i in range(m.ex):
    tim1=time.time()
    print (tim1-tim0,"sec","\n\n","experiment#",i)
    tim0=tim1
    index.append(i)
    i=m.i
    
    flag1=False#resample発生識別
    flag2=False#resample発生識別
    
    #不要なパーティクルの削除
    if m.approx_ratio!=1 and i!=0: #不要なパーティクルの削除
        m.w,m.x,de=m.reapprox(m.w,m.x)
        Xd=np.delete(Xd,de,0)
        m.U,m.C,nnn=m.reapprox(m.U,m.C)
    
    #パーティクルのリサンプリング
    if 1.0/sum(m.w*m.w)<len(m.w)*m.resample_threshold: #パーティクルの再配分
        m.w,m.x=m.resample(m.w,m.x)
        flag1=True
        
    #量子操作のリサンプリング
    if 1.0/sum(m.U*m.U)<len(m.U)*m.resample_threshold: #量子操作群の再配分
        m.U,m.C=m.resample(m.U,m.C)
        flag2=True
     
    #重み最大のパーティクルについて確率のルックアップテーブルを作成.
    flag=flag1|flag2
    if flag==True or i==0:
        m.Prob_Lookup(m.x[np.argmax(m.w)],m.C)
    
    #効用の計算
    m.UtilIG()#効用を計算
    
    #効用最大の実験設計で確率のルックアップテーブルを作成
    flag=flag1|flag2
    if flag==True or i==0:
        m.Prob_Lookup(m.x,m.C_best)
    
    #ベイズ推定
    m.Update()#ベイズ推定

    for j in range(m.n_particles()):
        X.append(Xd[j])
        Xi.append(m.x[j][0]) #各試行でのパーティクル
        Y.append(i) #試行回数
        Z.append(m.w[j][0]) #重みの配列
    xout=m.Mean(m.w,m.x)
    print(xout[0]) #推定したハミルトニアンを出力
    var=np.var(Xi,keepdims=True) #パーティクルの分散の計算
    x_var.append(var) 
    print("分散: %f" %var) #分散の表示
    X_infer.append(xout[0][0]) #推定したハミルトニアン
    
#================================結果の描画=====================================
    #===============================重みを表示==================================
    wi=np.linspace(1,m.n_particles(),m.n_particles())
    plt.plot(wi,m.w)
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
    plt.axhline(y=m.D0,color="r")
    plt.plot(index, X_infer)
    plt.title("Infered_Particle")
    plt.show()

print(m.x[np.argmax(m.w)]) #最終的に最も確からしいと考えられたハミルトニアンを出力

print("============================ハイパーパラメータの表示======================\n")
print("実験回数:%d" %(m.ex))
print("実験で使ったデータ数:%d" %(m.d))
print("使用した効用:%s" %(m.Utility))
print("最初に用意したパーティクルの数:%d" %(m.n))
print("現在のパーティクルの数:%d" %len(m.w))
#===========================推定終了時刻を表示================================
finish=time.asctime()
print("finish:",finish)
