# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:01:39 2018

@author: yuta
"""

#================================ライブラリの読み込み===============================
import sys
sys.path.append("C:/Users/yuta/.ipython/profile_default/GRAPE/new")
sys.path.append("C:/Users/yuta/.ipython/profile_default/機械学習/サブpy")
from Q_H01 import*
from Q_module_grape_qutip_koga02 import*
from Bayes_function05 import*
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#==============================プログラム起動時刻の記録=============================
start=time.asctime()
print("start:",start)
#===============================インスタンス生成===================================
m=Bayes_Function()
#================================ベイズ推定の設定=================================
m.ptable_mode="cross" #cross or all
#===============================パラメータの変更===================================
m.ex=200 #試行回数
m.d=1000 #推定に使う実験データの数
m.n=10 #パーティクルの分割数
m.g=10 #実験設計の分割数
m.bayes_threshold=1
m.Ac_list=[-3.265,-6.5]
m.ParamH=[1,1,0,0,1,0] #D0,AN,QN,外部磁場,*炭素
m.RangeH=[5,0.5,0,0,0.5,0] #推定パラメータの広げる範囲
m.ParamC=[1,0,1] #ラビ周波数,MWwidth,MWfreq
m.RangeC=[5,0,10] #実験設計パラメータの拡張範囲
m.params() #パラメータの変更を反映
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
time.sleep(0.1)
#===============================実験シミュレーション開始=============================
tim0=time.time()
for i in range(m.ex):
    print("experiment#", i)
    m.i=i
    (m.i_list).append(i)
    
    #不要なパーティクルの削除
    if m.approx_ratio!=1 and i!=0: 
        m.reapprox_par()
        m.reapprox_exp()
    
    #パーティクルのリサンプリング
    if 1.0/sum(m.w*m.w)<len(m.w)*m.resample_threshold: #パーティクルの再配分
        m.w,m.x=m.resample(m.w,m.x)
        m.flag1=True
        
    #量子操作のリサンプリング
    if 1.0/sum(m.U*m.U)<len(m.U)*m.resample_threshold: #量子操作群の再配分
        m.U,m.C=m.resample(m.U,m.C)
        m.flag2=True
    
    #重み最大のパーティクルについて確率のルックアップテーブルを作成.
    flag=m.flag1|m.flag2
    if flag==True or i==0:
        m.Prob_Lookup(m.x[np.argmax(m.w)],m.C) #ptable_Cを用意する
          
    #効用の計算
    m.UtilIG()
    
    #効用最大の実験設計で確率のルックアップテーブルを作成
    flag=m.flag1|m.flag2
    if flag==True or i==0:
        if m.ptable_mode=="cross":
            m.Prob_Lookup(m.x,m.C_best) #ptable_xを用意するs
    
    #ベイズ推定
    m.Update()#ベイズ推定
    
    #ベイズリスクの計算
    m.UtilIG_bayes_risk()
    print("現在のベイズリスク",m.risk[i])

    xout=m.Mean(m.w,m.x)
    print("推定したハミルトニアン",xout[0]) #推定したハミルトニアンを出力
    
    m.show_w() #重みの表示
    m.show_U() #効用の表示
    m.show_r() #ベイズリスクの表示
    
    #1推定にかかった時間
    tim1=time.time()
    print (tim1-tim0,"sec","\n")
    tim0=tim1
    
    #推定を続けるか判断する
    if m.risk[i] < m.bayes_threshold:
        print("=======================End of estimation======================")
        break

print("最も確からしいハミルトニアン",m.x[np.argmax(m.w)]) #最終的に最も確からしいと考えられたハミルトニアンを出力
m.show_hyper_parameter()
#===========================推定終了時刻を表示================================
finish=time.asctime()
print("finish:",finish)
