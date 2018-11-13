# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:01:39 2018

@author: yuta
"""
"""
###############################################################################
#                                                                             #
#                     Bayes_functionに機能の説明があります                        #
#                                                                             #
###############################################################################
使用上の注意
・電圧は0Vにしない。Range,中心値に注意
"""
#================================ライブラリの読み込み===============================
import sys
#sys.path.append("C:/Users/yuta/.ipython/profile_default/GRAPE/new")
sys.path.append("C:/koga/実験系/king_branch/機械学習/サブpy")
sys.dont_write_bytecode = True #__pycache__の生成を防ぐ
from Q_H07 import*
#from Q_module_grape_qutip_koga02 import*
from Bayes_function16 import*
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#==============================プログラム起動時刻の記録=============================
start=time.asctime()
print("start:",start)
#===============================インスタンス生成===================================
m=Bayes_Function()
#================================ベイズ推定の設定=================================
m.ptable_mode="cross" #cross or all
#===============================パラメータの変更===================================
m.ex=1000 #試行回数
m.d=1000 #推定に使う実験データの数
m.a=0.98 #リサンプリング強度
m.approx_ratio=0.99 #パーティクルを残す割合
m.resampling_threshold=0.2 #リサンプリング閾値
m.bayes_threshold=10
m.wire=1
#============================実験パラメータの変更==================================
m.V1=1.5 #ワイヤ1の電圧
m.t=2.0 #MWwidth
m.tw=1.0 #wait time
m.g=     {"V1":5,"V2":30,"phi":10,  "MWwidth":5, "MWfreq":10,"tw":5} #量子操作において変更するパラメータの分割数 V1,V2,phi,t,MW_freq
m.ParamC={"V1":1, "V2":0, "phi":0,  "MWwidth":1,  "MWfreq":0,"tw":0} #V1,V2,phi,MWwidth,MWfreq 変更する場合は1
m.RangeC={"V1":2.9, "V2":1, "phi":2*np.pi,"MWwidth":3.98,"MWfreq":4,"tw":1.9} #実験設計パラメータの拡張範囲
#============================推定パラメータの変更==================================
m.a1=20
m.b1=1.0
m.a2=15
m.b2=0.95
m.Bo=-1.26 #外部磁場[MHz] 地磁気を打ち消すならば-1.26MHz
m.Ac_list=[]#[-3.265]
#炭素数に応じてParamH,RangeHの数だけParamH,RangeHを増やす
m.n=     {"a1":5, "b1":5, "a2":10,  "b2":10,  "w_theta":10,     "D0":10,"AN":5,"QN":0,"Bz":10} #推定基底毎のパーティクルの数 a1,b1,a2,b2,w_theta,D0,An,Qn,Bz *Ac
m.ParamH={"a1":1,  "b1":1,  "a2":0,  "b2":0,  "w_theta":0,      "D0":0,"AN":0,"QN":0,"Bz":0} #a1,b1,a2,b2,w_theta,D0,AN,QN,外部磁場,*炭素 変更する場合は1
m.RangeH={"a1":30, "b1":1,  "a2":20,"b2":0.8,"w_theta":2*np.pi,"D0":3,"AN":0.05,"QN":0,"Bz":8} #推定パラメータの広げる範囲
m.params() #パラメータの変更を反映
#==============================重み、パーティクルの初期化============================
m.init_x0() #真のハミルトニアンに炭素追加
m.init_x() #パーティクルに炭素追加
m.init_C() #実験設計の初期化
m.init_w() #重みの初期化
m.init_U() #効用の初期化
#==============================デバッグ用========================================
a_list=[]
b_list=[]
t_list=[]
V_list=[]
a1_max_list=[]
a1_min_list=[]
b1_max_list=[]
b1_min_list=[]
#===========================パーティクルと量子操作群の生成===========================
m.x=m.Particlemaker(m.x,m.n,m.ParamH,m.RangeH)#パーティクルの作成
m.weighting_matrix() #ベイズリスクを計算する際の重み行列を初期化, パーティクル生成後に呼び出す
m.C=m.Expmaker() #実験候補の作成
time.sleep(0.1)
#===============================実験シミュレーション開始=============================
print(m.x_dict) #始めのパーティクルの中心を表示
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
    for j in range(2):
        if 1.0/sum(m.U[j]*m.U[j])<len(m.U[j])*m.resample_threshold: #量子操作群の再配分
            m.U[j],m.C[j]=m.resample(m.U[j],m.C[j])
            m.flag2=True
    
    #確率のルックアップテーブルを作成.
    flag=m.flag1|m.flag2
    if flag==True or i==0:
        m.Prob_Lookup() #ptableを用意する
        
          
    #効用の計算
    #m.UtilIG() #効用を平均情報量とする場合
    m.UtilIG_bayes_risk() #効用をベイズリスクとする場合
    
    """
    t_list.append(m.C_best[3])
    plt.plot(m.i_list,t_list)
    plt.title("MWwidth",fontsize=24)
    plt.show()
    
    V_list.append(m.C_best[0])
    plt.plot(m.i_list,V_list)
    plt.title("V1",fontsize=24)
    plt.show()
    """
    #ベイズ推定
    m.Update()#ベイズ推定
    
    #ベイズリスクの計算
    m.Bayes_risk()
    print("現在のベイズリスク",m.risk[i])

    xout=m.Mean(m.w,m.x)
    print("推定したハミルトニアン",xout[0]) #推定したハミルトニアンを出力
    
    #実験設計を初期設定に戻す
    
    #==========================デバッグ用結果描画=================================
    m.Show_result()
    
    plt.figure(figsize=(10,7))
    
    a_list.append(xout[0][0])
    pa=m.Region_edge(0.95,"a1")
    a1_min_list.append(pa[0])
    a1_max_list.append(pa[1])
    plt.subplot(2,1,1)
    plt.hlines(m.x0_dict["a1"],0,m.i_list[i],"r",label="True a1")
    plt.xlabel("iteration number",fontsize=20)
    plt.ylabel("a1",fontsize=20)
    #plt.fill_between(m.i_list,a1_min_list,a1_max_list,facecolor='y',alpha=0.5)
    plt.plot(m.i_list,a_list,label="Infered a1")
    plt.legend(loc="upper right")
    plt.title("a1",fontsize=24)
    
    plt.subplot(2,1,2)
    plt.hlines(m.x0_dict["b1"],0,m.i_list[i],"r",label="True b1")
    plt.xlabel("iteration number",fontsize=20)
    plt.ylabel("b1",fontsize=20)
    b_list.append(xout[0][1])
    plt.plot(m.i_list,b_list,label="Infered b1")
    plt.legend(loc="upper right")
    plt.title("b1",fontsize=24)

    plt.tight_layout()
    plt.show()
    
    
    #m.Show_region(0.95)
        
    #1推定にかかった時間
    tim1=time.time()
    print (tim1-tim0,"sec","\n")
    tim0=tim1
    #推定を続けるか判断する
    if m.risk[i] < m.bayes_threshold:
        print("=======================End of estimation======================")
        break


print("最も確からしいハミルトニアン",xout[0]) #最終的に最も確からしいと考えられたハミルトニアンを出力
m.show_hyper_parameter()
#===========================推定終了時刻を表示================================
finish=time.asctime()
print("finish:",finish)
