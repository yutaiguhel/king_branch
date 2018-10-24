# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 06:30:16 2018

@author: yuta
"""
#=============================モジュールのインポート=================================
from qinfer import*
import numpy as np
import matplotlib.pyplot as plt
#===============================クラス定義=======================================
class Rabi(FiniteOutcomeModel):
    def __init__(self):
        #===========================ベイズ推定パラメータ===========================
        self.n_trial=10 #実験回数
        self.n_particles=1000 #パーティクルの数
        self.n_data=1000 #実験データ数
        super(Rabi,self).__init__()
        #=============================実験パラメータ==============================
        self.num_rabi=10
        self.rabi_min=10
        self.rabi_max=110
        self.num_MW_freq=10
        self.MW_freq_min=2865
        self.MW_freq_max=2875
        self.num_t_pi=10
        self.t_pi_min=10
        self.t_pi_max=110
        #============================モデルパラメータ==============================
        self.D0=2870
        self.D0_min=2865
        self.D0_max=2875
    
    @property
    def n_modelparams(self):#母数の数
        return 1
    @property
    def is_n_outcomes_constant(self):
        return True
    @property
    def expparams_dtype(self):#実験パラメータのデータ型を定義
        return [('rabi_freq', 'float'), ('MW_freq', 'float'), ('t_pi', 'float')]
    
    def expparams_dtype_(self):#実験パラメータのデータ型を定義
        return [('rabi_freq', 'float'), ('MW_freq', 'float'), ('t_pi', 'float')]
    
    def n_outcomes(self, expparams):#実験結果の数(ms=0, ms=±1にいる確率)
        return 2
    
    def are_models_valid(self, modelparams):#母数の範囲を制限.
        return np.all(np.logical_and(modelparams > 2864, modelparams <= 2876), axis=1)
    
    def exp_maker(self):
        rabi_freq = np.linspace(self.rabi_min, self.rabi_max, self.num_rabi)
        MW_freq = np.linspace(self.MW_freq_min, self.MW_freq_max, self.num_MW_freq)
        t_pi = np.linspace(self.t_pi_min, self.t_pi_max, self.num_t_pi)
        for rabi in rabi_freq:
            for MW in MW_freq:
                for t in t_pi:
                    if (((rabi==self.rabi_min) and (MW==self.MW_freq_min)) and (t==self.t_pi_min)):
                        exp = np.array([(rabi,MW,t),],dtype=Rabi.expparams_dtype_(self))
                    else:
                        exp_new = np.array([(rabi,MW,t),],dtype=Rabi.expparams_dtype_(self))
                        exp = np.vstack((exp, exp_new))
        return exp.reshape(self.num_rabi*self.num_MW_freq*self.num_t_pi,)
    
    def likelihood(self, outcomes, modelparams, expparams):
        #super(Rabi, self).likelihood(outcomes, modelparams, expparams)
        pr0 = np.cos(np.dot(modelparams[:], expparams['t_pi'].reshape(1,-1)))**2
        """
        outcomes: 整数で表される結果(ここでは0or1) ndarray
        pro0: 実験結果が1となる確率, ndarray (model# * exp_parameter#)
        """
        return FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)

#===============================インスタンス生成===================================
m = Rabi()
prior = UniformDistribution([m.D0_min, m.D0_max])
updater = SMCUpdater(m, m.n_particles, prior ,resampler=LiuWestResampler(0.9))
updater.plot_posterior_marginal()
#===============================実験パラメータ生成=================================
expparams = m.exp_maker() #shape (m.num_rabi * m.num_MW_freq * m.num_t_pi,)
#===============================真の値ベクトル====================================
D0_vec = np.array([m.D0])
#===================================メイン=======================================
for idx_exp in range(m.n_trial):
    #===============================最適な実験を選ぶ=============================
    U = np.zeros([1,expparams.size])
    for i in range(expparams.size):
        U[0][i] = updater.bayes_risk(expparams[i])
    exp_best = np.array([expparams[np.argmax(U[0])]],dtype=m.expparams_dtype)
    #===============================実験を繰り返し行う=============================
    datum = m.simulate_experiment(D0_vec, exp_best,repeat=m.n_data)
    #===============================重みを更新する===============================
    updater.batch_update(datum[:,0,0].reshape(-1,), expparams)
updater.plot_posterior_marginal()
#==================================結果の描画===================================
print("推定したD0:%f" %(updater.est_mean()))
print("推定したD0の共分散%f" %(updater.est_covariance_mtx()))
plt.legend(['Prior', 'Posterior'])
#plt.xlim(2869,2871)
plt.show()

