from qutip import *
from scipy import *
from numpy import *
import matplotlib.pyplot as plt
import os
import sys
#このファイルとQ_moduleを別々のフォルダにに入れる場合は、ここにmoduleのフォルダ名を入れる
sys.path.append('C:/Users/syzkd/Python_program') 
from Q_module_new import*
sys.path.append('C:/Users/syzkd/Python_program/GRAPE_qutip') 
from Q_module_grape_qutip import*
import time

#---------- Single or Cross ----------#
gf = Grape_funcs_for_single_wire()
#gf = Grape_funcs_for_cross_wire()

#---------- GRAPEファイル保存先 ----------#
gf.dir_name = "C:/Users/syzkd/Python_program/GRAPE_data"
gf.folder_name = "GRAPE_gate_qutip_test"

#----------このpython自体の保存場所と名前 ----------#
gf.python_dir_name = path.dirname(__file__)+"/"
gf.python_file_name =  path.basename(__file__)


#---------- NVに関するパラメータ ----------#
D0 = 2870 #電子スピンの分裂
AN = 2.165
Q = 0
C_known  = []
C_inhomo = []
Bz = 0
theta = (63.4)*pi/180.0 #ワイヤー角度

gf.H_params = [D0,Q,AN,C_known,Bz,theta]
gf.C_inhomo = C_inhomo       
print("C_inhomo = "+str(C_inhomo))             
#---------- GRAPEに関するパラメータ ----------#       
count_max = 50

gf.fid_err_targ = 5e-5
gf.fid_err_tol  = 1e-4
gf.phi1 = 0
gf.phi2 = pi-theta
gf.pulse_time = 1000/1000.0
gf.t_div = 5/1000.0
gf.frequency_grape = D0
gf.power_max = 1000/80/2.0
gf.power_min = -gf.power_max
rate_hem = 2
t_hem = gf.pulse_time*rate_hem
gf.n_hem = int(round(t_hem/gf.t_div))
gf.freq_cut = 10
gf.init_pulse_type = "RNDFOURIER"
gf.fid_grad_change = 0

#---------- 作成したいGATE ----------#
frequency_ideal = D0
Ω1_ideal = 1000/0.001/2.0#/sin(theta)*(cos(theta)) 
Ω2_ideal = 0#1000/1.0/2.0/sin(theta)
Φ1_ideal = 0
Φ2_ideal = 0
pulse_time_ideal = 0.001/1000.0
H0ideal = defH0(D0,Q,AN,C_known,0)
H_ideal = Hamilton(0,frequency_ideal,Ω1_ideal,Ω2_ideal,Φ1_ideal,Φ2_ideal,theta,H0ideal)
gf.U_ideal = (-2*pi*1j*H_ideal*pulse_time_ideal).expm()

#---------- 作成したいSTATE ----------#
gf.state_to_state = False   #defaultはGATE評価、これがTrueの時だけSTATE評価になる
gf.state_init = tensor(zero,(zero+plus1+minus1)/sqrt(3),Hc)  #初期状態と終状態はどちらもket_vectorで入力してください
gf.state_goal = tensor(H,(zero+plus1+minus1)/sqrt(3),Hc)


#=============================================== GRAPE!!! ===============================================#


gf.params_Redefine()
gf.Hamiltonian_calc()
si = tensor(S0,S0)
sg = tensor(S0,S0)

tlist = linspace(gf.t_div,gf.t_div*gf.n_tslot,gf.n_tslot)
test = 100

start = time.time()
for i in range(test):
    params = [gf.D0,gf.Q,gf.AN,gf.C_known,gf.Bz,gf.theta]
    gf.pulse_shape = 1000/100/2.0*ones(gf.n_tslot).reshape([-1,1])#*sin(2*pi*1*tlist)
    fid_err = gf.calc_fid_err(params)
    p = []
    
    for i in range(gf.n_tslot):
        U = gf.Uarb[i]
        p = p+[(sg*U*si*U.dag()).tr()]
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time/float(test)) + "[sec]")
plt.plot(tlist,p)
plt.show()

start = time.time()
for i in range(test):
    H0 = defH0(gf.D0,gf.Q,gf.AN,gf.C_known,gf.Bz)
    Hdetu = gf.frequency_grape*tensor(Szz,III)
    Hdrive1 = tensor(Hxdrive(1,0,gf.phi1,0,gf.theta),III)
    H = H0-Hdetu+1000/100/2.0*Hdrive1
    p1 = []
    for i in range(gf.n_tslot):
        t = gf.t_div*(i+1)
        U = (-2*pi*1j*H*t).expm()
        p1 = p1+[(sg*U*si*U.dag()).tr()]
elapsed_time1 = time.time() - start
print ("elapsed_time1:{0}".format(elapsed_time1/float(test)) + "[sec]")        
plt.plot(tlist,p1)
plt.show()   
print("rate = "+str(elapsed_time/elapsed_time1)) 




