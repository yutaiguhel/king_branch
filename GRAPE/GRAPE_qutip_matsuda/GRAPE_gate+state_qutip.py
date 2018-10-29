from qutip import *
from scipy import *
from numpy import *
import matplotlib.pyplot as plt
import os
import sys
#このファイルとQ_moduleを別々のフォルダにに入れる場合は、ここにmoduleのフォルダ名を入れる
sys.path.append("C:/Users/yuta/.ipython/profile_default/GRAPE/GRAPE_qutip/__pycache__")
from Q_module_new import*
from Q_module_grape_qutip import*
import time
start = time.time()

#---------- Single or Cross ----------#
gf = Grape_funcs_for_single_wire()
#gf = Grape_funcs_for_cross_wire()

#---------- GRAPEファイル保存先 ----------#
gf.dir_name = "C:/Users/yuta/.ipython/profile_default/GRAPE/GRAPE_qutip/__pycache__"
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
gf.phi2 = 0
gf.pulse_time = 1000/1000.0
gf.t_div = 5/1000.0
gf.frequency_grape = D0
gf.power_max = 1000/50/2.0
gf.power_min = -gf.power_max
rate_hem = 2
t_hem = gf.pulse_time*rate_hem
gf.n_hem = int(round(t_hem/gf.t_div))
gf.freq_cut = 10
gf.init_pulse_type = "RNDFOURIER"
gf.fid_grad_change = 0

#---------- 作成したいGATE ----------#
frequency_ideal = D0
Ω1_ideal = 1000/1.0/2.0#/sin(theta)*(cos(theta)) 
Ω2_ideal = 0#1000/1.0/2.0/sin(theta)
Φ1_ideal = 0
Φ2_ideal = 0
pulse_time_ideal = 1.0/1000.0
H0ideal = defH0(D0,Q,AN,C_known,0)
H_ideal = Hamilton(0,frequency_ideal,Ω1_ideal,Ω2_ideal,Φ1_ideal,Φ2_ideal,theta,H0ideal)
gf.U_ideal = (-2*pi*1j*H_ideal*pulse_time_ideal).expm()

#---------- 作成したいSTATE ----------#
gf.state_to_state = True
gf.state_init = tensor(zero,(zero+plus1+minus1)/sqrt(3))
gf.state_goal = tensor(H,(zero+plus1+minus1)/sqrt(3))


#=============================================== GRAPE!!! ===============================================#


gf.params_Redefine()
gf.Hamiltonian_calc()

fid_err = 1
count = 0
while fid_err > gf.fid_err_tol:
    gf.pulse_shape = gf.make_init_pulse()
    print("~~~~~~~~~~ try : "+str(count)+" ~~~~~~~~~~")  
    result = gf.run_optimization()
    fid_err = result.fid_err
    print("iter = "+str(result.num_iter))
    print("fid_err = "+str(result.fid_err))
    count = count+1 
    if count>=count_max:
        print("== ERROR: Cannot find optimezed pulse ==")
        sys.exit()

gf.pulse_shape = result.final_amps        
gf.fid_err_list = [result.fid_err]             
gf.storage_open() 
gf.storage_data(0)      
for i in range(2):                 
    gf.pulse_shape = gf.FREQ_cut(gf.pulse_shape)
    result = gf.run_optimization()
    gf.pulse_shape = result.final_amps
    print("iter = "+str(result.num_iter))
    print("fid_err = "+str(result.fid_err))
    gf.fid_err_list = gf.fid_err_list+[result.fid_err]             
    gf.storage_data(i+1)

gf.storage_fid_err()        
gf.storage_close()    

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")


