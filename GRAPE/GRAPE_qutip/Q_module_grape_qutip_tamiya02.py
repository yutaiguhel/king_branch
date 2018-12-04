from qutip import*
from numpy import *
import numpy as np
from scipy import*
import datetime
import os
import shutil
import sys
set_printoptions(threshold=inf)
import scipy.linalg as sclin 
import scipy.optimize._minimize as sp_min
import scipy.optimize as sp_opt

import qutip.control.pulsegen as pulsegen
import qutip.control.dynamics as dynamics
import qutip.control.pulseoptim as cpo  # noqa

import matplotlib.pyplot as plt
#このファイルとQ_moduleを別々のフォルダにに入れる場合は、ここにmoduleのフォルダ名を入れる
sys.path.append(r"C:\data&prog\koga\Github\king_branch\機械学習\サブpy")
from Q_module_new02 import*
from Q_H07 import*

class Grape_funcs(Q_H):
    def __init__(self):
        self.count_max = 50
        self.dir_name = 0
        self.folder_name = 0
        self.python_dir_name = 0
        self.python_file_name =  0
        #GRAPEの制約条件
        self.pulse_time = 1000.0/1000.0
        self.t_div = 5.0/1000
        self.divition_number=self.pulse_time/self.t_div
        self.C_inhomo = []
        self.H_ideal = 0
        self.frequency_grape = 2870
        self.phi1 = 0
        self.phi2 = 0
        self.power_max = 1000/50/2.0
        self.fid_err_targ = 0.005 #目的とするinfidelity
        self.fid_err_tol  = 0.01 #許容infidelity
        #self.power_min = -(self.power_max)
        #高周波カットに関するパラメータ
        self.rate_hem = 2
        #self.t_hem = self.pulse_time*self.rate_hem
        #self.n_hem = int(round(self.t_hem/self.t_div))
        self.freq_cut = 10
        self.init_pulse_type = "RNDFOURIER"
        #GRAPEの初期パラメータ
        self.fid_err=1
        self.pulse_shape = 0 #ラビ周波数配列が格納される
        self.Amp1_j = 0
        self.Amp2_j = 0
        self.phi1_j = 0
        self.phi2_j = 0
        self.fid_err_list = []
        #self.state_to_state = True
        self.state_to_state=True
        #stateGRAPEに必要なパラメータ
        self.state_init = tensor(zero,(zero+plus1+minus1)/sqrt(3),Hc,Hc)
        self.state_goal = tensor(H,(zero+plus1+minus1)/sqrt(3),Hc,Hc)
        #ProcessGRAPEに必要なパラメータ
        self.frequency_ideal = self.D0
        self.Ω1_ideal = 1000/1.0/2.0#/sin(theta)*(cos(theta)) 
        self.Ω2_ideal = 0#1000/1.0/2.0/sin(theta)
        self.Φ1_ideal = 0
        self.Φ2_ideal = 0
        self.pulse_time_ideal = 1.0/1000.0
        #self.H0ideal = defH0(D0,Q,AN,self.Ac_list,0)
        #self.H_ideal = Hamilton(0,self.frequency_ideal,self.Ω1_ideal,self.Ω2_ideal,self.Φ1_ideal,self.Φ2_ideal,theta,self.H0ideal)
        #self.U_ideal = (-2*pi*1j*self.H_ideal*self.pulse_time_ideal).expm()
        self.num_dir=0
        self.wire1_amp=[]
        self.wire2_amp=[]
        self.data=[]
        

    """
    Create and return a pulse generator object matching the given type.
    The pulse generators each produce a different type of pulse,
    see the gen_pulse function description for details.
    These are the random pulse options:
        
        RND - Independent random value in each timeslot
        RNDFOURIER - Fourier series with random coefficients
        RNDWAVES - Summation of random waves
        RNDWALK1 - Random change in amplitude each timeslot
        RNDWALK2 - Random change in amp gradient each timeslot
    
    These are the other non-periodic options:
        
        LIN - Linear, i.e. contant gradient over the time
        ZERO - special case of the LIN pulse, where the gradient is 0
    
    These are the periodic options
        
        SINE - Sine wave
        SQUARE - Square wave
        SAW - Saw tooth wave
        TRIANGLE - Triangular wave
    
    If a Dynamics object is passed in then this is used in instantiate
    the PulseGen, meaning that some timeslot and amplitude properties
    are copied over.
    
    """  
    def params(self):
        #GRAPEのパラメータ
        self.theta=(63.4)*pi/180.0 #ワイヤー角度
        self.power_min = -(self.power_max)
        self.t_hem = self.pulse_time*self.rate_hem
        self.n_hem = int(round(self.t_hem/self.t_div))
        self.n_tslot = int(np.round(self.pulse_time/self.t_div))
        self.f_list = linspace(0,1.0/self.t_div,self.n_tslot+self.n_hem*2)
        #NVのパラメータ
        self.QN_GRAPE=0
        self.Bz=self.Be+self.Bo
        self.H_params = [self.D0,self.QN_GRAPE,self.AN,self.Ac_list,self.Bz,self.theta]
        self.H0ideal = defH0(D0,self.QN_GRAPE,self.AN,self.Ac_list,0)
        self.H_ideal = Hamilton(0,self.frequency_ideal,self.Ω1_ideal,self.Ω2_ideal,self.Φ1_ideal,self.Φ2_ideal,theta,self.H0ideal)
        self.U_ideal = (-2*pi*1j*self.H_ideal*self.pulse_time_ideal).expm()

    def Hamiltonian_calc(self):     
        number_C_known = len(self.Ac_list)
        number_H       = len(self.C_inhomo)
        if number_C_known != 0:
            II_C_known = Qobj(qeye(2**number_C_known),dims=[list(2*ones(number_C_known)),list(2*ones(number_C_known))])
            if number_H != 0:
                HhfCz_inhomo = 0
                for i in range(number_H):
                    HhfCz_inhomo = HhfCz_inhomo-self.C_inhomo[i]*tensor(ket2dm(basis(number_H,i)),Sz,III,sigz)   
                II_inhomo = qeye(number_H)
                self.H_NV      = tensor(II_inhomo,defH0(self.D0,self.QN_GRAPE,self.AN,self.Ac_list,self.Bz),II)+HhfCz_inhomo
                self.Hdrive1   = tensor(II_inhomo,Hxdrive(1,0,self.phi1,0,self.theta),III,II_C_known,II)
                self.Hdrive2   = tensor(II_inhomo,Hxdrive(0,1,0,self.phi2,self.theta),III,II_C_known,II)
                self.Hdetuning = tensor(II_inhomo,Szz,III,II_C_known,II)
                self.Hdrift    = self.H_NV-(self.frequency_grape)*self.Hdetuning
                self.U0        = tensor(II_inhomo,III,III,II_C_known,II)
                self.Uinit     = tensor(II_inhomo,III,III,II_C_known,II)
                self.U_goal = tensor(II_inhomo,self.U_ideal,II)
                self.fid_norm = np.real((self.U_goal.dag()*self.U_goal).tr())            
            else:
                self.H_NV      = defH0(self.D0,self.QN_GRAPE,self.AN,self.Ac_list,self.Bz)
                self.Hdrive1   = tensor(Hxdrive(1,0,self.phi1,0,self.theta),III,II_C_known)
                self.Hdrive2   = tensor(Hxdrive(0,1,0,self.phi2,self.theta),III,II_C_known)
                self.Hdetuning = tensor(Szz,III,II_C_known)
                self.Hdrift    = self.H_NV-(self.frequency_grape)*self.Hdetuning
                self.U0        = tensor(III,III,II_C_known)
                self.Uinit     = tensor(III,III,II_C_known)
                self.U_goal = tensor(self.U_ideal)
                self.fid_norm = np.real((self.U_goal.dag()*self.U_goal).tr())
        else:
            if number_H != 0:
                HhfCz_inhomo = 0
                for i in range(number_H):
                    HhfCz_inhomo = HhfCz_inhomo-self.C_inhomo[i]*tensor(ket2dm(basis(number_H,i)),Sz,III,sigz)   
                II_inhomo = qeye(number_H)
                self.H_NV      = tensor(II_inhomo,defH0(self.D0,self.QN_GRAPE,self.AN,self.Ac_list,self.Bz),II)+HhfCz_inhomo
                self.Hdrive1   = tensor(II_inhomo,Hxdrive(1,0,self.phi1,0,self.theta),III,II)
                self.Hdrive2   = tensor(II_inhomo,Hxdrive(0,1,0,self.phi2,self.theta),III,II)
                self.Hdetuning = tensor(II_inhomo,Szz,III,II)
                self.Hdrift    = self.H_NV-(self.frequency_grape)*self.Hdetuning
                self.U0        = tensor(II_inhomo,III,III,II)
                self.Uinit     = tensor(II_inhomo,III,III,II)
                self.U_goal = tensor(II_inhomo,self.U_ideal,II)
                self.fid_norm = np.real((self.U_goal.dag()*self.U_goal).tr())            
            else:
                self.H_NV      = defH0(self.D0,self.QN_GRAPE,self.AN,self.Ac_list,self.Bz)
                self.Hdrive1   = tensor(Hxdrive(1,0,self.phi1,0,self.theta),III)
                self.Hdrive2   = tensor(Hxdrive(0,1,0,self.phi2,self.theta),III)
                self.Hdetuning = tensor(Szz,III)
                self.Hdrift    = self.H_NV-(self.frequency_grape)*self.Hdetuning
                self.U0        = tensor(III,III)
                self.Uinit     = tensor(III,III)
                self.U_goal = tensor(self.U_ideal)
                self.fid_norm = np.real((self.U_goal.dag()*self.U_goal).tr())  
                
        if self.state_to_state == True:
            self.Uinit = self.state_init
            self.U_goal = self.state_goal                
        
    def storage_open(self):
        #--------　保存先の作成　--------#
        self.dir_num = 0
        while os.path.isdir(self.dir_name+"/"+self.folder_name+"_"+str(self.dir_num))==True:
            self.dir_num=self.dir_num+1
        
        os.mkdir(self.dir_name+"/"+self.folder_name+"_"+str(self.dir_num))
        os.chdir(self.dir_name+"/"+self.folder_name+"_"+str(self.dir_num))
        #---------- このpython_program自体のコピー ----------#
        copy_storage(self.dir_name+"/"+self.folder_name+"_"+str(self.dir_num)+"/",self.python_dir_name,[self.python_file_name])

    def storage_close(self):        
        os.chdir("../") 
    
    def storage_data(self,i):
        print("<< ↓"+self.dir_name+"/"+self.folder_name+"_"+str(self.dir_num)+"/"+"%.1f"%(self.pulse_time*1000)+"ns_"+str(i)+"↓ >>")    
        self.data_choice()
        os.mkdir("%.1f"%(self.pulse_time*1000)+"ns_"+str(i))
        os.chdir("%.1f"%(self.pulse_time*1000)+"ns_"+str(i))
        
        np.savetxt("amplitude_wire1.txt",self.Amp1_j,newline="\r\n")
        np.savetxt("amplitude_wire2.txt",self.Amp2_j,newline="\r\n")
        np.savetxt("phase_wire1.txt",self.phi1_j,newline="\r\n")
        np.savetxt("phase_wire2.txt",self.phi2_j,newline="\r\n")    
        np.savetxt("MW_time_Data.txt",np.array([self.n_tslot,self.t_div]),newline="\r\n")
        np.savetxt("MW_modu_Data_wire1.txt",np.c_[self.Amp1_j,(self.frequency_grape*ones(self.n_tslot))*2*pi,self.phi1_j],newline="\r\n")
        np.savetxt("MW_modu_Data_wire2.txt",np.c_[self.Amp2_j,(self.frequency_grape*ones(self.n_tslot))*2*pi,self.phi2_j],newline="\r\n")
         
        plt.xlabel("Time [μs]")
        plt.ylabel("Ω [Mhz]")
        plt.plot(linspace(self.pulse_time/float(self.n_tslot),self.pulse_time,self.n_tslot),self.Amp1_j)
        plt.plot(linspace(self.pulse_time/float(self.n_tslot),self.pulse_time,self.n_tslot),self.Amp2_j)
        plt.grid(b="on")
        plt.savefig("Ω.png")
        plt.show()
        os.chdir("../")
        print("<< ↑"+self.dir_name+"/"+self.folder_name+"_"+str(self.dir_num)+"/"+"%.1f"%(self.pulse_time*1000)+"ns_"+str(i)+"↑ >>")

    def storage_fid_err(self):
        np.savetxt("Infidelity.txt",self.fid_err_list,newline="\r\n")
        plt.xlabel("file")
        plt.ylabel("Infidelity")
        plt.plot(self.fid_err_list)
        plt.grid(b="on")
        plt.savefig("Infidelity.png")
        plt.show()

        
                              
class Grape_funcs_for_single_wire(Grape_funcs):   
        
    def make_init_pulse(self):
        class_pulse = pulsegen.create_pulse_gen(pulse_type=self.init_pulse_type, dyn=None, pulse_params=None)
        class_pulse.num_tslots = self.n_tslot
        init_pulse_prescale = class_pulse.gen_pulse()
        pulse = init_pulse_prescale*self.power_max/max(init_pulse_prescale)
        return pulse.reshape([self.n_tslot,1]) 
                          

    def FREQ_cut(self,pulse):  
        pulse = pulse.reshape(self.n_tslot)
        #print("========== Before FFT ==========")
        #plt.plot(pulse)
        #plt.grid(b="on")
        #plt.show()
        
        pulse_hem_fft = hstack((zeros(self.n_hem),pulse,zeros(self.n_hem)))
        t_list = linspace(self.t_div,self.pulse_time+self.t_hem*2,self.n_tslot+self.n_hem*2)
        FFT = np.fft.fft(pulse_hem_fft)/(self.n_tslot+self.n_hem*2)
        FFT_cut = hstack((FFT[0:int((self.n_tslot+self.n_hem*2)/(1/self.t_div/self.freq_cut))],zeros(len(FFT)-int((self.n_tslot+self.n_hem*2)/(1/self.t_div/self.freq_cut)))))*2
        pulse_hem = np.fft.fft(FFT_cut)[::-1].real
        pulse = pulse_hem[self.n_hem:len(pulse_hem)-self.n_hem]
    
        #plt.plot(self.f_list,FFT)
        #plt.plot(self.f_list,FFT_cut)
        #plt.xlim(0,self.freq_cut*1.5)
        #plt.grid(b="on")
        #plt.show()
        pulse = np.clip(pulse,self.power_min,self.power_max)
        
        #print("========== After FFT ==========")        
        #plt.plot(pulse)
        #plt.grid(b="on")
        #plt.show()

        return pulse.reshape([self.n_tslot,1])
    
    def run_optimization(self):       
        CPO = cpo.create_pulse_optimizer(2*pi*self.Hdrift,[2*pi*self.Hdrive1],self.Uinit,self.U_goal,self.n_tslot,self.pulse_time,
                                   amp_lbound=self.power_min, 
                                   amp_ubound=self.power_max,
                                   fid_err_targ=self.fid_err_targ,
                                   min_grad=1e-10,
                                   alg = "GRAPE",
                                   max_iter=500, max_wall_time=180,
                                   method_params=None,
                                   optim_method='fmin_l_bfgs_b',
                                   dyn_type='UNIT',
                                   dyn_params=None,
                                   prop_type='DEF',
                                   fid_type='DEF',
                                   fid_params={'phase_option': 'PSU'},
                                   init_pulse_type="RNDFOURIER", 
                                   pulse_scaling=1.0,
                                   pulse_offset = 0.0,
                                   ramping_pulse_type=None,
                                   ramping_pulse_params=None,
                                   log_level = 30,
                                   gen_stats=False)
        CPO.dynamics.initialize_controls(self.pulse_shape)
        return CPO.run_optimization()

    def calc_fid_err(self):
        CPO = cpo.create_pulse_optimizer(2*pi*self.Hdrift,[2*pi*self.Hdrive1],self.Uinit,self.U_ideal,self.n_tslot,self.pulse_time,
                                   dyn_type='UNIT',
                                   dyn_params=None,
                                   prop_type='DEF',
                                   fid_type='DEF',
                                   fid_params={'phase_option': 'PSU'},
                                   pulse_scaling=1.0,
                                   pulse_offset = 0.0,
                                   log_level = 30,
                                   gen_stats=False)
        CPO.dynamics.initialize_controls(self.pulse_shape)
        fid_err = CPO.dynamics.fid_computer.get_fid_err()  
        self.Uarb = CPO.dynamics.fwd_evo
        self.Uarb_last = CPO.dynamics.full_evo
        return fid_err
    
    def data_choice(self):
        self.Amp1_j = self.pulse_shape.reshape(self.n_tslot)
        self.Amp2_j = zeros(self.n_tslot)
        self.phi1_j = self.phi1*np.ones(self.n_tslot)
        self.phi2_j = zeros(self.n_tslot)   

    def main(self):
        self.params()#パラメーター群を用意
        self.Hamiltonian_calc()#ハミルトニアンを用意
        count=0
        while self.fid_err > self.fid_err_tol:
            self.pulse_shape = self.make_init_pulse()#初期化波形をpulse_shapeに格納(time vs rabi_frequency) 
            result = self.run_optimization()
            self.fid_err = result.fid_err
            count = count+1 
            if count>=self.count_max:
                print("== ERROR: Cannot find optimezed pulse ==")
                sys.exit()

        self.pulse_shape = result.final_amps        
        self.fid_err_list = [result.fid_err]                 
        for i in range(2):#高周波カットを二回行っている
            self.pulse_shape = self.FREQ_cut(self.pulse_shape)
            result = self.run_optimization()
            self.pulse_shape = result.final_amps
            print("iter = "+str(result.num_iter))
            print("fid_err = "+str(result.fid_err))
            self.fid_err_list = self.fid_err_list+[result.fid_err]       
            
    def inverse_func_rabi_frequency(self,a,b):
        for i in range(self.pulse_shape.size):
            if self.pulse_shape[i]<=a*b*b/2.0:
                self.wire_amp.append(b+sqrt(b*b-(2*b/a)*self.pulse_shape[i]))
            else:
               self.pulse_shape[i]>a*b*b/2.0
                self.wire_amp.append(b)
                
    def time(self):
        t_temp=np.full(int(self.divition_number),self.t_div)#t_divの値に初期化 #配列の要素数はint型で指定
        self.t=np.cumsum(t_temp)#累積和をとる
    
    def csv_output(self,pulse_shape,a,b):
        self.inverse_func_rabi_frequency(a,b)
        self.time()
        self.data=np.vstack((self.t,self.wire1_shape,self.wire_amp))

        cd="\\\\UNICORN\\data&prog\\機械学習用テストフォルダ\\GRAPE\\Setting"
        while os.path.isdir(cd+str(self.num_dir))==True:
            self.num_dir=self.num_dir+1
        print("ファイル#: %d" %(self.num_dir))
        if self.num_dir==0:
            os.mkdir(cd+str(self.num_dir))
        np.savetxt(cd+str(self.num_dir),self.data.T,newline="\n",delimiter=",")
        os.chdir("../")
        
    def Expsim_GRAPE(self,x,):
        """
        パーティクルxに実験シミュレーションを行う関数
        """
        #量子状態の初期化
        self.rho_init()
        
        #System Hamiltonian
        self.H_0(x)
        
        #回転座標系に乗るためのハミルトニアン
        self.H_rot(self.pulse_shape[i]) #C[4]:MW周波数
        
        #回転座標系に乗った時のドライブハミルトニアン
        self.Vdrive_all(x,wire_amp[i],C[1],C[2]) #C[0]:V1, C[1]:V2, C[2]:ワイヤ間の位相差phi
        
        #ドライブハミルトニアンで時間発展
        self.Tevo(C[3]) #C[3]:MWwidth
            
        #ms=0で測定
        expect0=self.exp(self.rho) #ms=0で測定
        if expect0 > 1.0:
            print("Probability Error")
            print(expect0)
            expect0=1
        return expect0
        
class Grape_funcs_for_cross_wire(Grape_funcs):   
     
    def make_init_pulse(self):
        init_pulse = zeros([2,self.n_tslot])
        class_pulse = pulsegen.create_pulse_gen(pulse_type=self.init_pulse_type, dyn=None, pulse_params=None)
        class_pulse.num_tslots = self.n_tslot
        init_pulse_prescale = class_pulse.gen_pulse()
        init_pulse[0] = init_pulse_prescale*self.power_max/max(init_pulse_prescale)
        init_pulse[1] = init_pulse_prescale*self.power_max/max(init_pulse_prescale)
        return init_pulse.T
                         
    def FREQ_cut(self,pulse):  
        pulse = pulse.T
        #print("========== Before FFT ==========")
        #plt.plot(pulse[0])
        #plt.plot(pulse[1])
        #plt.grid(b="on")
        #plt.show()
        
        for i in range(2):
            pulse_hem_fft = hstack((zeros(self.n_hem),pulse[i],zeros(self.n_hem)))
            t_list = linspace(self.t_div,self.pulse_time+self.t_hem*2,self.n_tslot+self.n_hem*2)
            FFT = np.fft.fft(pulse_hem_fft)/(self.n_tslot+self.n_hem*2)
            FFT_cut = hstack((FFT[0:int((self.n_tslot+self.n_hem*2)/(1/self.t_div/self.freq_cut))],zeros(len(FFT)-int((self.n_tslot+self.n_hem*2)/(1/self.t_div/self.freq_cut)))))*2
            pulse_hem = np.fft.fft(FFT_cut)[::-1].real
            pulse[i] = pulse_hem[self.n_hem:len(pulse_hem)-self.n_hem]
            #plt.plot(self.f_list,FFT)
            #plt.plot(self.f_list,FFT_cut)
            #plt.xlim(0,self.freq_cut*1.5)
            #plt.grid(b="on")
            #plt.show()
        pulse = np.clip(pulse,self.power_min,self.power_max)
        
        #print("========== After FFT ==========")        
        #plt.plot(pulse[0])
        #plt.plot(pulse[1])
        #plt.grid(b="on")
        #plt.show()
        return pulse.T
    
    def run_optimization(self):       
        CPO = cpo.create_pulse_optimizer(2*pi*self.Hdrift,[2*pi*self.Hdrive1,2*pi*self.Hdrive2],self.Uinit,self.U_goal,self.n_tslot,self.pulse_time,
                                   amp_lbound=self.power_min, 
                                   amp_ubound=self.power_max,
                                   fid_err_targ=self.fid_err_targ,
                                   min_grad=1e-10,
                                   alg = "GRAPE",
                                   max_iter=500, max_wall_time=180,
                                   method_params=None,
                                   optim_method='fmin_l_bfgs_b',
                                   dyn_type='UNIT',
                                   dyn_params=None,
                                   prop_type='DEF',
                                   fid_type='DEF',
                                   fid_params={'phase_option': 'PSU'},
                                   init_pulse_type="RNDFOURIER", 
                                   pulse_scaling=1.0,
                                   pulse_offset = 0.0,
                                   ramping_pulse_type=None,
                                   ramping_pulse_params=None,
                                   log_level = 30,
                                   gen_stats=False)

        CPO.dynamics.initialize_controls(self.pulse_shape)
        return CPO.run_optimization()

    def calc_fid_err(self):
        CPO = cpo.create_pulse_optimizer(2*pi*self.Hdrift,[2*pi*self.Hdrive1,2*pi*self.Hdrive2],self.Uinit,self.U_ideal,self.n_tslot,self.pulse_time,
                                   dyn_type='UNIT',
                                   dyn_params=None,
                                   prop_type='DEF',
                                   fid_type='DEF',
                                   fid_params={'phase_option': 'PSU'},
                                   pulse_scaling=1.0,
                                   pulse_offset = 0.0,
                                   log_level = 30,
                                   gen_stats=False)
        CPO.dynamics.initialize_controls(self.pulse_shape)
        fid_err = CPO.dynamics.fid_computer.get_fid_err()
        self.Uarb = CPO.dynamics.fwd_evo
        self.Uarb_last = CPO.dynamics.full_evo
        return fid_err       
    
    def data_choice(self):
        self.Amp1_j = self.pulse_shape.T[0]
        self.Amp2_j = self.pulse_shape.T[1]
        self.phi1_j = self.phi1*np.ones(self.n_tslot)
        self.phi2_j = self.phi2*np.ones(self.n_tslot)
        
    def main(self):
        self.params()#パラメーター群を用意
        self.Hamiltonian_calc()#ハミルトニアンを用意
        count=0
        while self.fid_err > self.fid_err_tol:
            self.pulse_shape = self.make_init_pulse()#初期化波形をpulse_shapeに格納(time vs rabi_frequency) 
            result = self.run_optimization()
            self.fid_err = result.fid_err
            count = count+1 
            if count>=self.count_max:
                print("== ERROR: Cannot find optimezed pulse ==")
                sys.exit()

        self.pulse_shape = result.final_amps        
        self.fid_err_list = [result.fid_err]                 
        for i in range(2):#高周波カットを二回行っている
            self.pulse_shape = self.FREQ_cut(self.pulse_shape)
            result = self.run_optimization()
            self.pulse_shape = result.final_amps
            print("iter = "+str(result.num_iter))
            print("fid_err = "+str(result.fid_err))
            self.fid_err_list = self.fid_err_list+[result.fid_err]
            
    def inverse_func_rabi_frequency(self,pulse_shape,a,b):
        self.wire1_shape=pulse_shape[:,0]#pulse_shapeの0番目の列を抽出
        self.wire2_shape=pulse_shape[:,1]#pulse_shapeの1番目の列を抽出

        for i in range(self.wire1_shape.size):
            if self.wire1_shape[i]<=a*b*b/2.0:
                self.wire1_amp.append(b+sqrt(b*b-(2*b/a)*self.wire1_shape[i]))
            else:
                self.wire1_shape[i]>a*b*b/2.0
                self.wire1_amp.append(b)

        for i in range(self.wire2_shape.size):
            if self.wire2_shape[i]<=a*b*b/2.0:
                self.wire2_amp.append(b+sqrt(b*b-(2*b/a)*self.wire1_shape[i]))
            else:
                self.wire2_shape[i]>a*b*b/2.0
                self.wire2_amp.append(b)
                
    def time(self):
        t_temp=np.full(int(self.divition_number),self.t_div)#t_divの値に初期化 #配列の要素数はint型で指定
        self.t=np.cumsum(t_temp)#累積和をとる
    
    def csv_output(self,pulse_shape,a,b):
        self.inverse_func_rabi_frequency(pulse_shape,a,b)
        self.time()
        self.data=np.vstack((self.t,self.wire1_shape,self.wire2_shape,self.wire1_amp,self.wire2_amp))

        cd="\\\\UNICORN\\data&prog\\機械学習用テストフォルダ\\GRAPE\\Setting"
        while os.path.isdir(cd+str(self.num_dir))==True:
            self.num_dir=self.num_dir+1
        print("ファイル#: %d" %(self.num_dir))
        if self.num_dir==0:
            os.mkdir(cd+str(self.num_dir))
        np.savetxt(cd+str(self.num_dir),self.data.T,newline="\n",delimiter=",")
        os.chdir("../")
            
            
        
