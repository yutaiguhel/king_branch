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

    def max_weight_x(self,x,w):
        id=np.argmax(w)
        return x[id]

    def main(self,x,w):
        x_max=self.max_weight_x(x,w)
        self.params()#パラメーター群を用意
        """
        self.a1=xout[0]
        self.a2=xout[1]
        """
        self.D0=x_max[5]
        self.AN=x_max[6]
        self.Bz=x_max[8]
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

#GRAPEを使うハミルトニアンラーニングの部分のみクラス化する試み
class GRAPE_Learning(Grape_funcs):
    
    def __init__(self):
        Q_H.__init__(self)
        self.ex=100 #試行回数
        self.i=0 #現在の試行回数
        self.n={"a1":5,"b1":5,"a2":5,"b2":5,"w_theta":5,"D0":5,"AN":5,"QN":5,"Bz":5} #推定基底毎のパーティクルの数 a1,b1,a2,b2,w_theta,D0,An,Qn,Bz
        self.d=1000 #一度の推定に使用する実験データの数
        self.a=0.75 #パーティクルの再配分における移動強度
        self.resample_threshold=0.5 #パーティクルの再配分を行う判断をする閾値
        self.approx_ratio=0.98 #不要なパーティクルを削除する際の残す割合
        self.bayes_threshold=1 #推定を終えるベイズリスクの閾値
        self.flag1=False #パーティクルの数が変化したらTrue
        self.p_exp=0 #真値におけるms=0にいた確率
        self.dir_num=0 #作成したテキストファイルの番号
        self.Data_num=0 #読み取ったテキストファイルの番号
        self.Data=0 #実験データ 一個目:操作後の発光量、2つ目:ms=0の発光量
        #結果格納配列
        self.i_list=[]
        self.ptable=[]
        #パーティクル
        self.ParamH={"a1":0,"b1":0,"a2":0,"b2":0,"w_theta":0,"D0":1,"AN":0,"QN":0,"Bz":0} #変更するパーティクルのパラメータ
        self.RangeH={"a1":5,"b1":3,"a2":10,"b2":5,"w_theta":2*np.pi,"D0":10,"AN":0,"QN":0,"Bz":0} #変更する範囲
        #実験設計
        self.V1=1 #ワイヤ1の電圧[V]
        self.V2=1 #ワイヤ2の電圧[V]
        self.phi=180*pi/180 #ワイヤ間の位相差[rad]

    def Mean(self,w,x): #重み付き平均を計算する関数
        """
        w:重み
        x:パラメータ
        wで重みづけされたxの平均を返す
        """
        mu=w*x
        
        #パーティクル数方向に和を取る
        mu=np.sum(mu,axis=0)
        return mu
    
    def weighting_matrix(self):
        """
        ベイズリスクの重み行列を作成
        ParamHの要素が1ならば対応する重み行列の要素も1
        つまり、ベイズリスクを考慮する
        """
        self.Q=np.zeros([len(self.x0), len(self.x0)])
        
        #広げたパラメータのみ考慮する
        for i,p in enumerate(self.ParamH):
            if self.ParamH[p]==1:
                px=1
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
    
    def reapprox(self,w,x,mode): #不要となったパーティクルを削除する関数
        """
        m:残すパーティクルの数
        ws:昇順に並び替えた重み
        wsの(m+1)番目の要素よりも大きい重みのパーティクルは残す
        """
        n=len(w)
        ws=sorted(w)
        m=floor(n*(1.0-self.approx_ratio))
        if m<1:
            m=0
        j=0
        delist=[]
        
        #delistにm個たまるまで継続
        while j!=m:
            i=0
            for i in range(n):
                if w[i]==ws[j] and n!=0:
                    delist.append(i)
                    j=j+1
                if j==m:
                    break
        #重み、パーティクルから不要なパーティクルを削除
        w=np.delete(w,delist,0)
        x=np.delete(x,delist,0)
        
        w=w/sum(w)
        
        if mode=="par":
            if n != len(w):
                self.flag1=True
                print("reapprox_par")
            else:
                self.flag1=False
        elif mode=="exp":
            if n != len(w):
                self.flag2=True
                print("reapprox_exp")
            else:
                self.flag2=False
        return w,x
            
    
    def Particlemaker(self,x,n,Param,Range): #パーティクルを生成する関数
        #itertools.product与えられた変数軍([x1,x2,x3],[y1,y2])の総重複なし組み合わせを配列として出力
        """
        パーティクルを生成する関数
        最小値:x-Range/2
        最大値:x+Range/2
        """
        N=len(x)
        temp=[]
        
        #ParamHが1のパーティクルのみ全幅RangeHで広げる
        for i,p in enumerate(Param):
            if(Param[p]==1):
                temp.append(np.linspace(x[i]-Range[p]/2,x[i]+Range[p]/2,n[p]))
            else:
                temp.append([x[i]])
        return(np.array(list(itertools.product(*temp))))

    def p_array(self):
        p_array=np.zeros([self.n_particles()])
        
    def Expsim_GRAPE(self):
        """
        パーティクルxに実験シミュレーションを行う関数
        """
        #量子状態の初期化
        self.rho_init()
        
        #System Hamiltonian
        self.H_0(self.x)

        CPO = cpo.create_pulse_optimizer(2*pi*self.H_0,[2*pi*self.Hdrive1],self.Uinit,self.U_ideal,self.n_tslot,self.pulse_time,
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
        
        self.rho=self.Uarb_last*self.rho_init()*self.Uarb_last.dag()
        
        #ms=0で測定
        expect0=self.exp(self.rho) #ms=0で測定
        if expect0 > 1.0:
            print("Probability Error")
            print(expect0)
            expect0=1
        return expect0
    
    def Update_GRAPE(self):
        """
        パーティクルの重みを更新する関数
        """
        self.p_exp=self.Expsim_GRAPE(self.x)
        #結果を乱数にする場合
        num=binomial(self.d, self.p_exp) #実験をd回行いｍs=0であった回数
        
        #尤度を計算
        temp=binom.pmf(num,n=self.d,p=self.p_array)#各パーティクルでの実験でms=0にいた確率
        temp=temp.reshape([len(temp),1])

        #重みの更新
        self.w=temp*self.w 
        
        #重みの全要素が0だった場合の例外処理
        if np.sum(self.w)==0:
            self.w=np.ones([n_particles(),1])/n_particles()
        else:
            self.w=self.w/np.sum(self.w) #重みの規格化      