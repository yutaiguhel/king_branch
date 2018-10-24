# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:59:21 2018
original program: GRAPEsingle_stateS_scipy.py

@author: yuta
"""

from qutip import *
from scipy import *
import scipy.linalg as sclin 
import qutip.control.pulsegen as pulsegen
import qutip.control.dynamics as dynamics
from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys 
import scipy.optimize._minimize as sp_min
#このファイルとQ_moduleを別々のフォルダにに入れる場合は、ここにmoduleのフォルダ名を入れる
sys.path.append("C:/Users/yuta/.ipython/profile_default/機械学習/サブpy")
from Q_module_new02 import*
import time
import path
start = time.time()

class Parameter:
    def __init__(self,D0,AN,Q,C_known,C_inhomo,Bz,theta,phi1,phi2,
                 state_init,state_goal,weighting_value,permax,pulse_time,t_div,target_list):
        self.D0=D0
        self.AN=AN
        self.Q=Q
        self.C_known=C_known
        self.C_inhomo=C_inhomo
        self.Bz=Bz
        self.theta=theta
        self.phi1=phi1
        self.phi2=phi2
        self.state_init=state_init
        self.state_goal=state_goal
        self.weighting_value=weighting_value
        self.permax=permax
        self.pulse_time=pulse_time
        self.t_div=t_div
        self.target_list=target_list
    
    def status(self):
        return self.D0,self.AN,self.Q,self.C_known,self.C_inhomo,self.Bz,self.theta,self.phi1,self.phi2,self.D0,self.AN,self.Q,self.C_known,self.C_inhomo,self.Bz,self.theta,self.phi1,self.phi2
        
    def physical_status(self):
        physical_param=[self.D0,self.AN,self.Q,self.C_known,self.C_inhomo,
                        self.Bz,self.theta,self.phi1,self.phi2]
        return physical_param
    
    def GRAPE_status(self):
        GRAPE_param=[self.state_init,self.state_goal,self.weighting_value,
                     self.permax,self.pulse_time,self.t_div,self.target_list]
        return GRAPE_param

class GRAPE(Parameter):
    #----------このpython自体の保存場所と名前 ----------#
    #python_dir_name = path.dirname(__file__)+"/"
    #python_file_name =  path.basename(__file__)
    def optimize(self):
        frequency_grape = self.D0+self.AN#-C_known[0]
        #---------- 作成したいSTATE ----------#
        
        n_goal = len(self.target_list)
        n_tslot = int(np.round(self.pulse_time/self.t_div))
        
        permin = -self.permax
        rate_hem = 2 #触らない
        freq_cut = 10 #触らない
        class_pulse = pulsegen.create_pulse_gen(pulse_type='RNDFOURIER', dyn=None, pulse_params=None)
        class_pulse.num_tslots = n_tslot
        #class_pulse.num_ctrl = 1
        init_pulse_prescale = class_pulse.gen_pulse()
        init_pulse = init_pulse_prescale*self.permax/max(init_pulse_prescale)
        #init_pulse = 1/pulse_time/2.0*(sin(2*pi*(AN)*linspace(t_div,pulse_time,n_tslot)))
        
        t_hem = self.pulse_time*rate_hem
        n_hem = int(round(t_hem/self.t_div))
        f_list = linspace(0,1.0/self.t_div,n_tslot+n_hem*2)
        Φ1_j = np.zeros(n_tslot)
        Φ2_j = np.zeros(n_tslot)
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
        def callback(self,pulse,f_list,t_hem,n_hem,n_tslot,freq_cut,permin):
            #print("========== Before FFT ==========") 描画する時はコメントアウトしない
            """
            plt.plot(pulse)
            plt.grid(b="on")
            plt.show()
            """
            pulse_hem_fft = hstack((zeros(n_hem),pulse,zeros(n_hem)))
            t_list = linspace(self.t_div,self.pulse_time+t_hem*2,n_tslot+n_hem*2)
            FFT = np.fft.fft(pulse_hem_fft)/(n_tslot+n_hem*2)
            FFT_cut = hstack((FFT[0:int((n_tslot+n_hem*2)/(1/self.t_div/freq_cut))],zeros(len(FFT)-int((n_tslot+n_hem*2)/(1/self.t_div/freq_cut)))))*2
            pulse_hem = np.fft.fft(FFT_cut)[::-1].real
            pulse = pulse_hem[n_hem:len(pulse_hem)-n_hem]
            """描画したいときはコメントアウトしない
            plt.plot(f_list,FFT)
            plt.plot(f_list,FFT_cut)
            plt.xlim(0,freq_cut)
            plt.grid(b="on")
            plt.show()
            pulse = np.clip(pulse,permin,self.permax)
            
            print("========== After FFT ==========")        
            plt.plot(pulse)
            plt.grid(b="on")
            plt.show()
            """
            return pulse
            
        #=============================================== GRAPE!!! ===============================================#
        
        number_C_known = len(self.C_known)
        number_H       = len(self.C_inhomo)
        II_C_known = Qobj(qeye(2**number_C_known),dims=[list(2*ones(number_C_known)),list(2*ones(number_C_known))])
        
        HhfCz_inhomo = 0
        if number_C_known == 0:
            for i in range(number_H):
                HhfCz_inhomo = HhfCz_inhomo-self.C_inhomo[i]*tensor(ket2dm(basis(number_H,i)),Sz,III,sigz)
        else:
            for i in range(number_H):
                HhfCz_inhomo = HhfCz_inhomo-self.C_inhomo[i]*tensor(ket2dm(basis(number_H,i)),Sz,III,II_C_known,sigz)
        
        if number_H == 0:
            II_inhomo = Qobj([[1]])
        else:
            II_inhomo = qeye(number_H)
        
        H0        = defH0(self.D0,self.Q,self.AN,self.C_known,self.Bz)
        Hdrift    = matrix((tensor(II_inhomo,H0,II)+HhfCz_inhomo).full())
        Hdrive1   = matrix(tensor(II_inhomo,Hxdrive(1,0,0,0,self.theta),III,II_C_known,II).full())
        Hdrive2   = matrix(tensor(II_inhomo,Hxdrive(0,1,0,0,self.theta),III,II_C_known,II).full())
        Hdetuning = matrix(tensor(II_inhomo,Szz,III,II_C_known,II).full())
        U0        = matrix(tensor(II_inhomo,III,III,II_C_known,II).full())
        
        
        for i in range(n_goal):
            if number_H == 0:
                self.state_init[i] = matrix(tensor(II_inhomo,self.state_init[i],II/2.0).full())
                self.state_goal[i] = matrix(tensor(II_inhomo,self.state_goal[i],II/2.0).full())
            else:
                self.state_init[i] = matrix(tensor(II_inhomo/number_H,self.state_init[i],II/2.0).full())
                self.state_goal[i] = matrix(tensor(II_inhomo/number_H,self.state_goal[i],II/2.0).full())
        """
        weighting_value_targ = []
        for i in range(n_goal):
            weighting_value_targ = weighting_value_targ+[weighting_value[target_list[i]]]
        """   
        
        bounds = [(permin,self.permax)]*n_tslot
        pulse = init_pulse
        
        state_init=self.state_init
        t_div=self.t_div
        state_goal=self.state_goal
        weighting_value=self.weighting_value
        def grape_fid_err(pulse):
            U_b = U0
            fid_sum = []
            for i in range(n_goal):
                for j in range(n_tslot):
                    U_b = matrix(sclin.expm(-2*pi*1j*(Hdrift-(frequency_grape)*Hdetuning+pulse[j]*Hdrive1)*self.t_div))*U_b
                state_final = U_b*state_init[i]*conj(U_b.T)
                state_final_qobj = Qobj(state_final  )
                state_goal_qobj  = Qobj(state_goal[i]) 
                fid_sum = fid_sum+[fidelity(state_final_qobj,state_goal_qobj)*weighting_value[i]]
            fid_ave = sum(fid_sum)/sum(weighting_value)   
            #print("fid = "+str(fid_ave)) 描画する時はコメントアウトしない
            return 1-fid_ave
        
        def grape_fid_err_grad(pulse):
            U_rho_j   = []
            U_lamda_j = []
            fid_err_grad_ave = []
            U_rho   = U0
            U_lamda = U0
            for j in range(n_tslot):
                U_rho   = matrix(sclin.expm(-2*pi*1j*(Hdrift-(frequency_grape)*Hdetuning+pulse[j]*Hdrive1)*t_div))*U_rho
                U_lamda = conj(matrix(sclin.expm(-2*pi*1j*(Hdrift-(frequency_grape)*Hdetuning+pulse[n_tslot-j-1]*Hdrive1)*t_div)).T)*U_lamda
                U_rho_j   = U_rho_j  +[U_rho  ]           
                U_lamda_j = U_lamda_j+[U_lamda]
                
            for j in range(n_tslot):
                fid_err_grad_j_sum = []
                for k in range(n_goal):
                    lamda = (U_lamda_j[n_tslot-j-2])*state_goal[k]*conj(U_lamda_j[n_tslot-j-2].T)
                    state_j = U_rho_j[j]*state_init[k]*conj(U_rho_j[j].T)
                    fid_err_grad_j = np.real(1j*2*pi*t_div*trace(lamda*(Hdrive1*state_j-state_j*Hdrive1)))
                    fid_err_grad_j_sum = fid_err_grad_j_sum+[fid_err_grad_j*weighting_value[k]]
                fid_err_grad_ave = fid_err_grad_ave+[sum(fid_err_grad_j_sum)/sum(weighting_value)]
         
            return array(fid_err_grad_ave)
        
        for i in range(2):                 
            pulse = callback(self,pulse,f_list,t_hem,n_hem,n_tslot,freq_cut,permin)
            result = sp_min.minimize(grape_fid_err,pulse,method="L-BFGS-B",jac=grape_fid_err_grad,bounds=bounds,\
                                options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-9, 'maxfun': 15000, 'maxiter': 1500, 'iprint': -1, 'maxls': 20})#,tol=1e-2)
            pulse = result.x
        Ω1_j = pulse #ラビ周波数の配列
        Ω2_j = np.zeros(n_tslot)
        
        return Φ1_j, Ω1_j
        """
        #--------　保存先の作成　--------#
        dir_num = 0
        while os.path.isdir(dir_name+"/"+folder_name+"_"+str(dir_num))==True:
            dir_num=dir_num+1
        
            #---------- GRAPEファイル保存先 ----------#
        dir_name = "C:/koga/python/GRAPE/GRAPE_result"
        folder_name = "stateS_qutip_test"
        os.mkdir(dir_name+"/"+folder_name+"_"+str(dir_num))
        os.chdir(dir_name+"/"+folder_name+"_"+str(dir_num))
        print("\n\n#=============== GRAPEデータ保存先 ===============#\n"+dir_name+"/"+folder_name+"_"+str(dir_num)+"\n\n")    
        #---------- このpython_program自体のコピー ----------#
        #copy_storage(dir_name+"/"+folder_name+"_"+str(dir_num)+"/",python_dir_name,[python_file_name])
        
        os.mkdir("%.1f"%(self.pulse_time*1000)+"ns")
        os.chdir("%.1f"%(self.pulse_time*1000)+"ns")
        
        np.savetxt("amplitude_wire1.txt",Ω1_j,newline="\r\n")
        np.savetxt("amplitude_wire2.txt",Ω2_j,newline="\r\n")
        np.savetxt("phase_wire1.txt",Φ1_j,newline="\r\n")
        np.savetxt("phase_wire2.txt",Φ2_j,newline="\r\n")    
        np.savetxt("MW_time_Data.txt",np.array([n_tslot,t_div]),newline="\r\n")
        np.savetxt("MW_modu_Data_wire1.txt",np.c_[Ω1_j,(frequency_grape*ones(n_tslot))*2*pi,Φ1_j],newline="\r\n")
        np.savetxt("MW_modu_Data_wire2.txt",np.c_[Ω2_j,(frequency_grape*ones(n_tslot))*2*pi,Φ2_j],newline="\r\n")
                
        plt.xlabel("Time [μs]")
        plt.ylabel("Ω [Mhz]")
        plt.plot(linspace(self.pulse_time/n_tslot,self.pulse_time,int(n_tslot)),pulse)
        plt.grid(b="on")
        plt.savefig("Ω.png")
        plt.show()
        
        os.chdir("../") 
        print(" ↑"+dir_name+"/"+folder_name+"_"+str(dir_num)+"/"+"%.1f"%(self.pulse_time*1000)+"ns↑\n#==================================================================#")
        os.chdir("../") 
        
        
        
        elapsed_time = time.time() - start
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        return Φ1_j, Ω1_j
        """
        
        
        
        
