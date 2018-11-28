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
import qutip.control.optimizer as optimizer
import qutip.control.pulseoptim as cpo  

import matplotlib.pyplot as plt

import os
import timeit
import scipy.optimize as spopt
import copy
import collections
# QuTiP
from qutip import Qobj
import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules
import qutip.control.optimresult as optimresult
import qutip.control.termcond as termcond
import qutip.control.errors as errors
import qutip.control.dynamics as dynamics
import qutip.control.pulsegen as pulsegen
import qutip.control.dump as qtrldump

#このファイルとQ_moduleを別々のフォルダにに入れる場合は、ここにmoduleのフォルダ名を入れる
sys.path.append('C:/Users/syzkd/Python_program') 
from Q_module_new import*


def InUtility_wrapper(self,*args):
    p = 1-self.fid_err_func_wrapper(*args)
    return 1-p#Utility0(1-fid_err)

def InUtility_grad_wrapper(self,*args):
    p = self.fid_err_func_wrapper(*args)
    dpdc = -self.fid_err_grad_wrapper(*args)
    return -dpdc#Utility0_grad(fid_err)*fid_err_grad
    
def HL_run_optimization(self, term_conds=None):
    """
    Optimise the control pulse amplitudes to minimise the fidelity error
    using the L-BFGS-B algorithm, which is the constrained
    (bounded amplitude values), limited memory, version of the
    Broyden–Fletcher–Goldfarb–Shanno algorithm.

    The optimisation end when one of the passed termination conditions
    has been met, e.g. target achieved, gradient minimum met
    (local minima), wall time / iteration count exceeded.

    Essentially this is wrapper to the:
    scipy.optimize.fmin_l_bfgs_b function
    This in turn is a warpper for well established implementation of
    the L-BFGS-B algorithm written in Fortran, which is therefore
    very fast. See SciPy documentation for credit and details on
    this function.

    If the parameter term_conds=None, then the termination_conditions
    attribute must already be set. It will be overwritten if the
    parameter is not None

    The result is returned in an OptimResult object, which includes
    the final fidelity, time evolution, reason for termination etc
    
    """
    self.init_optim(term_conds)
    term_conds = self.termination_conditions
    dyn = self.dynamics
    cfg = self.config
    self.optim_var_vals = self._get_optim_var_vals()
    self._build_method_options()

    st_time = timeit.default_timer()
    self.wall_time_optimize_start = st_time

    if self.stats is not None:
        self.stats.wall_time_optim_start = st_time
        self.stats.wall_time_optim_end = 0.0
        self.stats.num_iter = 1

    bounds = self._build_bounds_list()
    result = self._create_result()

    if self.approx_grad:
        fprime = None
    else:
        fprime = self.InUtility_grad_wrapper


    if 'accuracy_factor' in self.method_options:
        factr = self.method_options['accuracy_factor']
    elif 'ftol' in self.method_options:
        factr = self.method_options['ftol']
    elif hasattr(term_conds, 'accuracy_factor'):
        factr = term_conds.accuracy_factor
    else:
        factr = 1e7

    if 'max_metric_corr' in self.method_options:
        m = self.method_options['max_metric_corr']
    elif 'maxcor' in self.method_options:
        m = self.method_options['maxcor']
    elif hasattr(self, 'max_metric_corr'):
        m = self.max_metric_corr
    else:
        m = 10

    if self.log_level <= logging.INFO:
        msg = ("Optimising pulse(s) using {} with "
                    "'fmin_l_bfgs_b' method").format(self.alg)
        if self.approx_grad:
            msg += " (approx grad)"
        logger.info(msg)

    try:
        optim_var_vals, fid, res_dict = spopt.fmin_l_bfgs_b(
            self.InUtility_wrapper, self.optim_var_vals,
            fprime=fprime,
            approx_grad=self.approx_grad,
            callback=self.iter_step_callback_func,
            bounds=self.bounds, m=m, factr=factr,
            pgtol=term_conds.min_gradient_norm,
            disp=self.msg_level,
            maxfun=term_conds.max_fid_func_calls,
            maxiter=term_conds.max_iterations)

        amps = self._get_ctrl_amps(optim_var_vals)
        dyn.update_ctrl_amps(amps)
        warn = res_dict['warnflag']
        if warn == 0:
            result.grad_norm_min_reached = True
            result.termination_reason = "function converged"
        elif warn == 1:
            result.max_iter_exceeded = True
            result.termination_reason = ("Iteration or fidelity "
                                         "function call limit reached")
        elif warn == 2:
            result.termination_reason = res_dict['task']

        result.num_iter = res_dict['nit']
    except errors.OptimizationTerminate as except_term:
        self._interpret_term_exception(except_term, result)

    end_time = timeit.default_timer()
    self._add_common_result_attribs(result, st_time, end_time)

    return result

class Grape_funcs(object):
    def __init__(self):
        self.dir_name = 0
        self.folder_name = 0
        self.python_dir_name = 0
        self.python_file_name =  0
        self.pulse_time = 1
        self.t_div = 1
        self.H_params = []
        self.C_inhomo = []
        self.H_ideal = 0
        self.frequency_grape = 2870
        self.phi1 = 0
        self.phi2 = 0
        self.power_max = 0
        self.power_min = 0
        self.n_hem = 0
        self.freq_cut = 0
        self.init_pulse_type = "RNDFOURIER"
        self.pulse_shape = 0
        self.Amp1_j = 0
        self.Amp2_j = 0
        self.phi1_j = 0
        self.phi2_j = 0
        self.fid_err_targ = 0
        self.fid_err_tol  = 0
        self.fid_err_list = []
        self.state_to_state = False
        

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
    def params_Redefine(self):  
        self.n_tslot = int(np.round(self.pulse_time/self.t_div))
        self.D0 = self.H_params[0]
        self.Q  = self.H_params[1]
        self.AN = self.H_params[2]
        self.C_known = self.H_params[3]
        self.Bz = self.H_params[4]
        self.theta = self.H_params[5]
        self.f_list = linspace(0,1.0/self.t_div,self.n_tslot+self.n_hem*2)
        self.t_hem = self.n_hem*self.t_div

    def Hamiltonian_calc(self):     
        number_C_known = len(self.C_known)
        number_H       = len(self.C_inhomo)
        if number_C_known != 0:
            II_C_known = Qobj(qeye(2**number_C_known),dims=[list(2*ones(number_C_known)),list(2*ones(number_C_known))])
            if number_H != 0:
                HhfCz_inhomo = 0
                for i in range(number_H):
                    HhfCz_inhomo = HhfCz_inhomo-self.C_inhomo[i]*tensor(ket2dm(basis(number_H,i)),Sz,III,II_C_known,sigz)   
                II_inhomo = qeye(number_H)
                self.H0        = tensor(II_inhomo,defH0(self.D0,self.Q,self.AN,self.C_known,self.Bz),II)+HhfCz_inhomo
                self.Hdrive1   = tensor(II_inhomo,Hxdrive(1,0,self.phi1,0,self.theta),III,II_C_known,II)
                self.Hdrive2   = tensor(II_inhomo,Hxdrive(0,1,0,self.phi2,self.theta),III,II_C_known,II)
                self.Hdetuning = tensor(II_inhomo,Szz,III,II_C_known,II)
                self.Hdrift    = self.H0-(self.frequency_grape)*self.Hdetuning
                self.U0        = tensor(II_inhomo,III,III,II_C_known,II)
                self.Uinit     = tensor(II_inhomo,III,III,II_C_known,II)
                self.U_goal = tensor(II_inhomo,self.U_ideal,II)
                self.fid_norm = np.real((self.U_goal.dag()*self.U_goal).tr())            
            else:
                self.H0        = defH0(self.D0,self.Q,self.AN,self.C_known,self.Bz)
                self.Hdrive1   = tensor(Hxdrive(1,0,self.phi1,0,self.theta),III,II_C_known)
                self.Hdrive2   = tensor(Hxdrive(0,1,0,self.phi2,self.theta),III,II_C_known)
                self.Hdetuning = tensor(Szz,III,II_C_known)
                self.Hdrift    = self.H0-(self.frequency_grape)*self.Hdetuning
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
                self.H0        = tensor(II_inhomo,defH0(self.D0,self.Q,self.AN,self.C_known,self.Bz),II)+HhfCz_inhomo
                self.Hdrive1   = tensor(II_inhomo,Hxdrive(1,0,self.phi1,0,self.theta),III,II)
                self.Hdrive2   = tensor(II_inhomo,Hxdrive(0,1,0,self.phi2,self.theta),III,II)
                self.Hdetuning = tensor(II_inhomo,Szz,III,II)
                self.Hdrift    = self.H0-(self.frequency_grape)*self.Hdetuning
                self.U0        = tensor(II_inhomo,III,III,II)
                self.Uinit     = tensor(II_inhomo,III,III,II)
                self.U_goal = tensor(II_inhomo,self.U_ideal,II)
                self.fid_norm = np.real((self.U_goal.dag()*self.U_goal).tr())            
            else:
                self.H0        = defH0(self.D0,self.Q,self.AN,self.C_known,self.Bz)
                self.Hdrive1   = tensor(Hxdrive(1,0,self.phi1,0,self.theta),III)
                self.Hdrive2   = tensor(Hxdrive(0,1,0,self.phi2,self.theta),III)
                self.Hdetuning = tensor(Szz,III)
                self.Hdrift    = self.H0-(self.frequency_grape)*self.Hdetuning
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
        print("========== Before FFT ==========")
        plt.plot(pulse)
        plt.grid(b="on")
        plt.show()
        
        pulse_hem_fft = hstack((zeros(self.n_hem),pulse,zeros(self.n_hem)))
        t_list = linspace(self.t_div,self.pulse_time+self.t_hem*2,self.n_tslot+self.n_hem*2)
        FFT = np.fft.fft(pulse_hem_fft)/(self.n_tslot+self.n_hem*2)
        FFT_cut = hstack((FFT[0:int((self.n_tslot+self.n_hem*2)/(1/self.t_div/self.freq_cut))],zeros(len(FFT)-int((self.n_tslot+self.n_hem*2)/(1/self.t_div/self.freq_cut)))))*2
        pulse_hem = np.fft.fft(FFT_cut)[::-1].real
        pulse = pulse_hem[self.n_hem:len(pulse_hem)-self.n_hem]
        plt.plot(self.f_list,FFT)
        plt.plot(self.f_list,FFT_cut)
        plt.xlim(0,self.freq_cut*1.5)
        plt.grid(b="on")
        plt.show()
        pulse = np.clip(pulse,self.power_min,self.power_max)
        
        print("========== After FFT ==========")        
        plt.plot(pulse)
        plt.grid(b="on")
        plt.show()
        return pulse.reshape([self.n_tslot,1])
    """
        dyn.drift_dyn_gen = drift
        dyn.ctrl_dyn_gen = ctrls
        dyn.initial = initial
        dyn.target = target
    """    

    
    def run_optimization(self):       
        optimizer.Optimizer.InUtility_wrapper = InUtility_wrapper
        optimizer.Optimizer.InUtility_grad_wrapper = InUtility_grad_wrapper
        optimizer.OptimizerLBFGSB.HL_run_optimization = HL_run_optimization        
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
        return CPO.HL_run_optimization()

    def calc_fid_err(self,params):
        D0,Q,AN,C_known,Bz,theta = params
        number_C_known = len(C_known)
        if number_C_known != 0:
            II_C_known = Qobj(qeye(2**number_C_known),dims=[list(2*ones(number_C_known)),list(2*ones(number_C_known))])
            H0 = defH0(D0,Q,AN,C_known,Bz)
            Hdetuning = tensor(Szz,III,II_C_known)
            Hdrift = H0-self.frequency_grape*Hdetuning
            Uinit = tensor(III,III,II_C_known)
            Hdrive1 = tensor(Hxdrive(1,0,self.phi1,0,self.theta),III,II_C_known)
            Hdrive2 = tensor(Hxdrive(0,1,0,self.phi2,self.theta),III,II_C_known)

        else:
            H0 = defH0(D0,Q,AN,C_known,Bz)
            Hdetuning = tensor(Szz,III)
            Hdrift = H0-self.frequency_grape*Hdetuning
            Uinit = tensor(III,III)
            Hdrive1 = tensor(Hxdrive(1,0,self.phi1,0,self.theta),III)
            Hdrive2 = tensor(Hxdrive(0,1,0,self.phi2,self.theta),III)
            
        CPO = cpo.create_pulse_optimizer(2*pi*Hdrift,[2*pi*Hdrive1,2*pi*Hdrive2],Uinit,self.U_ideal,self.n_tslot,self.pulse_time,
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
        #result = CPO._create_result()
        #CPO._add_common_result_attribs(result,0,1)
        fid_err = CPO.dynamics.fid_computer.get_fid_err()  
        self.Uarb = CPO.dynamics.fwd_evo
        self.Uarb_last = CPO.dynamics.full_evo
        return CPO.dynamics.fid_computer.get_fid_err()  
    
    def data_choice(self):
        self.Amp1_j = self.pulse_shape.reshape(self.n_tslot)
        self.Amp2_j = zeros(self.n_tslot)
        self.phi1_j = self.phi1*np.ones(self.n_tslot)
        self.phi2_j = zeros(self.n_tslot)           
        
        
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
        print("========== Before FFT ==========")
        plt.plot(pulse[0])
        plt.plot(pulse[1])
        plt.grid(b="on")
        plt.show()
        
        for i in range(2):
            pulse_hem_fft = hstack((zeros(self.n_hem),pulse[i],zeros(self.n_hem)))
            t_list = linspace(self.t_div,self.pulse_time+self.t_hem*2,self.n_tslot+self.n_hem*2)
            FFT = np.fft.fft(pulse_hem_fft)/(self.n_tslot+self.n_hem*2)
            FFT_cut = hstack((FFT[0:int((self.n_tslot+self.n_hem*2)/(1/self.t_div/self.freq_cut))],zeros(len(FFT)-int((self.n_tslot+self.n_hem*2)/(1/self.t_div/self.freq_cut)))))*2
            pulse_hem = np.fft.fft(FFT_cut)[::-1].real
            pulse[i] = pulse_hem[self.n_hem:len(pulse_hem)-self.n_hem]
            plt.plot(self.f_list,FFT)
            plt.plot(self.f_list,FFT_cut)
            plt.xlim(0,self.freq_cut*1.5)
            plt.grid(b="on")
            plt.show()
        pulse = np.clip(pulse,self.power_min,self.power_max)
        
        print("========== After FFT ==========")        
        plt.plot(pulse[0])
        plt.plot(pulse[1])
        plt.grid(b="on")
        plt.show()
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

    def calc_fid_err(self,params):
        D0,Q,AN,C_known,Bz,theta = params
        number_C_known = len(C_known)
        if number_C_known != 0:
            II_C_known = Qobj(qeye(2**number_C_known),dims=[list(2*ones(number_C_known)),list(2*ones(number_C_known))])
            H0 = defH0(D0,Q,AN,C_known,Bz)
            Hdetuning = tensor(Szz,III,II_C_known)
            Hdrift = H0-self.frequency_grape*Hdetuning
            Uinit = tensor(III,III,II_C_known)
            Hdrive1 = tensor(Hxdrive(1,0,self.phi1,0,self.theta),III,II_C_known)
            Hdrive2 = tensor(Hxdrive(0,1,0,self.phi2,self.theta),III,II_C_known)

        else:
            H0 = defH0(D0,Q,AN,C_known,Bz)
            Hdetuning = tensor(Szz,III)
            Hdrift = H0-self.frequency_grape*Hdetuning
            Uinit = tensor(III,III)
            Hdrive1 = tensor(Hxdrive(1,0,self.phi1,0,self.theta),III)
            Hdrive2 = tensor(Hxdrive(0,1,0,self.phi2,self.theta),III)
            
        CPO = cpo.create_pulse_optimizer(2*pi*Hdrift,[2*pi*Hdrive1,2*pi*Hdrive2],Uinit,self.U_ideal,self.n_tslot,self.pulse_time,
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
        #result = CPO._create_result()
        #CPO._add_common_result_attribs(result,0,1)
        self.Uarb = CPO.dynamics.fwd_evo
        self.Uarb_last = CPO.dynamics.full_evo
        return CPO.dynamics.fid_computer.get_fid_err()        
    
    def data_choice(self):
        self.Amp1_j = self.pulse_shape.T[0]
        self.Amp2_j = self.pulse_shape.T[1]
        self.phi1_j = self.phi1*np.ones(self.n_tslot)
        self.phi2_j = self.phi2*np.ones(self.n_tslot)

    
    
    
    