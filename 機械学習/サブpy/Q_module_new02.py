from qutip import*
from numpy import *
import numpy as np
from scipy import*
import datetime
import os
import shutil
import sys
set_printoptions(threshold=inf)
import matplotlib.pyplot as plt

#string
on = "on"
off = "off"
grape = "GRAPE"
GRAPE = "GRAPE"

#parameter
#physical parameter
D0=2870
AN=-2.2
Q=0
C_known  = [0.512/2.0] #既知の炭素
C_inhomo = []
Bz = 0
theta = (63.4)*pi/180.0 #angle between wire1 and wire2
phi1=0 #wire1 phase
phi2=0 #wire2 phase
#GRAPE parameter
pulse_time = 1000/1000.0 #pulse length
t_div = 5/1000.0 #⊿(pulse length) 
permax = 4#1000/150/2.0　ラビ周波数の上限

#carbon
II = qeye(2)
sigx = sigmax()
sigy = sigmay()
sigz = sigmaz()
sigma = [II,sigx,sigy,sigz]
up = basis(2,0)
down = basis(2,1)
Hc = (up+down)/sqrt(2)
Vc = (up-down)/sqrt(2)
Dc = (up+1j*down)/sqrt(2)
Ac = (up-1j*down)/sqrt(2)
Sup = ket2dm(up)
Sdown = ket2dm(down)
Sraise_c = jmat(1/2.0,"+")
Slower_c = jmat(1/2.0,"-")
SHc = ket2dm(Hc)
SVc = ket2dm(Vc)
SDc = ket2dm(Dc)
SAc = ket2dm(Ac)

#electoron/nuclear
zero = basis(3,1)
plus1 = basis(3,0)
minus1 = basis(3,2)
H = (plus1 + minus1)/sqrt(2)
V = (plus1 - minus1)/sqrt(2)
D = (plus1 + 1j*minus1)/sqrt(2)
A = (plus1 - 1j*minus1)/sqrt(2)
p0H = (plus1 + zero)/sqrt(2)
p0V = (plus1 - zero)/sqrt(2)
p0D = (plus1 + 1j*zero)/sqrt(2)
p0A = (plus1 - 1j*zero)/sqrt(2)
m0H = (minus1 + zero)/sqrt(2)
m0V = (minus1 - zero)/sqrt(2)
m0D = (minus1 + 1j*zero)/sqrt(2)
m0A = (minus1 - 1j*zero)/sqrt(2)
S0 = ket2dm(zero)
Sp = ket2dm(plus1)
Sm = ket2dm(minus1)
SH = ket2dm(H)
SV = ket2dm(V)
SD = ket2dm(D)
SA = ket2dm(A)
Sp0H = ket2dm(p0H)
Sp0V = ket2dm(p0V)
Sp0D = ket2dm(p0D)
Sp0A = ket2dm(p0A)
Sm0H = ket2dm(m0H)
Sm0V = ket2dm(m0V)
Sm0D = ket2dm(m0D)
Sm0A = ket2dm(m0A)
Sx = jmat(1,"x")
Sy = jmat(1,"y")
Sz = jmat(1,"z")
Sraise = jmat(1,"+")
Slower = jmat(1,"-")
III = qeye(3)
Sxx = Sx*Sx
Syy = Sy*Sy
Szz = Sz*Sz
Sxy = Sx*Sy
Syx = Sy*Sx
phi_p=(tensor(plus1,plus1) + tensor(minus1,minus1))/sqrt(2)
phi_m=(tensor(plus1,plus1) - tensor(minus1,minus1))/sqrt(2)
psi_p=(tensor(plus1,minus1) + tensor(minus1,plus1))/sqrt(2)
psi_m=(tensor(plus1,minus1) - tensor(minus1,plus1))/sqrt(2)
Phi_P=ket2dm(phi_p)
Phi_M=ket2dm(phi_m)
Psi_P=ket2dm(psi_p)
Psi_M=ket2dm(psi_m)
Cli_trans=array([[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
         11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,
         22.,  23.],
       [  1.,   0.,   3.,   2.,   6.,   7.,   4.,   5.,  11.,  10.,   9.,
          8.,  13.,  12.,  18.,  19.,  22.,  23.,  14.,  15.,  21.,  20.,
         16.,  17.],
       [  2.,   3.,   0.,   1.,   7.,   6.,   5.,   4.,  10.,  11.,   8.,
          9.,  20.,  21.,  15.,  14.,  23.,  22.,  19.,  18.,  12.,  13.,
         17.,  16.],
       [  3.,   2.,   1.,   0.,   5.,   4.,   7.,   6.,   9.,   8.,  11.,
         10.,  21.,  20.,  19.,  18.,  17.,  16.,  15.,  14.,  13.,  12.,
         23.,  22.],
       [  4.,   7.,   5.,   6.,  11.,   8.,   9.,  10.,   2.,   3.,   1.,
          0.,  22.,  17.,  21.,  12.,  14.,  18.,  13.,  20.,  23.,  16.,
         15.,  19.],
       [  5.,   6.,   4.,   7.,  10.,   9.,   8.,  11.,   1.,   0.,   2.,
          3.,  23.,  16.,  12.,  21.,  19.,  15.,  20.,  13.,  22.,  17.,
         18.,  14.],
       [  6.,   5.,   7.,   4.,   8.,  11.,  10.,   9.,   3.,   2.,   0.,
          1.,  16.,  23.,  20.,  13.,  18.,  14.,  12.,  21.,  17.,  22.,
         19.,  15.],
       [  7.,   4.,   6.,   5.,   9.,  10.,  11.,   8.,   0.,   1.,   3.,
          2.,  17.,  22.,  13.,  20.,  15.,  19.,  21.,  12.,  16.,  23.,
         14.,  18.],
       [  8.,   9.,  11.,  10.,   1.,   3.,   2.,   0.,   7.,   4.,   5.,
          6.,  19.,  14.,  22.,  16.,  20.,  12.,  23.,  17.,  15.,  18.,
         13.,  21.],
       [  9.,   8.,  10.,  11.,   2.,   0.,   1.,   3.,   6.,   5.,   4.,
          7.,  14.,  19.,  23.,  17.,  13.,  21.,  22.,  16.,  18.,  15.,
         20.,  12.],
       [ 10.,  11.,   9.,   8.,   3.,   1.,   0.,   2.,   4.,   7.,   6.,
          5.,  18.,  15.,  17.,  23.,  12.,  20.,  16.,  22.,  14.,  19.,
         21.,  13.],
       [ 11.,  10.,   8.,   9.,   0.,   2.,   3.,   1.,   5.,   6.,   7.,
          4.,  15.,  18.,  16.,  22.,  21.,  13.,  17.,  23.,  19.,  14.,
         12.,  20.],
       [ 12.,  13.,  21.,  20.,  18.,  19.,  14.,  15.,  22.,  17.,  23.,
         16.,   1.,   0.,   4.,   5.,   8.,  10.,   6.,   7.,   2.,   3.,
         11.,   9.],
       [ 13.,  12.,  20.,  21.,  14.,  15.,  18.,  19.,  16.,  23.,  17.,
         22.,   0.,   1.,   6.,   7.,  11.,   9.,   4.,   5.,   3.,   2.,
          8.,  10.],
       [ 14.,  19.,  15.,  18.,  22.,  16.,  23.,  17.,  20.,  21.,  12.,
         13.,   8.,   9.,   2.,   0.,   6.,   4.,   1.,   3.,  10.,  11.,
          7.,   5.],
       [ 15.,  18.,  14.,  19.,  17.,  23.,  16.,  22.,  12.,  13.,  20.,
         21.,  10.,  11.,   0.,   2.,   5.,   7.,   3.,   1.,   8.,   9.,
          4.,   6.],
       [ 16.,  23.,  22.,  17.,  12.,  21.,  20.,  13.,  19.,  14.,  15.,
         18.,   5.,   6.,   8.,  11.,   3.,   0.,  10.,   9.,   7.,   4.,
          1.,   2.],
       [ 17.,  22.,  23.,  16.,  21.,  12.,  13.,  20.,  14.,  19.,  18.,
         15.,   4.,   7.,   9.,  10.,   0.,   3.,  11.,   8.,   6.,   5.,
          2.,   1.],
       [ 18.,  15.,  19.,  14.,  16.,  22.,  17.,  23.,  21.,  20.,  13.,
         12.,  11.,  10.,   3.,   1.,   4.,   6.,   0.,   2.,   9.,   8.,
          5.,   7.],
       [ 19.,  14.,  18.,  15.,  23.,  17.,  22.,  16.,  13.,  12.,  21.,
         20.,   9.,   8.,   1.,   3.,   7.,   5.,   2.,   0.,  11.,  10.,
          6.,   4.],
       [ 20.,  21.,  13.,  12.,  19.,  18.,  15.,  14.,  17.,  22.,  16.,
         23.,   3.,   2.,   7.,   6.,  10.,   8.,   5.,   4.,   0.,   1.,
          9.,  11.],
       [ 21.,  20.,  12.,  13.,  15.,  14.,  19.,  18.,  23.,  16.,  22.,
         17.,   2.,   3.,   5.,   4.,   9.,  11.,   7.,   6.,   1.,   0.,
         10.,   8.],
       [ 22.,  17.,  16.,  23.,  13.,  20.,  21.,  12.,  15.,  18.,  19.,
         14.,   7.,   4.,  11.,   8.,   2.,   1.,   9.,  10.,   5.,   6.,
          0.,   3.],
       [ 23.,  16.,  17.,  22.,  20.,  13.,  12.,  21.,  18.,  15.,  14.,
         19.,   6.,   5.,  10.,   9.,   1.,   2.,   8.,  11.,   4.,   7.,
          3.,   0.]])
 
 
def Hxdrive(Ome1,Ome2,fai1,fai2,theta):
    Ug = (-1j*fai1*Sp-1j*fai1*Sm).expm()
    Ur = (-1j*(fai2-fai1)*Sp-1j*(fai2-fai1)*Sm).expm()
    Hxdrive0 = (1/2.0)*Ug*(Ome1*Sx+Ome2*Ur*(Sx*cos(theta)+Sy*sin(theta))*Ur.dag())*Ug.dag()
    return Hxdrive0

def Hxdrive_phase(Ome1_cos,Ome1_sin,Ome2_cos,Ome2_sin):
    """
    Ome1_cos = Ome1*cos(phi1)
    Ome1_sin = Ome1*sin(phi1)
    Ome2_cos = Ome2*cos(phi2+theta)
    Ome2_sin = Ome2*sin(phi2+theta)
    """
    Hdrive_cos = (Ome1_cos+Ome2_cos)/2.0*Sx
    Hdrive_sin = (Ome1_sin+Ome2_sin)/2.0*Qobj([[0,-1j,0],[1j,0,1j],[0,-1j,0]])/sqrt(2)   
    Hxdrive0 = Hdrive_cos+Hdrive_sin
    return Hxdrive0

def defH0(D0,Q,AN,AC_list,Bz):
    Hzfs = -D0*tensor(S0,III)
    Hzfsn = Q*tensor(III,S0)
    HhfNz = -AN*tensor(Sz,Sz)
    Hzeeman = Bz*tensor(Sz,III)
    H0 = Hzfs + Hzfsn + HhfNz + Hzeeman
    N = len(AC_list)
    if N!= 0:
        IIs_C = Qobj(qeye(2**N),dims=[list(2*ones(N)),list(2*ones(N))])
        H0 = tensor(H0,IIs_C)
    sum_SzSz = 0
    for i in range(N):
        IIb = qeye(2**i)
        IIa = qeye(2**(N-1-i))
        SzSz = Qobj(tensor(Sz,III,IIb,sigz,IIa),dims=[[3,3]+list(2*ones(N)),[3,3]+list(2*ones(N))])
        sum_SzSz = sum_SzSz+(-AC_list[i])*SzSz
                            
    return H0+sum_SzSz    

def Hamilton(pulse,freq,Ome1,Ome2,fai1,fai2,theta,H0):
    dims_full = H0.dims
    N = len(dims_full[0])-2
    if pulse == 0 :
            Hdrive = Qobj(tensor(Hxdrive(Ome1,Ome2,fai1,fai2,theta),III,qeye(2**N)),dims=dims_full)
            Hdetuning = -(freq)* Qobj(tensor(S0,III,qeye(2**N)),dims=dims_full)
            Hint = H0 + Hdrive - Hdetuning

    elif pulse == 1 :
            Hdrive = Qobj(tensor(III,Hxdrive(Ome1,Ome2,fai1,fai2,theta),qeye(2**N)),dims=dims_full)
            Hdetuning = (freq)* Qobj(tensor(III,S0,qeye(2**N)),dims=dims_full)
            Hint = H0 + Hdrive - Hdetuning
            
    else:
            i = pulse-2
            IIb = qeye(2**i)
            IIa = qeye(2**(N-1-i))
            Ug = Qobj(tensor(Sp,III,IIb,(exp(1j*fai1)*Sup+Sdown),IIa)+tensor(Sm,III,IIb,Sup+(exp(1j*fai1)*Sdown),IIa),dims=dims_full)
            Ur = Qobj(tensor(Sp,III,IIb,(exp(1j*(fai2-fai1))*Sup+Sdown),IIa)+tensor(Sm,III,IIb,(Sup+exp(1j*(fai2-fai1))*Sdown),IIa),dims=dims_full)
            SzzSx = Qobj(tensor(Szz,III,IIb,sigx,IIa),dims=dims_full)
            SzzSy = Qobj(tensor(Szz,III,IIb,sigy,IIa),dims=dims_full)
            SzSz = Qobj(tensor(Sz,III,IIb,sigz,IIa),dims=dims_full)
            Hrot_rfC = -(1/2.0)*(freq)*SzSz
            Hdrive = (1/2.0)*Ug*(Ome1*SzzSx+Ome2*Ur*(SzzSx*cos(theta)+SzzSy*sin(theta))*Ur.dag())*Ug.dag()
            Hint = H0 + Hdrive - Hrot_rfC
    return Hint



def Utime(H,t):
    Utime = (-2*pi*1j*H*t).expm()
    return Utime



def Hamilton_GRAPE(folder_name,theta,H0):
    file_name = list(zeros(3))
    file_name[0] = folder_name+"/MW_modu_Data_wire1.txt" 
    file_name[1] = folder_name+"/MW_modu_Data_wire2.txt"
    file_name[2] = folder_name+"/MW_time_Data.txt"

    #データの読み込み
    data_list = [[],[],[],[],[]]
    for i in range(3):        
        data_list[i] = loadtxt(file_name[i],"float","/t")

    t_div = data_list[2][1]
    Ome1  = data_list[0][:,0] 
    Ome2  = data_list[1][:,0] 
    frequency = data_list[0][:,1]/2/pi
    phi1  = data_list[0][:,2]
    phi2  = data_list[1][:,2]
    
    #ハミルトニアン
    H_list = []
    U_list = []
    for i in range(size(frequency)):
        H_div = Hamilton(0,frequency[i],Ome1[i],Ome2[i],phi1[i],phi2[i],theta,H0)            
        U_div = (-2*pi*1j*H_div*t_div).expm()
        H_list = H_list+[H_div]
        U_list = U_list+[U_div]
    
    return H_list,U_list,t_div    

def initialize(pE,pN,pC):
    #炭素はSupがpopulation1のとき初期化率100%とする。
    return tensor(pE*S0+(1-pE)/2.0*Szz, pN*S0+(1-pN)/2.0*Szz, pC*Sup+(1-pC)*Sdown)

def initialize(pE,pN,pC1,pC2):
    #炭素はSupがpopulation1のとき初期化率100%とする。
    return tensor(pE*S0+(1-pE)/2.0*Szz, pN*S0+(1-pN)/2.0*Szz, pC1*Sup+(1-pC1)*Sdown,pC2*Sup+(1-pC2)*Sdown)

def Projection(state_obs_Num,state_obs_list,state):
    Pro_list = []
    for i in range(size(state_obs_Num)):
        #Pro = fidelity(state_obs_list[state_obs_Num[i]-1],state)
        Pro = abs((state_obs_list[state_obs_Num[i]-1]*state).tr())
        Pro_list = Pro_list + [Pro]
    return Pro_list

def make_folder(dir_name,folder_name):
    time = datetime.datetime.today()
    if time.month <= 9:
        month = "0"+str(time.month)
    else:
        month = str(time.month)    

    i = 0
    new_folder_name = dir_name+str(time.year)+month+str(time.day)+"_"+folder_name
    if os.path.isdir(new_folder_name) == True:                              
        while os.path.isdir(new_folder_name) == True:
            new_folder_name = dir_name+str(time.year)+month+str(time.day)+"_"+folder_name+"_"+str(i)
            i=i+1
    os.mkdir(new_folder_name)
    return new_folder_name+"/"

def copy_storage(dir_name,dir_name_original,file_list):
    for i in range(size(file_list)):   
        if os.path.isfile(dir_name+file_list[i]) == True:
            print("\n\nError:"+dir_name+"に"+file_list[i]+"は既に存在します。\n\n")
            sys.exit()
        else:
            shutil.copyfile(dir_name_original+file_list[i],dir_name+file_list[i])
    return 0

def figure_storage(dir_name,file_name):
    if os.path.isfile(dir_name+file_name) == True:
        print("\n\nError:"+dir_name+"に"+file_name+"は既に存在します。\n\n")
        sys.exit()
    else:
        plt.savefig(dir_name+file_name)
    return 0    
    
def txt_storage(dir_name,file_name,txt):
    if os.path.isfile(dir_name+file_name) == True:
        print("\n\nError:"+dir_name+"に"+file_name+"は既に存在します。\n\n")
        sys.exit()
    else:
        file_txt = open(dir_name+file_name,"w")
        file_txt.write(txt)
        file_txt.close
    return 0       

def func_test():
    print("success")
    return 0
