# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:01:15 2018

@author: shiro
"""

"""
Bayes_functionに関する変更
"""

    """
    def __init__(self)内に追加
    """
        #信用区間
        self.xout_in_region_max={"a1":[],"b1":[],"a2":[],"b2":[],"w_theta":[],"D0":[],"AN":[],"QN":[],"Bz":[]}
        self.xout_in_region_min={"a1":[],"b1":[],"a2":[],"b2":[],"w_theta":[],"D0":[],"AN":[],"QN":[],"Bz":[]}
      

    def Estimate_credible_region(self,level):
        id_sorted=np.argsort(self.w,axis=0)[::-1]
        w_sorted=np.sort(self.w,axis=0)[::-1]
        cumsum_weights=np.cumsum(w_sorted)
        id_cred=cumsum_weights<=level
        if((id_cred==False).all()):
            x_range=self.x[id_sorted[0]]
        else:
            x_range_temp=self.x[id_sorted][id_cred]
            x_range=np.reshape(x_range_temp,[len(x_range_temp),len(self.x[0])])
        return x_range
    
    def Region_edge(self,level,param):
        """
        各推定における信用区間に含まれるパーティクルの最大もしくは最小値のarrayを得る関数
        paramで推定パラメーターを指定
        """
        x_region=self.Estimate_credible_region(level)
        print(x_region.shape)
        for i,p in enumerate(self.ParamH):
            if self.ParamH[p]==1 and p==param:
                temp=[]
                for j in range(x_region.shape[0]):
                    temp.append(x_region[j][i])
                self.xout_in_region_max[p].append(max(temp))
                self.xout_in_region_min[p].append(min(temp))
                
    def Region_edge_output(self,param,flag):
        """
        flagで戻り値が最大か最小かを指定(1:max,0:min)
        """
        for i,p in enumerate(self.ParamH):
            if self.ParamH[p]==1 and p==param:
                if flag==1:
                    print("xout_in_region_max",self.xout_in_region_max)
                    return self.xout_in_region_max[p]
                else:
                    print("xout_in_region_min",self.xout_in_region_min)
                    return self.xout_in_region_min[p]
                
                
"""
HL_koga_classに関する変更
"""
    m.Region_edge(0.95,"a1")
    """
    a_list.append(xout[0][0])
    plt.subplot(2,1,1)
    """
    plt.fill_between(m.i_list,m.Region_edge_output("a1",1),m.Region_edge_output("a1",0),color='lightgreen',alpha=0.3)
    """
    plt.hlines(m.x0_dict["a1"],0,m.i_list[i],"r",label="True a1")
    plt.xlabel("iteration number",fontsize=20)
    plt.ylabel("a1",fontsize=20)
    plt.plot(m.i_list,a_list,label="Infered a1")
    plt.legend(loc="upper right")
    plt.title("a1",fontsize=24)
    """
    
    
    m.Region_edge(0.95,"b1")
    """
    plt.subplot(2,1,2)
    plt.hlines(m.x0_dict["b1"],0,m.i_list[i],"r",label="True b1")
    plt.xlabel("iteration number",fontsize=20)
    plt.ylabel("b1",fontsize=20)
    b_list.append(xout[0][1])
    plt.plot(m.i_list,b_list,label="Infered b1")
    plt.legend(loc="upper right")
    plt.title("b1",fontsize=24)
    """
    plt.fill_between(m.i_list,m.Region_edge_output("b1",1),m.Region_edge_output("b1",0),color='lightgreen',alpha=0.3)
    
