# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:01:15 2018

@author: shiro
"""

    def Estimate_credible_region(self,level):
        id_sorted=np.argsort(self.w,axis=0)[::-1]
        w_sorted=np.sort(self.w,axis=0)[::-1]
        cumsum_weights=np.cumsum(w_sorted)
        id_cred=cumsum_weights<=level
        print(id_cred)
        if((id_cred==True).all()):
            x_range_temp=self.x[id_sorted[0]]
            x_range=np.reshape(x_range_temp,[len(x_range_temp),len(self.x[0])])
        else:
            x_range_temp=self.x[id_sorted][id_cred]
            x_range=np.reshape(x_range_temp,[len(x_range_temp),len(self.x[0])])
            
        return x_range
