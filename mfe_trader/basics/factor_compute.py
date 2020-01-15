# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:36:16 2019

@author: Administrator
"""
from datetime import datetime
from copy import deepcopy
from functools import reduce
from multiprocessing import cpu_count
from dask import dataframe as dd
from .stringtrans import body, repair
import numpy as np
import pandas as pd

cpu_n=cpu_count()

class FactorCompute(object):
    
    basic_factor = ["opened()","high()","low()","close()","volume()","opecinterst()"]
    
    def __init__(self):
        self._global_dict={}#self._global_dict应当只是运算数据的储存,计算完毕后数据就应当被清空
    
    def add_basic_factor(self, factor):
        if isinstance(factor, str) and factor not in self.basic_factor:
            self.basic_factor.append(factor)
        elif isinstance(factor, list):
            for i in factor:
                if i not in self.basic_factor:
                    self.basic_factor.append(factor)
    
    def remove_basic_factor(self, factor):
        if isinstance(factor, str) and factor in self.basic_factor:
            self.basic_factor.remove(factor)
        elif isinstance(factor, list):
            for i in factor:
                if i in self.basic_factor:
                    self.basic_factor.remove(factor)
    
    def set_basics_factor(self, factor_list):
        
        self.basic_factor = factor_list
        
    def get_basics_factor(self):
        
        return self.basic_factor
    
    def set_value(self,name,value):
            self._global_dict[name] = value
    
    def get_value(self,name=None,defValue=None):
        try:
            if name==None:
                result={}
                for i in self._global_dict.keys():
                    result[i]=self._global_dict[i]
                return result
            else:
                value=self._global_dict[name]
                return value
        except KeyError:
            return defValue
        
    def remove_value(self,name=None):
        if name in self._global_dict.keys():
            del self._global_dict[name]
        if not name:
            dicts = deepcopy(list(self._global_dict.keys()))
            for i in dicts:
                if i not in self.basic_factor:
                    del self._global_dict[i]
    
    def output(self, msg):
        print(f"{datetime.now()}\t{msg}")
            
    def compute(self,factor,overwrite=False,cpu_n=cpu_n):#运行因子计算前必须数据导入
        factor_trans=repair(body(factor))
        self.factor=factor
        self.factor_trans=factor_trans
        if overwrite==True:
            def check_dup(s):
                data=self.get_value()
                if s in data.keys():
                    print(s+' already exists and it is overwirtten now.')
                return False
        else:
            def check_dup(s):
                data=self.get_value()
                return (s in data.keys())
        
        def opened():
            return 'opened()'

        def high():
            return 'high()'

        def close():
            return "close()"

        def low():
            return "low()"

        def volume():
            return "volume()"

        def opeinterest():
            return "opeinterest()"

        def rank(df_name):#df_name是转换后的字符串
            s = 'rank' + '(' +df_name+ ')'#作为因子名
            if check_dup(s):
                return s
            df=self.get_value(df_name)
            if len(list(df.columns))==1:
                df_m=pd.DataFrame(np.array([1.0]*len(df.index)).reshape(-1,1),index=df.index,columns=df.columns)
            else:
                df_m=pd.DataFrame((df.rank(axis=1)-1.0).values/(df.rank(axis=1).max(axis=1).values.reshape(-1,1)-1.0),index=df.index,columns=df.columns)
            self.set_value(s,df_m)
            return s

        def delta(df_name,n):
            n = int(round(n))
            s = "delta(" + df_name + "," + str(n) + ")"
            if check_dup(s):
                return s
            df=self.get_value(df_name)
            if len(df.index)<=n:
                raise TypeError('data is not sufficient')
            df_m=df.diff(n)
            self.set_value(s,df_m)
            return s

        def plus(df_name1,df_name2):
            s = 'plus' + '(' + str(df_name1) + ',' + str(df_name2) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1)
                df2=self.get_value(df_name2)
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                df_m=df1+df2
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1)
                df_m=df1+df_name2
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2)
                df_m=df2+df_name1
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                df_m=df_name1+df_name2
            self.set_value(s,df_m)
            return s

        def subtract(df_name1, df_name2):
            s = 'subtract' + '(' + str(df_name1) + ',' + str(df_name2) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1)
                df2=self.get_value(df_name2)
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                df_m=df1-df2
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1)
                df_m=df1-df_name2
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2)
                df_m=df_name1-df2
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                df_m=df_name1-df_name2
            self.set_value(s,df_m)
            return s

        def multiply(df_name1,df_name2):
            s = 'multiply' + '(' + str(df_name1) + ',' + str(df_name2) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1)
                df2=self.get_value(df_name2)
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                df_m=df1*df2
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1)
                df_m=df1*df_name2
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2)
                df_m=df_name1*df2
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                df_m=df_name1*df_name2
            self.set_value(s,df_m)
            return s

        def divide(df_name1, df_name2):
            s = 'divide' + '(' + str(df_name1) + ',' + str(df_name2) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1)
                df2=self.get_value(df_name2)
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                df2[df2==0]=np.nan
                df_m=df1/df2
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1)
                if df_name2!=0:
                    df_name2_temp=df_name2
                else:
                    df_name2_temp=np.nan
                df_m=df1/df_name2_temp
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2)
                df2[df2==0]=np.nan
                df_m=df_name1/df2
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                if df_name2!=0:
                    df_name2_temp=df_name2
                else:
                    df_name2_temp=np.nan
                df_m=df_name1/df_name2_temp
            self.set_value(s,df_m)
            return s

        def power(df_name1, df_name2):
            s = 'power' + '(' + str(df_name1) + ',' + str(df_name2) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1)
                df2=self.get_value(df_name2)
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                df_m=df1.astype(float)**df2.astype(float)#Integers to negative integer powers are not allowed.
                df_m[df_m==np.inf]=np.nan
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1)
                df_m=df1.astype(float)**float(df_name2)#Integers to negative integer powers are not allowed.
                df_m[df_m==np.inf]=np.nan
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2)
                df_m=float(df_name1)**df2.astype(float)
                df_m[df_m==np.inf]=np.nan
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                df_m=float(df_name1)**float(df_name2)
                if df_m==np.inf:
                    df_m=np.nan
            self.set_value(s,df_m)
            return s

        def signedpower(df_name1, df_name2):
            s = 'signedpower' + '(' + str(df_name1) + ',' + str(df_name2) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1)
                df2=self.get_value(df_name2)
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                df_m=np.sign(df1) * abs(df1).astype(float) ** df2.astype(float)
                df_m[df_m==np.inf]=np.nan
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1)
                df_m=np.sign(df1)*df1.astype(float)**float(df_name2)#Integers to negative integer powers are not allowed.
                df_m[df_m==np.inf]=np.nan
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2)
                df_m=np.sign(df_name1)*float(df_name1)**df2.astype(float)
                df_m[df_m==np.inf]=np.nan
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                df_m=np.sign(df_name1)*float(df_name1)**float(df_name2)
                if df_m==np.inf:
                    df_m=np.nan
            self.set_value(s,df_m)
            return s

        def delay(df_name,n):
            #n = int(round(n))
            s = "delay(" + df_name + "," + str(n) + ")"
            if check_dup(s):
                return s
            df=self.get_value(df_name)
            if len(df.index)<n:
                raise TypeError('data is not sufficient')
            df_m=df.shift(n)
            self.set_value(s,df_m)
            return s

        def corr(df_name1,df_name2,n):
            n = int(round(n))
            s = "corr(" + df_name1 + "," + df_name2 + "," + str(n) + ")"
            if check_dup(s):
                return s
            df1=self.get_value(df_name1)
            df2=self.get_value(df_name2)
            if len(df1.index)<n:
                raise TypeError('data is not sufficient')
            if df1.shape!=df2.shape:
                raise TypeError('The shape of df2 is not the same as that of df1.')
            df_m=df1.rolling(n).corr(df2)
            self.set_value(s,df_m)
            return s

        def cov(df_name1,df_name2,n):
            n = int(round(n))
            s = "cov(" + df_name1 + "," + df_name2 + "," + str(n) + ")"
            if check_dup(s):
                return s
            df1=self.get_value(df_name1)
            df2=self.get_value(df_name2)
            if len(df1.index)<n:
                raise TypeError('data is not sufficient')
            if df1.shape!=df2.shape:
                raise TypeError('The shape of df2 is not the same as that of df1.')
            df_m=df1.rolling(n).cov(df2)
            self.set_value(s,df_m)
            return s

        def exponential(df_name):
            s = 'exponential' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.exp(df)
            else:
                df_m=np.exp(df_name)
            self.set_value(s,df_m)
            return s

        def logarithm(df_name):
            s = 'logarithm' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.log(df)
            else:
                df_m=np.log(df_name)
            self.set_value(s,df_m)
            return s

        def sign(df_name):
            s = 'sign' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.sign(df)
            else:
                df_m=np.sign(df_name)
            self.set_value(s,df_m)
            return s

        def arccos(df_name):
            s = 'arccos' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.arccos(df)
            else:
                df_m=np.arccos(df_name)
            self.set_value(s,df_m)
            return s

        def arcsin(df_name):
            s = 'arcsin' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.arcsin(df)
            else:
                df_m=np.arcsin(df_name)
                self.set_value(s,df_m)
            return s

        def arctan(df_name):
            s = 'arctan' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.arctan(df)
            else:
                df_m=np.arctan(df_name)
            self.set_value(s,df_m)
            return s

        def arccosh(df_name):
            s = 'arccosh' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.arccosh(df)
            else:
                df_m=np.arccosh(df_name)
            self.set_value(s,df_m)
            return s

        def arcsinh(df_name):
            s = 'arcsinh' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.arcsinh(df)
            else:
                df_m=np.arcsinh(df_name)
            self.set_value(s,df_m)
            return s

        def arctanh(df_name):
            s = 'arctanh' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.arctanh(df)
            else:
                df_m=np.arctanh(df_name)
            self.set_value(s,df_m)
            return s

        def cos(df_name):
            s = 'cos' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.cos(df)
            else:
                df_m=np.cos(df_name)
            self.set_value(s,df_m)
            return s

        def sin(df_name):
            s = 'sin' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.sin(df)
            else:
                df_m=np.sin(df_name)
            self.set_value(s,df_m)
            return s

        def tan(df_name):
            s = 'tan' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.tan(df)
            else:
                df_m=np.tan(df_name)
            self.set_value(s,df_m)
            return s

        def cosh(df_name):
            s = 'cosh' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.cosh(df)
            else:
                df_m=np.cosh(df_name)
            self.set_value(s,df_m)
            return s

        def tanh(df_name):
            s = 'tanh' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.tanh(df)
            else:
                df_m=np.tanh(df_name)
            self.set_value(s,df_m)
            return s

        def absolute(df_name):
            s = 'absolute' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.absolute(df)
            else:
                df_m=np.absolute(df_name)
            self.set_value(s,df_m)
            return s

        def ceiling(df_name):
            s = 'ceiling' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.ceil(df)
            else:
                df_m=np.ceil(df_name)
            self.set_value(s,df_m)
            return s

        def floor(df_name):
            s = 'floor' + '(' + str(df_name) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name,str):
                df=self.get_value(df_name)
                df_m=np.floor(df)
            else:
                df_m=np.floor(df_name)
            self.set_value(s,df_m)
            return s

        def tsminimum(df_name,n):
            n = int(round(n))
            s = "tsminimum(" + df_name + "," + str(n) + ")"
            if check_dup(s):
                return s
            df=self.get_value(df_name)
            if len(df.index)<n:
                raise TypeError('data is not sufficient')
            df_m=df.rolling(n).min()
            self.set_value(s,df_m)
            return s

        def tsmaximum(df_name,n):
            n = int(round(n))
            s = "tsmaximum(" + df_name + "," + str(n) + ")"
            if check_dup(s):
                return s
            df=self.get_value(df_name)
            if len(df.index)<n:
                raise TypeError('data is not sufficient')
            df_m=df.rolling(n).max()
            self.set_value(s,df_m)
            return s

        def tsrank(df_name,n):
            n = int(round(n))
            s = "tsrank(" + df_name + "," + str(n) + ")"
            if check_dup(s):
                return s
            df=self.get_value(df_name)
            if len(df.index)<n:
                raise TypeError('data is not sufficient')
            df=dd.from_pandas(self.get_value(df_name),npartitions=cpu_n)
            isnan=np.isnan
            sort=np.sort
            where=np.where
            def _rank_arr(array):
                s=array[~isnan(array)]
                res=where(sort(s)==s[-1])[0][0]+1
                return (res-1)/(len(s)-1)
            df_m=df.rolling(n).apply(_rank_arr).compute()
            self.set_value(s,df_m)
            return s

        def std(df_name,n):
            n = int(round(n))
            s = "std(" + df_name + "," + str(n) + ")"
            if check_dup(s):
                return s
            df=self.get_value(df_name)
            if len(df.index)<n:
                raise TypeError('data is not sufficient')
            df_m=df.rolling(n).std()
            self.set_value(s,df_m)
            return s

        def maximum(df_name1, df_name2):
            s = 'maximum' + '(' + str(df_name1) + ',' + str(df_name2) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1).fillna(-np.inf)
                df2=self.get_value(df_name2).fillna(-np.inf)
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                df_m=np.maximum(df1.fillna(-np.inf),df2.fillna(-np.inf))
                df_m[df_m==-np.inf]=np.nan
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1).fillna(-np.inf)
                df_m=np.maximum(df1,df_name2)
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2).fillna(-np.inf)
                df_m=np.maximum(df_name1,df2)
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                df_m=np.maximum(df_name1,df_name2)
            self.set_value(s,df_m)
            return s

        def minimum(df_name1, df_name2):
            s = 'minimum' + '(' + str(df_name1) + ',' + str(df_name2) + ')'
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1).fillna(np.inf)
                df2=self.get_value(df_name2).fillna(np.inf)
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                df_m=np.minimum(df1.fillna(np.inf),df2.fillna(np.inf))
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1).fillna(np.inf)
                df_m=np.minimum(df1,df_name2)
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2).fillna(np.inf)
                df_m=np.minimum(df_name1,df2)
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                df_m=np.minimum(df_name1,df_name2)
            self.set_value(s,df_m)
            return s

        def multiminimum(*df_name):
            list1 = []
            list2 = []
            for each in df_name:
                each_temp=self.get_value(each).fillna(value=np.inf)
                list1.append(each_temp)
                list2.append(each)
            shape=list1[0].shape
            for i in range(len(list2)):
                if list1[i].shape!=shape:
                    raise TypeError('Not all the shape of dataframes are the same' )
            s =str()
            for i in range(len(list2)):
                if i != len(list2) - 1:
                    s += str(list2[i]) + ','
                else:
                    s += str(list2[i])
            s = 'multiminimum' + '(' + s + ')'
            if check_dup(s):
                return s
            def min_df(df1,df2):
                return np.minimum(df1.fillna(np.inf),df2.fillna(np.inf))
            df_m=reduce(min_df,list1)
            df_m[df_m==np.inf]=np.nan
            self.set_value(s,df_m)
            return s

        def multimaximum(*df_name):
            list1 = []
            list2 = []
            for each in df_name:
                each_temp=self.get_value(each).fillna(value=-np.inf)
                list1.append(each_temp)
                list2.append(each)
            shape=list1[0].shape
            for i in range(len(list2)):
                if list1[i].shape!=shape:
                    raise TypeError('Not all the shape of dataframes are the same' )
            s =str()
            for i in range(len(list2)):
                if i != len(list2) - 1:
                    s += str(list2[i]) + ','
                else:
                    s += str(list2[i])
            s = 'multimaximum' + '(' + s + ')'
            if check_dup(s):
                return s
            def max_df(df1,df2):
                return np.maximum(df1.fillna(-np.inf),df2.fillna(-np.inf))
            df_m=reduce(max_df,list1)
            df_m[df_m==-np.inf]=np.nan
            self.set_value(s,df_m)
            return s

        def summation(df_name, n):
            n = int(round(n))
            s = 'summation' + '(' + df_name + ',' + str(n) + ')'
            if check_dup(s):
                return s
            df=self.get_value(df_name)
            if len(df.index)<n:
                raise TypeError('data is not sufficient')
            df_m=df.rolling(n).sum()
            self.set_value(s,df_m)
            return s

        def product(df_name,n):
            n = int(round(n))
            s = 'product' + '(' + df_name + ',' + str(n) + ')'
            if check_dup(s):
                return s
            df=dd.from_pandas(self.get_value(df_name),npartitions=cpu_n)
            if len(df.index)<n:
                raise TypeError('data is not sufficient')
            prod=np.product
            df_m=df.rolling(window=n).apply(prod).compute()
            self.set_value(s,df_m)
            return s

        def scale(df_name):
            s = "scale(" + df_name + ")"
            if check_dup(s):
                return s
            df=self.get_value(df_name)
            temp=np.abs(df).sum(axis=1)
            df_m=pd.DataFrame(np.array(df)/np.array(temp).reshape(-1,1),index=df.index,columns=df.columns)
            self.set_value(s,df_m)
            return s

        def tsmean(df_name, n):
            s = "tsmean(" + df_name + "," + str(n) + ")"
            if check_dup(s):
                return s
            df=self.get_value(df_name)
            if len(df.index)<n:
                raise TypeError('data is not sufficient')
            df_m=df.rolling(window=n).mean()
            self.set_value(s,df_m)
            return s

        def argminimum(df_name, n):
            n = int(round(n))
            s = "argminimum(" + df_name + "," + str(n) + ")"
            if check_dup(s):
                return s
            df=dd.from_pandas(self.get_value(df_name),npartitions=cpu_n)
            if len(df.index)<n:
                raise TypeError('data is not sufficient')
            argmin=np.argmin
            min_day=lambda x:n-1-argmin(x)
            df_m=df.rolling(window=n).apply(min_day).compute()
            self.set_value(s,df_m)
            return s

        def argmaximum(df_name, n):
            n = int(round(n))
            s = "argmaximum(" + df_name + "," + str(n) + ")"
            if check_dup(s):
                return s
            df=dd.from_pandas(self.get_value(df_name),npartitions=cpu_n)
            if len(df.index)<n:
                raise TypeError('data is not sufficient')
            argmax=np.argmax
            max_day=lambda x:n-1-argmax(x)
            df_m=df.rolling(window=n).apply(max_day).compute()
            self.set_value(s,df_m)
            return s

        def decayexponential(df_name, f, n):
            n = int(round(n))
            s = 'decayexponential' + '(' + df_name + ',' + str(f) + ',' + str(n) + ')'
            if check_dup(s):
                return s
            df=dd.from_pandas(self.get_value(df_name),npartitions=cpu_n)
            if len(df.index)<n:
                raise TypeError('data is not sufficient')
            step = range(0, n)
            f_arr=[f]*n
            weight = np.power(f_arr, step)
            summation=np.sum
            dot=np.dot
            decayexp=lambda x:dot(x,weight)/summation(weight)
            df_m=df.rolling(window=n).apply(decayexp).compute()
            self.set_value(s,df_m)
            return s

        def decaylinear(df_name, n):
            n = int(round(n))
            s = 'decaylinear' + '(' + df_name + ',' + str(n) + ')'
            if check_dup(s):
                return s
            df=dd.from_pandas(self.get_value(df_name),npartitions=cpu_n)
            if len(df.index)<n:
                raise TypeError('data is not sufficient')
            weight=np.arange(1,n+1)
            dot=np.dot
            summation=np.sum
            decaylin=lambda x:dot(x,weight)/summation(weight)
            df_m=df.rolling(window=n).apply(decaylin).compute()
            self.set_value(s,df_m)
            return s

        def tsregression(y, x, n, lag, retval):
            # Regression model y[t]= a+b*x[t-lag]
            # The values that can be returned (resid, a, b, estimate_of_y) are represented by 0, 1, 2, 3 respectively.
            # Parameter description: y is response variable, x is independent variable, n is regression sample size
            # lag is the number of delay days, retval controls return value
            s = "tsregression(" + y + "," + x + "," + str(n) + "," + str(lag) + "," + str(retval) + ")"
            if check_dup(s):
                return s
            n = int(round(n))
            lag = int(round(lag))
            df_y=self.get_value(y)
            df_x=self.get_value(x)
            if df_y.shape!=df_x.shape:
                raise TypeError('y and x do not have the same shape')
            if len(df_y.index)<n+lag or len(df_x.index)<n+lag:
                raise TypeError('data is not sufficient')
            y=df_y.iloc[lag:,:]
            index=y.index
            y.index=range(len(df_y.index)-lag)
            y.columns=range(len(df_y.columns))
            if lag!=0:
                x=df_x.iloc[:-lag,:]
            else:
                x=df_x
            x.index=range(len(df_x.index)-lag)
            x.columns=range(len(df_x.columns))
            covariance=x.rolling(n).cov(y)
            var=x.rolling(n).var()
            b=covariance/var
            a=y.rolling(n).mean()-b*x.rolling(n).mean()
            est_y=a+b*x
            resid=y-est_y
            r=pd.DataFrame(index=df_y.index[:lag],columns=df_y.columns)
            if retval==0:
                resid.index=index
                resid.columns=df_y.columns
                r=pd.concat([r,resid],axis=0,join='outer')
            elif retval==1:
                a.index=index
                a.columns=df_y.columns
                r=pd.concat([r,a],axis=0,join='outer')
            elif retval==2:
                b.index=index
                b.columns=df_y.columns
                r=pd.concat([r,b],axis=0,join='outer')
            elif retval==3:
                est_y.index=index
                est_y.columns=df_y.columns
                r=pd.concat([r,est_y],axis=0,join='outer')
            else:
                raise TypeError('The value of retval is invalid!')
            self.set_value(s,r)
            return s

        def returns():
            s = 'returns()'
            if check_dup(s):
                return s
            df=self.get_value('close()')
            if len(df.index)<2:
                raise TypeError('data is not sufficient')
            df_m=df.pct_change(1)
            self.set_value(s,df_m)
            return s
        
        def ic(df_name,n,m):
            n=int(round(n))
            m=int(round(m))
            s="ic("+df_name+','+str(n)+','+str(m)+")"
            if check_dup(s):
                return s
            if not isinstance(df_name,str):
                raise TypeError(df_name+'is not a string')
            return_str='delta(close,'+str(m)+')/delay(close,'+str(m)+')'
            self.factor_compute(return_str)
            returns=self.get_value(repair(body(return_str))).shift(-m)
            df=self.get_value(df_name)
            df_m=df.rolling(n).corr(returns)
            self.set_value(s,df_m)
            return s
        
        def notequal(df_name1, df_name2):
            s = "notequal(" + str(df_name1) + "," + str(df_name2) + ")"
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1)
                df2=self.get_value(df_name2)
                df1_null=df1.isnull().astype(float)
                df2_null=df2.isnull().astype(float)
                df_null=df1_null+df2_null
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                r=(df1!=df2).astype(float)
                r[df_null>0]=np.nan
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1)
                df1_null=df1.isnull()
                r=(df1!=df_name2).astype(float)
                r[df1_null==True]=np.nan
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2)
                df2_null=df2.isnull()
                r=(df2!=df_name1).astype(float)
                r[df2_null==True]=np.nan
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                if df_name1!=np.nan and df_name2!=np.nan: 
                    r=float(df_name1!=df_name2)
                else:
                    r=np.nan
            self.set_value(s,r)
            return s

        def lessthan(df_name1, df_name2):
            s = "lessthan(" + str(df_name1) + "," + str(df_name2) + ")"
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1)
                df2=self.get_value(df_name2)
                df1_null=df1.isnull().astype(float)
                df2_null=df2.isnull().astype(float)
                df_null=df1_null+df2_null
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                r=(df1<df2).astype(float)
                r[df_null>0]=np.nan
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1)
                df1_null=df1.isnull()
                r=(df1<df_name2).astype(float)
                r[df1_null==True]=np.nan
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2)
                df2_null=df2.isnull()
                r=(df_name1<df2).astype(float)
                r[df2_null==True]=np.nan
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                if df_name1!=np.nan and df_name2!=np.nan: 
                    r=float(df_name1<df_name2)
                else:
                    r=np.nan
            self.set_value(s,r)
            return s

        def lessorequal(df_name1, df_name2):
            s = "lessorequal(" + str(df_name1) + "," + str(df_name2) + ")"
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1)
                df2=self.get_value(df_name2)
                df1_null=df1.isnull().astype(float)
                df2_null=df2.isnull().astype(float)
                df_null=df1_null+df2_null
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                r=(df1<=df2).astype(float)
                r[df_null>0]=np.nan
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1)
                df1_null=df1.isnull()
                r=(df1<=df_name2).astype(float)
                r[df1_null==True]=np.nan
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2)
                df2_null=df2.isnull()
                r=(df_name1<=df2).astype(float)
                r[df2_null==True]=np.nan
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                if df_name1!=np.nan and df_name2!=np.nan: 
                    r=float(df_name1<=df_name2)
                else:
                    r=np.nan
            self.set_value(s,r)
            return s

        def greaterthan(df_name1, df_name2):
            s = "greaterthan(" + str(df_name1) + "," + str(df_name2) + ")"
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1)
                df2=self.get_value(df_name2)
                df1_null=df1.isnull().astype(float)
                df2_null=df2.isnull().astype(float)
                df_null=df1_null+df2_null
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                r=(df1>df2).astype(float)
                r[df_null>0]=np.nan
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1)
                df1_null=df1.isnull()
                r=(df1>df_name2).astype(float)
                r[df1_null==True]=np.nan
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2)
                df2_null=df2.isnull()
                r=(df_name1>df2).astype(float)
                r[df2_null==True]=np.nan
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                if df_name1!=np.nan and df_name2!=np.nan: 
                    r=float(df_name1>df_name2)
                else:
                    r=np.nan
            self.set_value(s,r)
            return s
    
        def greaterorequal(df_name1, df_name2):
            s = "greaterorequal(" + str(df_name1) + "," + str(df_name2) + ")"
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1)
                df2=self.get_value(df_name2)
                df1_null=df1.isnull().astype(float)
                df2_null=df2.isnull().astype(float)
                df_null=df1_null+df2_null
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                r=(df1>=df2).astype(float)
                r[df_null>0]=np.nan
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1)
                df1_null=df1.isnull()
                r=(df1>=df_name2).astype(float)
                r[df1_null==True]=np.nan
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2)
                df2_null=df2.isnull()
                r=(df_name1>=df2).astype(float)
                r[df2_null==True]=np.nan
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                if df_name1!=np.nan and df_name2!=np.nan: 
                    r=float(df_name1>=df_name2)
                else:
                    r=np.nan
            self.set_value(s,r)
            return s

        def equal(df_name1,df_name2):
            s = "equal(" + str(df_name1) + "," + str(df_name2) + ")"
            if check_dup(s):
                return s
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1)
                df2=self.get_value(df_name2)
                df1_null=df1.isnull().astype(float)
                df2_null=df2.isnull().astype(float)
                df_null=df1_null+df2_null
                if df1.shape!=df2.shape:
                    raise TypeError('The shape of df2 is not the same as that of df1.')
                r=(df1==df2).astype(float)
                r[df_null>0]=np.nan
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1)
                df1_null=df1.isnull()
                r=(df1==df_name2).astype(float)
                r[df1_null==True]=np.nan
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2)
                df2_null=df2.isnull()
                r=(df_name1==df2).astype(float)
                r[df2_null==True]=np.nan
            elif not isinstance(df_name1,str) and not isinstance(df_name2,str):
                if df_name1!=np.nan and df_name2!=np.nan: 
                    r=float(df_name1==df_name2)
                else:
                    r=np.nan
            self.set_value(s,r)
            return s

        def also(df_name1, df_name2):
            s = "also(" + df_name1 + "," + df_name2 + ")"
            if check_dup(s):
                return s
            df1=self.get_value(df_name1).astype(float)
            df2=self.get_value(df_name2).astype(float)
            r=df1*df2
            self.set_value(s,r)
            return s

        def oror(df_name1, df_name2):
            s = "oror(" + df_name1 + "," + df_name2 + ")"
            if check_dup(s):
                return s
            df1=self.get_value(df_name1).astype(float)
            df2=self.get_value(df_name2).astype(float)
            r=(df1+df2)>0
            self.set_value(s,r)
            return s

        def negative(df_name):
            s = "negative(" + df_name + ")"
            if check_dup(s):
                return s
            df=self.get_value(df_name).astype(float)
            r=(df!=1.0).astype(float)
            self.set_value(s,r)
            return s

        def condition(field1, field2, field3):
            s = "condition(" + str(field1) + "," + str(field2) + "," + str(field3) + ")"
            if check_dup(s):
                return s
            try:
                df1=self.get_value(field1)
            except:
                raise TypeError('field1 is not accepted')
            mask=np.isnan(df1.values)
            if isinstance(field2, str) and isinstance(field3, str):
                df2=self.get_value(field2)
                df3=self.get_value(field3)
                if df2.shape!=df3.shape:
                    raise TypeError('The shape of df3 is not the same as that of df2.')
                df_m=np.where(df1.values,df2.values,df3.values)
                df_m[mask]=np.nan
                result=pd.DataFrame(df_m,index=df1.index,columns=df1.columns)
            elif isinstance(field2,str):
                df2=self.get_value(field2)
                df_m=np.where(df1.values,df2.values,field3)
                df_m[mask]=np.nan
                result=pd.DataFrame(df_m,index=df1.index,columns=df1.columns)
            elif isinstance(field3,str):
                df3=self.get_value(field3)
                df_m=np.where(df1.values,field2,df3.values)
                df_m[mask]=np.nan
                result=pd.DataFrame(df_m,index=df1.index,columns=df1.columns)
            else:
                df_m=np.where(df1.values,field2,field3)
                df_m[mask]=np.nan
                result=pd.DataFrame(df_m,index=df1.index,columns=df1.columns)
            self.set_value(s,result)
            return s

        def counttt(condition, n):
            n = int(round(n))
            s = 'counttt(' + condition + ',' + str(n) + ')'
            if check_dup(s):
                return s
            df=dd.from_pandas(self.get_value(condition).astype(np.float),npartitions=cpu_n)
            if len(df.index)<n:
                raise TypeError('data is not sufficinet')
            f=lambda x:np.sum(x[~np.isnan(x)])
            df_m=df.rolling(n).apply(f).compute()
            self.set_value(s,df_m)
            return s

        def regbeta(Y, X, n):
            n = int(round(n))
            s = "regbeta(" + Y + "," + X + ")"
            if check_dup(s):
                return s
            temps = tsregression(Y, X, n, 0, 2)
            r=self.get_value(temps)
            self.set_value(s,r)
            return s
        
        def cross(df_name1,df_name2):
            s='cross('+str(df_name1)+','+str(df_name2)+')'
            if isinstance(df_name1,str) and isinstance(df_name2,str):
                df1=self.get_value(df_name1)
                df2=self.get_value(df_name2)
                judge=(df1>df2).astype(float)
                judge_shift=judge.shift(1)
                df_m=judge-judge_shift
            elif isinstance(df_name1,str):
                df1=self.get_value(df_name1)
                judge=(df1>df_name2).astype(float)
                judge_shift=judge.shift(1)
                df_m=judge-judge_shift
            elif isinstance(df_name2,str):
                df2=self.get_value(df_name2)
                judge=(df_name1>df2).astype(float)
                judge_shift=judge.shift(1)
                df_m=judge-judge_shift
            self.set_value(s,df_m)
            return s
        
        def nthcon(condition):
            s='nthcon('+condition+')'
            def count(x):
                cum_x=pd.Series(x).cumsum()
                sh_x=pd.Series(range(len(x)))
                s=pd.concat([cum_x,sh_x],axis=1)
                def minus_min(x):
                    return x-min(x)
                result=pd.Series(x).groupby(s['cum_x']).apply(minus_min)
                return result
            df=self.get_value(condition)
            df_m=df.apply(count)
            self.set_value(s,df_m)
            return s
            
        def sma(df_name, n, m):
            n = int(round(n))
            m = int(round(m))
            s = 'sma' + '(' + df_name + ',' + str(n) + ',' + str(m) + ')'
            if check_dup(s):
                return s
            df=self.get_value(df_name)
            if len(df.index)<n:
                raise TypeError('data is not sufficient')
            a = n * 1.0 / m - 1
            r = df.ewm(com=a, axis=0)
            df_m=r.mean()
            self.set_value(s,df_m)
            return s
        exec(factor_trans)
        self.output(f"{factor} has been computed")
        return self.get_value(self.factor_trans)