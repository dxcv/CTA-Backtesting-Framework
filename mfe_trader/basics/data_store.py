# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:02:03 2019

@author: Administrator
"""
from collections import defaultdict
from pandas import read_csv
from datetime import datetime

from mfe_trader.basics.constant import Interval

class DataStore:
    """数据处理类"""
    def __init__(self, backtesting_engine = None):
        """DataStore对应的回测类"""
        self.backtesting_engine = backtesting_engine
        
        """相关数据信息"""
        self.vtsymbol_data_map = defaultdict(dict)      #vtsymbol:行情数据
        self.vtsymbols = []                             #vtsymbol的列表
        self.intervals = []                             #interval的列表
        self.vtsymbol_interval_map = defaultdict(list)  #vtsymbol:interval
        
        """interval从频率低到频率高的排序"""
        self.interval_sort = {Interval.Min_1:1,
                              Interval.Min_5:2,
                              Interval.Min_15:3,
                              Interval.Min_30:4,
                              Interval.Hour_1:5,
                              Interval.Day_1:6,
                              Interval.Week_1:7,
                              Interval.Month_1:8}
        
    def add_data_from_csv(self, setting: dict):
        """向数据处理类中添加csv数据"""
        if "vtsymbol" not in setting:
            raise ValueError(" vtsymbol is not in setting.")
        if "interval" not in setting:
            raise ValueError("interval is not in setting.")
        if "path" not in setting:
            raise ValueError("path is not in setting.")
        if "datetime_row" not in setting:
            raise ValueError("datetime_row is not in setting.")
        
        """生成symbol_data"""
        symbol_data = SymbolDataDf(setting["vtsymbol"], 
                                      setting["interval"], 
                                      setting["path"], 
                                      setting["datetime_row"])
        
        settings = {"vtsymbol":symbol_data.vtsymbol,
                    "interval":symbol_data.interval,
                    "data":symbol_data.data
                }
        
        self.update_data_info(settings)
        
    def update_data_info(self, setting):
        """
        更新：
        vtsymbol_data_map,vtsymbols,vtsymbol_interval_map,intervals
        """
        vtsymbol = setting["vtsymbol"]
        interval = setting["interval"]
        data = setting["data"]
        
        self.vtsymbol_data_map[vtsymbol][interval] = data
        if vtsymbol not in self.vtsymbols:
            self.vtsymbols.append(vtsymbol)
        if interval not in self.vtsymbol_interval_map[vtsymbol]:
            self.vtsymbol_interval_map[vtsymbol].append(interval)
        if interval not in self.intervals:
            self.intervals.append(interval)
        
    def func_interval_sort(self, interval1, interval2):
        """
        比较两个Interval的周期长短
        周期: interval1<interval2，返回True
        周期: interval1>interval2, 返回False
        """
        interval_sort = self.interval_sort
        if interval_sort[interval1]<interval_sort[interval2]:
            return True
        else:
            return False
        
    def resample(self, vtsymbol: str, target_freq: Interval):
        """将高频率的数据转化为低频率数据"""
        trans = {Interval.Min_1:"1min",
                 Interval.Min_5:"5min",
                 Interval.Min_15:"15min",
                 Interval.Min_30:"30min",
                 Interval.Hour_1:"1h",
                 Interval.Day_1:"1d",
                 Interval.Week_1:"1w",
                 Interval.Month_1:"1M"
                }
        intervals = self.vtsymbol_interval_map[vtsymbol]
        
        if target_freq in intervals:
            """如果所求频率数据已经存在，则直接返回数据"""
            self.output("Data with target_freq has already exists")
            return self.vtsymbol_data_map[vtsymbol][target_freq]
        
        for i in intervals:
            """如果所求频率数据不存在"""
            if self.func_interval_sort(i, target_freq):
                """如果当前数据存在比target_freq频率更高的数据，则可进行降频"""
                target_data = self.vtsymbol_data_map[vtsymbol][i].resample(trans[target_freq]).agg({
                        "open":"first","high":"max",
                        "low":"min","close":"last",
                        "volume":"sum","openinterest":"last"}).dropna(axis=0)
                self.vtsymbol_data_map[vtsymbol][target_freq] = target_data
                self.vtsymbol_interval_map[vtsymbol].append(target_freq)
                self.intervals.append(target_freq)
                self.output("Data with target_freq has been generated")
                return target_data
        
            if (intervals.index(i) == len(intervals)-1) and (not self.func_interval_sort(i,target_freq)):
                raise ValueError("Resample fails")
                
class SymbolDataDf:
    
    """数据类，从csv文件中导入数据"""
    def __init__(self, vtsymbol: str, interval: Interval, path: str, datetime_row: str):
        
        self.vtsymbol = vtsymbol
        self.interval = interval
        self.load_csv(path, datetime_row)
        
    def load_csv(self, path: str, datetime_row: str):
        """载入csv文件中的行情数据"""
        try:
            df = read_csv(path)
        except:
            df = read_csv(path, encoding = "ANSI")
        
        df.set_index(datetime_row, inplace = True)
        df.index = self.stand_index(df.index)
        
        self.data = df
    
    def stand_index(self, datelist):
        """
        将导入的数据的时间索引改成datetime格式
        """
        def _stand_date1(date):#将日期形式转化成标准的datetime形式
            try:
                return datetime.strptime(date,"%Y-%m-%d")
            except:
                try:
                    return datetime.strptime(date,"%Y-%m-%d %H:%M")
                except:
                    return datetime.strptime(date,"%Y-%m-%d %H:%M:%S")
            
        def _stand_date2(date):#将tb的时间转化为有效的datetime的形式
            try:
                return datetime.strptime(date,'%Y/%m/%d %H:%M')
            except:
                try:
                    return datetime.strptime(date,'%Y/%m/%d')
                except:
                    return datetime.strptime(date,"%Y/%m/%d %H:%M:%S")
                
        def _stand_date3(date):
            try:
                return datetime.strptime(date,"%Y\%m\%d %H:%M")
            except:
                try:
                    return datetime.strptime(date,"%Y\%m\%d")
                except:
                    return datetime.strptime(date,"%Y\%m\%d %H:%M:%S")
        
        def _stand_date4(date):
            try:
                return datetime.strptime(date,"%m/%d/%Y")
            except:
                try:
                    return datetime.strptime(date,"%m/%d/%Y %H:%M")
                except:
                    return datetime.strptime(date,"%m/%d/%Y %H:%M:%S")

        def stand_date(i):
            try:
                return _stand_date1(i)
            except:
                try:
                    return _stand_date2(i)
                except:
                    try:
                        return _stand_date3(i)
                    except:
                        return _stand_date4(i)
        result = [stand_date(i) for i in datelist]
        result = sorted(result)
        return result
        
    def output(self, msg):
        print(f"{datetime.now()}\t{msg}")