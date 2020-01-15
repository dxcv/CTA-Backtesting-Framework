# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:45:54 2019

@author: Administrator
"""
from datetime import datetime
from abc import abstractmethod

class Template:
    
    def __init__(self, backtesting_engine, optimization = None):
        self.optimization = optimization
        self.backtesting_engine = backtesting_engine
        self.pos = 0
        self.bar_count = 0
        self.data_size = 1
        self.bars_since_entry = 0
        self.last_entry_price = 0
        self.last_pos = 0
        
        self.factors = {}
        
    @abstractmethod
    def OnInit(self):
        """订阅数据并确定回测频率"""
        raise NotImplementedError
    @abstractmethod
    def OnBar(self, bar):
        """K线回调函数，写交易逻辑"""
        raise NotImplementedError
    
    @abstractmethod
    def OnOrder(self, order):
        """委托回调函数"""
        raise NotImplementedError
        
    @abstractmethod
    def OnTrade(self, trade):
        """成交回调函数"""
    
    def set_data_size(self, length: int):
        self.data_size = length
        self.backtesting_engine.data_size = length
        
    def subscribe_factor(self, name, factor):
        if not self.optimization:
            self.backtesting_engine.subscribe_factor(name, factor)
        if self.optimization:
            """如果是优化参数情况下，只要记录下factors即可，在optimization的init_strategy方法中会实现相关功能"""
            self.factors[name] = factor

    def buy(self, price, volume):
        self.backtesting_engine.buy(price, volume)
        
    def sell(self, price, volume):
        self.backtesting_engine.sell(price, volume)
    
    def sell_short(self, price, volume):
        self.backtesting_engine.sell_short(price, volume)
        
    def buy_to_cover(self,price, volume):
        self.backtesting_engine.buy_to_cover(price,volume)
        
    def output(self, msg):
        print(f"{datetime.now()}\t{msg}")
        
    def strategy_output(self, msg):
        print(f"策略回测时间:{self.datetime}--INFO--{msg}")