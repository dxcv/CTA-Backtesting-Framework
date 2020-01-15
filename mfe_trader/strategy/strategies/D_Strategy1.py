# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:16:28 2019

@author: Administrator
"""

import talib as ta
import numpy as np
from ..template import Template

class D_Strategy1(Template):
    
    factor_paramList = ["fastLength", "slowLength"]
    general_paramList = ["volume"]
        
    def __init__(self, backtesting_engine,optimization = None):
        super(D_Strategy1, self).__init__(backtesting_engine, optimization)
        self.volume = 1
        
    def OnInit(self):
        self.subscribe_factor("fastMA", "tsmean(close,5)")
        self.subscribe_factor("slowMA", "tsmean(close,30)")
        self.set_data_size(35)

    def OnBar(self, bar):
        if self.bar_count>=38:
            close = np.array(self.array.close).astype(float)
            _, signal, _ = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            if signal[-1] > 0 and self.array.fastMA.iloc[-1]>self.array.slowMA.iloc[-1] and self.pos == 0:
                self.buy(99999,10000)
            elif signal[-1] < 0 and self.pos == 0:
                self.sell_short(1,10000)
            elif self.pos>0 and self.array.fastMA.iloc[-1]<self.array.slowMA.iloc[-1]:
                self.sell(1,self.pos)
            elif self.pos<0 and self.array.close.iloc[-1]>self.last_entry_price*(1+0.01):
                self.buy_to_cover(99999,-self.pos)
            elif signal[-1] < 0 and self.pos >0:
                self.sell(1,self.pos)
            elif signal[-1] > 0 and self.pos<0:
                self.buy_to_cover(99999,-self.pos)
            
            
            
        self.strategy_output(self.pos)
        
    def OnOrder(self, order):
        pass
        
    def OnTrade(self, trade):
        self.last_entry_price = trade.price