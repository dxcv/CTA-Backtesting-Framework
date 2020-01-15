# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 01:23:17 2019

@author: Administrator
"""

from ..template import Template
from mfe_trader.basics.constant import Interval

class C_Strategy(Template):
    
    factor_paramList = ["fastLength","slowLength"]
    general_paramList = ["volume"]
        
    def __init__(self, backtesting_engine,optimization = None):
        super(C_Strategy, self).__init__(backtesting_engine, optimization)
        self.volume = 1
        
    def OnInit(self):
        self.subscribe("I9888.CFFEX", Interval.Day_1)
        self.subscribe_factor("fastMA", "tsmean(close,12)")
        self.subscribe_factor("slowMA", "tsmean(close,14)")
        self.set_data_size(3)

    def OnBar(self, bar):
        if self.bar_count>=5:
            if self.array.fastMA.iloc[-1]>self.array.slowMA.iloc[-1] and self.pos == 0:
                self.buy("IF888.CFFEX",bar.close*1.01,5000)
            elif self.array.fastMA.iloc[-1]<self.array.slowMA.iloc[-1] and self.pos >0:
                self.sell("IF888.CFFEX", bar.close*0.99,self.pos)
            elif self.array.fastMA.iloc[-1]>self.array.slowMA.iloc[-1] and self.pos<0:
                self.buy_to_cover("IF888.CFFEX", bar.close*1.01,-self.pos)
            elif self.array.fastMA.iloc[-1]<self.array.slowMA.iloc[-1] and self.pos == 0:
                self.sell_short("IF888.CFFEX", bar.close*0.99,5000)
            
        self.strategy_output(self.pos)
        
    def OnOrder(self, order):
        pass
        #self.strategy_output(f"下单价格为:{round(order.price,3)},下单量为:{order.volume},状态为:{order.status.value}")
        
    def OnTrade(self, trade):
        pass
        #self.strategy_output(f"成交价格为:{round(trade.price,3)},成交量量为:{trade.volume}")