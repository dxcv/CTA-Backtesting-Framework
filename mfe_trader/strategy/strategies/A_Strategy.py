# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:46:37 2019

@author: Administrator
"""
from ..template import Template

class A_Strategy(Template):
    
    factor_paramList = ["fastLength","slowLength"]
    general_paramList = ["volume"]
        
    def __init__(self, backtesting_engine,optimization = None):
        super(A_Strategy, self).__init__(backtesting_engine, optimization)
        self.volume = 1
        
    def OnInit(self):
        self.subscribe_factor("fastMA", "tsmean(close,12)")
        self.subscribe_factor("slowMA", "tsmean(close,14)")
        self.set_data_size(3)

    def OnBar(self, bar):
        if self.bar_count>=5:
            if self.array.fastMA.iloc[-1]>self.array.slowMA.iloc[-1] and self.pos == 0:
                self.buy(99999,10000)
            elif self.pos<0 and self.array.close.iloc[-1]>self.last_entry_price*(1+0.01):
                self.buy_to_cover(99999,-self.pos)
            elif self.array.fastMA.iloc[-1]<self.array.slowMA.iloc[-1] and self.pos >0:
                self.sell(1,self.pos)
            elif self.array.fastMA.iloc[-1]>self.array.slowMA.iloc[-1] and self.pos<0:
                self.buy_to_cover(99999,-self.pos)
            elif self.array.fastMA.iloc[-1]<self.array.slowMA.iloc[-1] and self.pos == 0:
                self.sell_short(1,10000)
            
        self.strategy_output(self.pos)
        
    def OnOrder(self, order):
        pass
        #self.strategy_output(f"下单价格为:{round(order.price,3)},下单量为:{order.volume},状态为:{order.status.value}")
        
    def OnTrade(self, trade):
        self.last_entry_price = trade.price