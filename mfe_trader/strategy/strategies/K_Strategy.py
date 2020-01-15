# AROONOSC
import talib as ta
import numpy as np
from ..template import Template

class K_Strategy(Template):
    
    factor_paramList = ["fastLength", "slowLength"]
    general_paramList = ["volume"]
        
    def __init__(self, backtesting_engine,optimization = None):
        super(K_Strategy, self).__init__(backtesting_engine, optimization)
        self.volume = 1
        
    def OnInit(self):
        self.subscribe_factor("fastMA", "tsmean(close,12)")
        self.subscribe_factor("slowMA", "tsmean(close,14)")
        self.set_data_size(40)

    def OnBar(self, bar):
        if self.bar_count >= 41:
            close = np.array(self.array.close)
            signal = ta.PPO(close, fastperiod=12, slowperiod=26, matype=0)
            if signal[-1] > 0 and self.pos == 0:
                self.buy(99999,5000)
            elif signal[-1] < 0 and self.pos >0:
                self.sell(1,self.pos)
            elif signal[-1] > 0 and self.pos<0:
                self.buy_to_cover(99999,-self.pos)
            elif signal[-1] < 0 and self.pos == 0:
                self.sell_short(1,5000)
            
        self.strategy_output(self.pos)
        
    def OnOrder(self, order):
        pass
        
    def OnTrade(self, trade):
        pass