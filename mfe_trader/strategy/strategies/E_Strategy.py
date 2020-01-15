# RSI
import talib as ta
import numpy as np
from ..template import Template

class E_Strategy(Template):
    
    factor_paramList = ["fastLength", "slowLength"]
    general_paramList = ["volume"]
        
    def __init__(self, backtesting_engine,optimization = None):
        super(E_Strategy, self).__init__(backtesting_engine, optimization)
        self.volume = 1
        
    def OnInit(self):
        self.subscribe_factor("fastMA", "tsmean(close,12)")
        self.subscribe_factor("slowMA", "tsmean(close,14)")
        self.set_data_size(15)

    def OnBar(self, bar):
        if self.bar_count>=16:
            close = np.array(self.array.close)
            signal = ta.real = ta.RSI(close, timeperiod=14)             
            if signal[-1] <= 30:
                if self.pos == 0:
                    self.buy(99999, 5000)
                elif self.pos < 0:
                    self.buy_to_cover(99999, -self.pos)
            elif signal[-1] >= 70:
                if self.pos == 0:
                    self.sell_short(1, 5000)
                elif self.pos > 0:
                    self.sell(1, self.pos)
            else:
                if self.pos < 0:
                    self.buy_to_cover(9999, -self.pos)
                elif self.pos > 0:
                    self.sell(1, self.pos)
            
        self.strategy_output(self.pos)
        
    def OnOrder(self, order):
        pass
        
    def OnTrade(self, trade):
        pass