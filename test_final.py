# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:27:33 2019

@author: Administrator
"""

from datetime import datetime
from mfe_trader.basics.constant import Interval
from mfe_trader.backtesting.Backtesting import BacktestingEngine

#%%
from mfe_trader.strategy.strategies.D_Strategy1 import D_Strategy1
setting1_1 = {"vtsymbol":"HC888.SHFE", 
           "interval": Interval.Day_1,
           "path": r"hc888_1d.csv", 
           "datetime_row": "datetime"}
engine1 = BacktestingEngine()
engine1.add_data(setting1_1)
engine1.add_strategy(D_Strategy1)
setting1_2 = {"start":datetime(2011,5,1,0,0),
           "end":datetime(2019,11,5,0,0),
           "captial":10000000,
           "rate":0.000185,
           "slippage":0}#一定要设置成datetime_list里面有的
engine1.set_parameters(setting1_2)

engine1.run_backtest()
engine1.calculate_result()
result1 = engine1.calculate_statistics()
engine1.show_chart()

#%%
from mfe_trader.strategy.strategies.A_Strategy import A_Strategy
setting2_1 = {"vtsymbol":"I9888.DCE", 
           "interval": Interval.Day_1,
           "path": r"i9888_1d.csv", 
           "datetime_row": "datetime"}
engine2 = BacktestingEngine()
engine2.add_data(setting2_1)
engine2.add_strategy(A_Strategy)
setting2_2 = {"start":datetime(2011,5,1,0,0),
           "end":datetime(2019,11,5,0,0),
           "captial":10000000,
           "rate":0.000155,
           "slippage":0}#一定要设置成datetime_list里面有的
engine2.set_parameters(setting2_2)

engine2.run_backtest()
engine2.calculate_result()
result2 = engine2.calculate_statistics()
engine2.show_chart()

#%%
from mfe_trader.strategy.strategies.A_Strategy import A_Strategy
setting3_1 = {"vtsymbol":"J9888.DCE", 
           "interval": Interval.Day_1,
           "path": r"j9888_1d.csv", 
           "datetime_row": "datetime"}
engine3 = BacktestingEngine()
engine3.add_data(setting3_1)
engine3.add_strategy(A_Strategy)
setting3_2 = {"start":datetime(2011,5,1,0,0),
           "end":datetime(2019,11,5,0,0),
           "captial":10000000,
           "rate":0.000185,
           "slippage":0}#一定要设置成datetime_list里面有的
engine3.set_parameters(setting3_2)

engine3.run_backtest()
engine3.calculate_result()
result3 = engine3.calculate_statistics()
engine3.show_chart()
#%%
engine_1 =engine3+engine2+engine2+engine2+engine1