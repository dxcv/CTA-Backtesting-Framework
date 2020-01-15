# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:25:28 2019

@author: Administrator
"""

from datetime import datetime
from mfe_trader.basics.constant import Interval
from mfe_trader.backtesting.Optimization import Optimization
from mfe_trader.strategy.strategies.A_Strategy import A_Strategy

setting1 = {"params":{"fastLength":range(1,10),
                      "slowLength":range(2,20),
                      "volume":range(1,10)
                     },
            "target":"sharpe_ratio",
            "backtesting":{"start":datetime(2011,1,5,0,0),
                           "end":datetime(2019,1,5,0,0),
                           "captial":10000000,
                           "rate":0.001,
                           "slippage":0},
            "data":{"vtsymbol":"RB888.SHFE", 
                     "interval": Interval.Day_1,
                     "path": "rb888_1d.csv", 
                     "datetime_row": "datetime"},
            "strategy": A_Strategy
           }
opt = Optimization(setting1)
opt.init_strategy()
opt.run_optimization()