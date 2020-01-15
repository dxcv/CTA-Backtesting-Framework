# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:44:05 2019

@author: Administrator
"""
import time
from datetime import datetime
from itertools import product
from pandas import DataFrame
from mfe_trader.strategy.template import Template
from mfe_trader.basics.data_store import DataStore
from mfe_trader.backtesting.Backtesting import BacktestingEngine

class Optimization:

    def __init__(self, settings: dict):
        """构造函数"""
        self.data_store_center = DataStore(self)
        
        self.params = settings["params"]
        self.target = settings["target"]
        self.backtesting_params = settings["backtesting"]
        
        self.init_params(settings["params"])
        
        self.add_data(settings["data"])
        self.add_strategy(settings["strategy"])
        
        self.engines = []
        
        self.optimization_result = DataFrame(index = list(self.params.keys())+[self.target])
        
        self.results = {}
                
    def add_strategy(self, strategy: Template):
        self.strategy = strategy
        self.factor_paramList = self.strategy.factor_paramList
        self.general_paramList = self.strategy.general_paramList
        
    def add_data(self, settings: dict):
        self.data_store_center.add_data_from_csv(settings)
        
    def init_strategy(self):
        
        append = self.engines.append
        start = time.time()
        for params in self.param_df.itertuples():
            param = {}
            engine = BacktestingEngine(data_store=self.data_store_center,optimization = self)
            engine.set_parameters(self.backtesting_params)
            engine.add_strategy(self.strategy)
            for name in self.factor_paramList:
                param[name] = getattr(params,name)
                for factor in engine.strategy.factors:
                    if name in engine.strategy.factors[factor]:
                        engine.strategy.factors[factor] = engine.strategy.factors[factor].replace(name,str(getattr(params,name)))
                        engine.subscribe_factor(factor, engine.strategy.factors[factor])
                
            for name in self.general_paramList:
                param[name] = getattr(params,name)
                setattr(engine.strategy, name, getattr(params, name))
            append([param,engine])
        end = time.time()
        self.output(f"不同参数条件下策略初始化完毕,耗时{round(end-start,4)}秒")
    
    def run_backtesting(self, param, engine):
        engine.run_backtest()
        result = engine.calculate_result()
        self.results[list(param.values())] = result[self.target]

    def run_optimization(self):
        for param, engine in self.engines:
            self.output(f"回测第{self.engines.index([param,engine])+1}个策略")
            engine.run_backtest()
            engine.calculate_result()
            result = engine.calculate_statistics()
            self.optimization_result.append(list(param.values())+[result[self.target]])
        self.optimization_result = self.optimization_result.T
        return self.optimization_result
        
    def init_params(self, params: dict):
        """将参数规整化"""
        param_df = []
        append = param_df.append
        for item in product(*(params.values())):
            append(list(item))
            
        self.param_df = DataFrame(param_df)
        self.param_df.columns = list(params.keys())
        return self.param_df
    
    def output(self, msg):
        print(f"{datetime.now()}\t{msg}")