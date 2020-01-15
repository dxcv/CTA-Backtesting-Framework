# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:50:06 2019

@author: Administrator
"""
import numpy as np
from datetime import date, datetime
from pandas import DataFrame
import matplotlib.pyplot as plt
from collections import defaultdict

from mfe_trader.basics.stringtrans import body, repair
from mfe_trader.basics.constant import (Interval,
                                        Direction,
                                        Offset,
                                        Status)
from mfe_trader.basics.functions import extract_vt_symbol
from mfe_trader.basics.object import (BarData,
                                      OrderData,
                                      TradeData)
from mfe_trader.basics.data_store import DataStore
from mfe_trader.basics.factor_compute import FactorCompute

class BacktestingEngine:
    
    """回测参数"""
    paramList = ["start","end","captial","rate",
                 "slippage","size","pricetick"
                 ]
    
    """基本元转换字典"""
    name_trans = {"open":"opened()","high":"high()","low":"low()","close":"close()",
                  "volume":"volume()","opeinterest":"opeinterest()"}
    
    def __init__(self, data_store = None, optimization = None):
        
        self.optimization = optimization
        """数据"""
        if not data_store:
            self.data_store = DataStore(self)
        else:
            self.data_store = data_store
            #self.update_data_info()
            
        self.history_data = None
        
        """标的信息"""
        self.vtsymbol = ""
        self.interval: Interval = Interval.Day_1
        self.symbol = ""
        self.exchange = None
        self.start = None
        self.end = None
        self.capital = 10000000
        self.rate = 0
        self.slippage = 0
        self.size = 1
        self.pricetick = 0
        
        """策略信息"""
        self.strategy = None
        self.bar: BarData
        self.datetime = None
        self.data_size = 10
        
        """委托信息"""
        self.active_limit_orders = {}
        self.limit_orders = {}
        self.limit_order_count = 0
        
        """成交信息"""
        self.trades = {}
        self.trade_count = 0
        
        """每日结果信息"""
        self.daily_results = {}
        self.daily_df = None
        
        """历史数据"""
        self.history_data = None
        
        """因子信息"""
        self.factors = []
        self.factor_ftrans_map = {}
        self.factor_value_map = {}
        self.ftrans_value_map = {}
        self.additional_factor = []
        
        """初始化因子计算框架"""
        self.factor_compute = FactorCompute()

    def init_factor_data(self):
        for factor in self.history_data.columns:
            self.factors.append(factor)
            factor_trans = repair(body(factor))
            self.factor_ftrans_map[factor] = factor_trans
            value = self.history_data[[factor]]
            value.columns = [self.vtsymbol]
            self.factor_value_map[factor] = value
            self.factor_value_map[factor_trans] = value
            
            if factor in self.name_trans:
                trans_name = self.name_trans[factor]
                self.factor_compute.set_value(trans_name, value)
            
    def set_basic_factor(self, factor: list):
        """设置基本元"""
        self.factor_compute.set_basic_factor(factor)
        
    def add_basic_factor(self, factor):
        """添加基本元"""
        self.factor_compute.add_basic_factor(factor)
        
    def remove_basic_factor(self, factor):
        """删除基本元"""
        self.factor_compute.remove_basic_factor(factor)
    
    def get_basic_factor(self):
        """获取基本元"""
        return self.factor_compute.get_basic_factor()
    
    def set_basics_trans(self, name_trans: dict):
        """设置名称转换字典"""
        self.name_trans = name_trans
        
    def add_basics_trans(self, name, trans_name):
        """添加名称转换"""
        self.name_trans[name] = trans_name
        
    def remove_basics_trans(self, name):
        """删除名称转换"""
        del self.name_trans[name]
        
    def compute_factor(self, factor: str):
        """因子计算"""
        self.factors.append(factor)
        
        factor_trans = repair(body(factor))
        self.factor_ftrans_map[factor] = factor_trans
        
        value = self.factor_compute.compute(factor)
        self.factor_value_map[factor] = value
        self.ftrans_value_map[factor_trans] = value
        
        self.factor_compute.remove_value()          #计算完因子后将除基本元以外的数据删除，避免影响新因子计算的效率
        return value
        
    def subscribe_factor(self, name, factor):
        """订阅因子数据并添加到回测历史数据中"""
        self.additional_factor.append(name)
        value = self.compute_factor(factor)
        if not isinstance(self.history_data, DataFrame):
            raise ValueError("factor cannot be added when history data is empty")
        else:
            self.history_data[name] = value
        
    def set_parameters(self, setting: dict):
        for name in setting:
            if name in self.paramList:
                setattr(self, name, setting[name])
        
    def add_data(self, setting: dict):
        self.data_store.add_data_from_csv(setting)
        self.subscribe(setting["vtsymbol"],setting["interval"])
    
    def add_strategy(self, strategy):
        self.strategy = strategy(self, self.optimization)
        self.output(f"{strategy.__name__}成功添加")
        self.strategy.OnInit()
        self.strategy.vtsymbol = self.vtsymbol
        self.strategy.interval = self.interval
        self.output(f"{strategy.__name__}已经初始化")
        
    def strategy_output(self, msg):
        print(f"策略回测时间:{self.datetime}--INFO--{msg}")
    
    def get_vtsymbol(self):
        return self.data_store.vtsymbols
    
    def data_resample(self, vtsymbol: str, target_freq: Interval):
        
        result = self.data_store.resample(vtsymbol, target_freq)
        return result
    
    def update_daily_close(self, price: float):
        
        d = self.datetime.date()
        
        daily_result = self.daily_results.get(d, None)
        if daily_result:
            daily_result.close_price = price
        else:
            self.daily_results[d] = DailyResult(d, price)
            
    def calculate_result(self):
        self.output("开始计算每日盈亏")
        
        if not self.trades:
            self.output("成交记录为空，无法计算每日盈亏")
            return
        
        for trade in self.trades.values():
            d = trade.datetime.date()
            daily_result = self.daily_results[d]
            daily_result.add_trade(trade)
            
        pre_close = 0
        start_pos = 0
        
        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                    pre_close, start_pos, self.size, self.rate, self.slippage)
            
            pre_close = daily_result.close_price
            start_pos = daily_result.end_pos
            
        results = defaultdict(list)
        
        for daily_result in self.daily_results.values():
            for key, value in daily_result.__dict__.items():
                results[key].append(value)
        
        self.daily_df = DataFrame.from_dict(results).set_index("date")
        self.output("计算每日盈亏完成")
        return self.daily_df
    
    def __add__(self, *engine):
        """将若干个回测实例相加，实现多个资金曲线的叠加"""
        engine1 = BacktestingEngine()
        engine1.daily_df = DataFrame(columns = self.daily_df.columns)
        for i in engine:
            engine1.capital = self.capital + i.capital
            engine1.daily_df["net_pnl"] = self.daily_df["net_pnl"] + i.daily_df["net_pnl"]
            engine1.daily_df["commission"] = self.daily_df["commission"]+i.daily_df["commission"]
            engine1.daily_df["slippage"] = self.daily_df["slippage"]+i.daily_df["slippage"]
            engine1.daily_df["turnover"] = self.daily_df["turnover"]+i.daily_df["turnover"]
            engine1.daily_df["trade_count"] = self.daily_df["trade_count"]+i.daily_df["trade_count"]
        engine1.daily_df.dropna(axis=0,how="all",inplace=True)
        
        engine1.calculate_statistics()
        engine1.calculate_statistics()
        engine1.show_chart()
        
        return engine1
    
    def new_bar(self, bar):
        
        self.update_bar_var()
        self.cross_limit_order(bar)
        self.strategy.OnBar(bar)
        self.update_daily_close(bar.close)
        
    def update_bar_var(self):
        self.strategy.bar_count += 1
        self.strategy.last_pos = self.strategy.pos
    
    def cross_limit_order(self, bar):
        long_cross_price = self.bar.low
        short_cross_price = self.bar.high
        long_best_price = self.bar.open
        short_best_price = self.bar.open
        
        for order in list(self.active_limit_orders.values()):
            if order.status == Status.SUBMITTING:
                order.status == Status.NOTTRADED
                """委托提交回调"""
                self.strategy.OnOrder(order)
            
            long_cross = (
                    order.direction == Direction.LONG
                    and order.price >= long_cross_price
                    and long_cross_price>0)
            
            short_cross = (
                    order.direction == Direction.SHORT
                    and order.price <= short_cross_price
                    and short_cross_price > 0)
            
            if not long_cross and not short_cross:
                continue
            
            order.traded = order.volume
            order.status = Status.ALLTRADED
            """委托成交回调"""
            self.strategy.OnOrder(order)
            
            self.active_limit_orders.pop(order.orderid)
            
            if long_cross:
                trade_price = min(order.price, long_best_price)
                pos_change = order.volume
            else:
                trade_price = max(order.price, short_best_price)
                pos_change = -order.volume
                
            self.trade_count += 1
            
            trade = TradeData(
                    symbol=order.symbol,
                    exchange=order.exchange,
                    orderid=order.orderid,
                    tradeid=str(self.trade_count),
                    direction=order.direction,
                    offset=order.offset,
                    price=trade_price,
                    volume=order.volume,
                    time_=self.bar.datetime.strftime("%H:%M:%S"))
            
            trade.datetime = self.bar.datetime
            
            self.strategy.pos += pos_change
            
            self.strategy.OnTrade(trade)
            
            self.trades[trade.tradeid] = trade
        
        if self.strategy.pos != 0 and self.strategy.pos*self.strategy.last_pos >=0:
            """仅适用于不加减仓的情况"""
            if self.strategy.last_pos == 0: #如果当前bar刚刚开仓，则bars_since_entry为0
                self.strategy.bars_since_entry = 0
            else:                           #如果之前已经开仓，则bars_since_entry加1
                self.strategy.bars_since_entry += 1
        elif self.strategy.pos == 0:        #如果当前仓位为0，则bars_since_entry清零
            self.strategy.bars_since_entry = 0
            
        if self.strategy.last_pos == 0 and self.strategy.pos != 0:
            tradeid_list = sorted([int(i) for i in self.trades.keys()])
            self.strategy.last_entry_price = self.trades[str(tradeid_list[-1])].price
        
    def buy(self, price, volume):
        self.send_order(Direction.LONG, Offset.OPEN, price, volume, self.vtsymbol)
        
    def sell(self, price, volume):
        self.send_order(Direction.SHORT, Offset.CLOSE, price, volume, self.vtsymbol)
    
    def sell_short(self,  price, volume):
        self.send_order(Direction.SHORT, Offset.OPEN, price, volume,self.vtsymbol)
        
    def buy_to_cover(self, price, volume):
        self.send_order(Direction.LONG, Offset.CLOSE, price, volume, self.vtsymbol)
        
    def send_order(
            self,
            direction: Direction,
            offset: Offset,
            price: float,
            volume: float,
            vtsymbol: str
            ):
    
        self.limit_order_count += 1
        
        order = OrderData(direction = direction,
                         offset = offset,
                         orderid = str(self.limit_order_count),
                         datetime = self.bar.datetime,
                         status = Status.SUBMITTING,
                         price = price,
                         volume = volume,
                         vtsymbol = vtsymbol)
       
        self.active_limit_orders[order.orderid] = order
        self.limit_orders[order.orderid] = order
        
        return order.orderid
    
    def dec_start_end(self):
        """管理回测起止时间"""
        start0 = self.history_data.index[0] #历史数据起始时间点
        end0 = self.history_data.index[-1]  #历史数据结束时间点
        """设定的起止时间超出范围，那么根据数据起止时间设置回测起止时间"""
        if self.start<start0:
            self.start = start0
        if self.end>end0:
            self.end = end0
        
        """如果时间没有在历史数据的时间点中"""
        array = np.sort(np.array(list(self.history_data.index)))
        try:
            if self.start not in self.datetime_list:
                self.start = array[array>self.start][0]
            if self.end not in self.datetime_list:
                self.end = array[array<self.end][-1]
        except:
            pass
            
        self.output(f"时间戳已经调整完毕, start={self.start}, end={self.end}")
    
    def run_backtest(self):
        """运行回测"""
        
        """设置回测起止时间点"""
        self.dec_start_end()
        
        """开始回测"""
        backtest_data = self.history_data.loc[self.start:self.end]  #确定回测历史数据
        
        for i in backtest_data.itertuples():                        #开始遍历所有时间点
            
            """基本bar数据的处理"""
            bar = BarData(datetime = i.Index,
                          vtsymbol = self.vtsymbol,
                          interval = self.interval,
                          open = i.open,
                          high = i.high,
                          low = i.low,
                          close = i.close,
                          volume = i.volume,
                          openinterest = i.opeinterest)
            
            """附加因子数据的处理"""
            for factor in self.additional_factor:
                setattr(bar, factor, getattr(i, factor))
            
            """self.array的处理"""
            index_num = list(backtest_data.index).index(i.Index)
            self.strategy.array = backtest_data.iloc[index_num-self.data_size+1:index_num+1]
            
            """回测类的bar,datetime,策略类的bar,datetime的处理"""
            self.bar = bar
            self.datetime = bar.datetime
            self.strategy.bar = bar
            self.strategy.datetime = bar.datetime
            
            self.new_bar(bar)
            
        self.output("回测结束")
            
    def subscribe(self, vtsymbol, interval: Interval):
        """策略订阅行情数据，并初始化基本元数据"""
        if vtsymbol not in self.data_store.vtsymbol_data_map:
            raise ValueError(f"{vtsymbol} is not found")
        else:
            if not(interval in self.data_store.vtsymbol_interval_map[vtsymbol]):
                try:
                    self.data_store.vtsymbol_data_map[vtsymbol][interval] = self.data_store.resample(vtsymbol, interval)
                    if interval == Interval.Min_30:
                        pass
                    if interval == Interval.Hour_1:
                        pass
                    
                    if interval not in self.data_store.intervals:
                        self.data_store.intervals.append(interval)
                except:
                    raise ValueError("Resample is not successful")
        
        self.history_data = self.data_store.vtsymbol_data_map[vtsymbol][interval]
        self.vtsymbol = vtsymbol
        self.interval = interval
        self.symbol, self.exchange = extract_vt_symbol(self.vtsymbol)
        
        self.init_factor_data()
    
    def output(self, msg):
        print(f"{datetime.now()}\t{msg}")
        
    def calculate_statistics(self, df: DataFrame = None, output=True):
        
        self.output("开始计算策略统计指标")
        if df is None:
            df = self.daily_df
            
        if df is None:
            raise ValueError("无成交数据，无法计算统计指标")
        else:
            df["balance"] = df["net_pnl"].cumsum() + self.capital
            df["return"] = np.log(df["balance"]/df["balance"].shift(1)).fillna(0)
            df["highlevel"] = df["balance"].cummax()
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"]/df["highlevel"] * 100
            
            start_date = df.index[0]
            end_date = df.index[-1]
            
            total_days = len(df)
            profit_days = len(df[df["net_pnl"]>0])
            loss_days = len(df[df["net_pnl"]<0])
            
            end_balance = df["balance"].iloc[-1]
            max_drawdown = df["drawdown"].min()
            max_ddpercent = df["ddpercent"].min()
            
            total_net_pnl = df["net_pnl"].sum()
            daily_net_pnl = total_net_pnl/total_days
            
            total_commission = df["commission"].sum()
            daily_commission = total_commission/total_days
            
            total_slippage = df["slippage"].sum()
            daily_slippage = total_slippage/total_days
            
            total_turnover = df["turnover"].sum()
            daily_turnover = total_turnover/total_days
            
            total_trade_count = df["trade_count"].sum()
            daily_trade_count = total_trade_count/total_days
            
            total_return = (end_balance/self.capital-1)*100
            annual_return = total_return/total_days * 250
            daily_return = df["return"].mean()*100
            return_std = df["return"].std()*100

            if return_std:
                sharpe_ratio = daily_return/return_std*np.sqrt(240)
            else:
                sharpe_ratio = 0
                
            return_drawdown_ratio = -annual_return/max_ddpercent
            
        if output:
            self.output("_" *30)
            self.output(f"回测起始日期:\t{start_date}")
            self.output(f"回测结束日期:\t{end_date}")
            
            self.output(f"总交易日:\t{total_days}")
            self.output(f"盈利交易日：\t{profit_days}")
            self.output(f"亏损交易日：\t{loss_days}")

            self.output(f"起始资金：\t{self.capital:,.2f}")
            self.output(f"结束资金：\t{end_balance:,.2f}")

            self.output(f"总收益率：\t{total_return:,.2f}%")
            self.output(f"年化收益：\t{annual_return:,.2f}%")
            self.output(f"最大回撤: \t{max_drawdown:,.2f}")
            self.output(f"百分比最大回撤: {max_ddpercent:,.2f}%")

            self.output(f"总盈亏：\t{total_net_pnl:,.2f}")
            self.output(f"总手续费：\t{total_commission:,.2f}")
            self.output(f"总滑点：\t{total_slippage:,.2f}")
            self.output(f"总成交金额：\t{total_turnover:,.2f}")
            self.output(f"总成交笔数：\t{total_trade_count}")

            self.output(f"日均盈亏：\t{daily_net_pnl:,.2f}")
            self.output(f"日均手续费：\t{daily_commission:,.2f}")
            self.output(f"日均滑点：\t{daily_slippage:,.2f}")
            self.output(f"日均成交金额：\t{daily_turnover:,.2f}")
            self.output(f"日均成交笔数：\t{daily_trade_count}")

            self.output(f"日均收益率：\t{daily_return:,.2f}%")
            self.output(f"收益标准差：\t{return_std:,.2f}%")
            self.output(f"Sharpe_Ratio：\t{sharpe_ratio:,.2f}")
            self.output(f"收益回撤比：\t{return_drawdown_ratio:,.2f}")

        statistics = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
        }
        self.statistics = statistics
        return statistics
    
    def show_chart(self, df: DataFrame = None):
        """"""
        # Check DataFrame input exterior        
        if df is None:
            df = self.daily_df

        # Check for init DataFrame        
        if df is None:
            return

        plt.figure(figsize=(10, 16))

        balance_plot = plt.subplot(4, 1, 1)
        balance_plot.set_title("Balance")
        df["balance"].plot(legend=True)

        drawdown_plot = plt.subplot(4, 1, 2)
        drawdown_plot.set_title("Drawdown")
        drawdown_plot.fill_between(range(len(df)), df["drawdown"].values)

        pnl_plot = plt.subplot(4, 1, 3)
        pnl_plot.set_title("Daily Pnl")
        df["net_pnl"].plot(kind="bar", legend=False, grid=False, xticks=[])

        distribution_plot = plt.subplot(4, 1, 4)
        distribution_plot.set_title("Daily Pnl Distribution")
        df["net_pnl"].hist(bins=50)

        plt.show()

class DailyResult:
    
    def __init__(self, date: date, close_price: float):
        self.date = date
        self.close_price = close_price
        self.pre_close = 0
        
        self.trades = []
        self.trade_count = 0
        
        self.start_pos = 0
        self.end_pos = 0
        
        self.turnover = 0
        self.commission = 0
        self.slippage = 0
        
        self.trading_pnl = 0
        self.holding_pnl = 0
        self.total_pnl = 0
        self.net_pnl = 0
        
    def add_trade(self, trade: TradeData):
        """"""
        self.trades.append(trade)
        
    def calculate_pnl(self,
                      pre_close: float,
                      start_pos: float,
                      size: int,
                      rate: float,
                      slippage: float):
        
        """"""
        self.pre_close = pre_close
        
        self.start_pos = start_pos
        self.end_pos = start_pos
        """假设仓位没有变的持仓损益"""
        self.holding_pnl = self.start_pos*(self.close_price-
                                           self.pre_close)*size
        
        self.trade_count = len(self.trades)
        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change = trade.volume
            else:
                pos_change = -trade.volume
            
            turnover = trade.price*trade.volume*size
            """因为仓位变化带来的持仓损益"""
            self.trading_pnl += pos_change * (self.close_price - trade.price) *size
            
            self.end_pos += pos_change
            self.turnover += turnover
            self.commission += turnover * rate
            self.slippage += trade.volume*size*slippage
            
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage