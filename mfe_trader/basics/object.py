# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:39:48 2019

@author: Administrator
"""

from datetime import datetime
from dataclasses import dataclass

from .constant import (Exchange,
                       Interval,
                       Offset,Direction,
                       Status
                       )
from .functions import extract_vt_symbol
@dataclass
class BarData:
    datetime: datetime
    exchange: Exchange = Exchange.NULL
    vtsymbol: str = ""
    symbol: str = ""
    interval: Interval = ""
    open: float= 0
    high: float = 0
    low: float = 0
    close: float = 0
    volume: float = 0
    openinterest: float = 0
    
    
    def __post_init__(self):
        if self.symbol and self.exchange!=Exchange.NULL and not self.vtsymbol:
            self.vtsymbol = f"{self.symbol}.{self.exchange}"
        elif self.vtsymbol and self.exchange!=Exchange.NULL and not self.symbol:
            self.symbol = extract_vt_symbol(self.vt_symbol)[0]
        elif self.vtsymbol and self.exchange==Exchange.NULL and not self.symbol:
            self.symbol = extract_vt_symbol(self.vtsymbol)[0]
            self.exchange = extract_vt_symbol(self.vtsymbol)[1]
        elif self.vtsymbol and self.symbol and self.exchange==Exchange.NULL:
            self.exchange = extract_vt_symbol(self.vtsymbol)[1]
        elif self.symbol and self.exchange!=Exchange.NULL and self.vtsymbol:
            pass
        else:
            raise ValueError("symbol and exchange is not enough")
            
@dataclass
class OrderData:
    
    datetime: datetime
    direction: Direction
    offset: Offset
    status: Status
    orderid: str
    price: float 
    volume: float
    exchange: Exchange = Exchange.NULL
    traded: float = 0.0
    vtsymbol: str = ""
    symbol: str = ""
    
    def __post_init__(self):
        if self.symbol and self.exchange!=Exchange.NULL and not self.vtsymbol:
            self.vtsymbol = f"{self.symbol}.{self.exchange}"
        elif self.vtsymbol and self.exchange!=Exchange.NULL and not self.symbol:
            self.symbol = extract_vt_symbol(self.vtsymbol)[0]
        elif self.vtsymbol and self.exchange==Exchange.NULL and not self.symbol:
            self.symbol = extract_vt_symbol(self.vtsymbol)[0]
            self.exchange = extract_vt_symbol(self.vtsymbol)[1]
        elif self.vtsymbol and self.symbol and self.exchange==Exchange.NULL:
            self.exchange = extract_vt_symbol(self.vtsymbol)[1]
        elif self.symbol and self.exchange!=Exchange.NULL and self.vtsymbol:
            pass
        else:
            raise ValueError("symbol and exchange is not enough")
            
@dataclass
class TradeData:
    
    orderid: str
    tradeid: str
    symbol: str = ""
    vtsymbol: str = ""
    exchange: Exchange = Exchange.NULL
    direction: Direction = ""
    offset: Offset = Offset.NONE
    price: float = 0
    volume: float = 0
    time_: str = ""
    
    def __post_init__(self):
        if self.symbol and self.exchange!=Exchange.NULL and not self.vtsymbol:
            self.vtsymbol = f"{self.symbol}.{self.exchange}"
        elif self.vtsymbol and self.exchange!=Exchange.NULL and not self.symbol:
            self.symbol = extract_vt_symbol(self.vtsymbol)[0]
        elif self.vtsymbol and self.exchange==Exchange.NULL and not self.symbol:
            self.symbol = extract_vt_symbol(self.vtsymbol)[0]
            self.exchange = extract_vt_symbol(self.vtsymbol)[1]
        elif self.vtsymbol and self.symbol and self.exchange==Exchange.NULL:
            self.exchange = extract_vt_symbol(self.vtsymbol)[1]
        elif self.symbol and self.exchange!=Exchange.NULL and self.vtsymbol:
            pass
        else:
            raise ValueError("symbol and exchange is not enough")