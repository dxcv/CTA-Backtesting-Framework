# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:37:47 2019

@author: Administrator
"""

from enum import Enum
class Interval(Enum):
    """数据频率枚举类"""
    Min_1 = "1Min"
    Min_5 = "5Min"
    Min_15 = "15Min"
    Min_30 = "30Min"
    Hour_1 = "1Hour"
    Day_1 = "1Day"
    Week_1 = "1Week"
    Month_1 = "1Month"

class Exchange(Enum):
    """交易所枚举类"""
    NULL = ""
    CFFEX = "CFFEX" 
    SHFE = "SHFE"
    CZCE = "CZCE"
    DCE = "DCE"
    INE = "INE"
    SSE = "SSE"
    SZSE = "SZSE"
    SGE = "SGE"
    
class Direction(Enum):
    """订单的方向"""
    LONG = "多"
    SHORT = "空"
    
class Offset(Enum):
    
    NONE = ""
    OPEN = "开"
    CLOSE = "平"
    CLOSETODAY = "平今"
    CLOSEYESTERDAY = "平昨"
    
class Status(Enum):
    SUBMITTING = "提交中"
    NOTTRADED = "未成交"
    PARTTRADED = "部分成交"
    ALLTRADED = "全部成交"
    CANCELLED = "已撤销"
    REJECTED = "拒单"