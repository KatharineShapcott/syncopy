# -*- coding: utf-8 -*-
# 
# Syncopy container class for statistical data
# 
# Created: 2019-07-25 13:57:57
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-25 14:27:10>

# Builtin/3rd party package imports
from abc import ABC

# Local imports
from .base_data import BaseData

class ConnectivityData(BaseData, ABC):
    pass