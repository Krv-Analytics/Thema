# File: tests/test_iSpace.py 
# Lasted Updated: 03-11-24 
# Updated By: SW 

import os 
import pytest
import pandas as pd 
from ...src import cleaning as c 


class test_iSpace:
    """
    Test class for iSpace
    """

    def test_init_0(self):
        data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, None],
        'C': ['a', 'b', 'c']
    })
        data=None, scaler="standard_scaler", encoding="one_hot", drop_columns=[], impute_methods=None, impute_columns=None, num_samples=1,verbose=True
