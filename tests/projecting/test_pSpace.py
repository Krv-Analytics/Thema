# File: tests/test_iSpace.py 
# Lasted Updated: 03-13-24 
# Updated By: SW 

import tempfile
from tests import test_utils as ut
from thema.cleaning.iSpace import iSpace
from thema.projecting.pSpace import pSpace

class Test_pSpace: 
    """Testing class for pSpace"""

    def test_init_defaults(self): 
        temp_file = ut.create_temp_data_file(ut.test_data_0)
        temp_file_path = ut.create_temp_data_file(ut.test_data_0, "pkl")
        test_iSpace = iSpace(data=temp_file_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            test_iSpace.fit_space(temp_dir, 10)
            temp_yaml = ut.create_temp_yaml(data=temp_file, out_dir=temp_dir)
            x = pSpace(temp_yaml)