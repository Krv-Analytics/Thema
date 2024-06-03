# File: tests/multiverse/system/inner/test_moon.py
# Lasted Updated: 06-02-24
# Updated By: JW

import os
import pytest
import pickle
import random
import tempfile
import numpy as np
from pandas.testing import assert_frame_equal
from thema.multiverse.system.inner.moon import Moon


class Test_Moon:
    """
    Testing class for Moon
    """

    def test_init_empty(self):
        with pytest.raises(TypeError):
            Moon()

    def test_init_defaults(self, temp_dataFile_0, test_data_0):
        x = Moon(data=temp_dataFile_0.name)
        assert_frame_equal(x.data, test_data_0)

    def test_fit_pkl(self, temp_dataFile_0, test_data_0):
        x = Moon(
            data=temp_dataFile_0.name,
            imputeColumns=["A", "B"],
            imputeMethods=["drop", "drop"],
        )
        assert_frame_equal(x.data, test_data_0)
        x.fit()
        assert len(x.imputeData.select_dtypes(include="object").columns) == 0
        assert x.imputeData.isna().sum().sum() == 0

    def test_fit_csv(self, temp_csv_dataFile, test_data_0):
        x = Moon(data=temp_csv_dataFile.name)
        assert_frame_equal(x.data, test_data_0)
        x.fit()
        assert len(x.imputeData.select_dtypes(include="object").columns) == 0
        assert x.imputeData.isna().sum().sum() == 0

    def test_fit_xlsx(self, temp_excel_dataFile, test_data_0):
        x = Moon(data=temp_excel_dataFile.name)
        assert_frame_equal(x.data, test_data_0)
        x.fit()
        assert len(x.imputeData.select_dtypes(include="object").columns) == 0
        assert x.imputeData.isna().sum().sum() == 0

    def test_dropColumns(self, temp_dataFile_0):
        x = Moon(data=temp_dataFile_0.name, dropColumns=["A"])
        x.fit()
        assert "A" not in x.imputeData.columns

    def test_scaler(self, temp_dataFile_4):
        x = Moon(
            data=temp_dataFile_4.name,
            dropColumns=[
                "Cat1",
                "Cat2",
                "Cat3",
                "Cat4",
                "Cat5",
            ],
            scaler=None,
        )
        assert x.scaler == None
        x.fit()
        for col in x.imputeData.columns:
            assert x.imputeData[col].std() != 1

        x.scaler = "standard"
        x.fit()
        assert x.scaler == "standard"

        # Approximate Unit Variance
        for col in x.imputeData.columns:
            assert np.round(x.imputeData[col].var(), 1) == 1.0

    def test_save(self, temp_dataFile_0):
        x = Moon(data=temp_dataFile_0.name)
        x.fit()

        with tempfile.NamedTemporaryFile() as temp_file:
            x.save(temp_file.name)

            assert os.path.exists(temp_file.name)

            with open(temp_file.name, "rb") as f:
                y = pickle.load(f)
        temp_file.close()
        assert_frame_equal(x.data, y.data)
        assert x.encoding == y.encoding
        assert x.dropColumns == y.dropColumns
        assert x.scaler == y.scaler
        assert x.imputeColumns == y.imputeColumns
        assert x.imputeMethods == y.imputeMethods

    def test_determinism(self, temp_dataFile_0):
        seed = random.randint(1, 100)
        x = Moon(
            data=temp_dataFile_0.name,
            encoding="one_hot",
            scaler="standard",
            imputeColumns=["B"],
            imputeMethods=["sampleNormal"],
            seed=seed,
        )
        y = Moon(
            data=temp_dataFile_0.name,
            encoding="one_hot",
            scaler="standard",
            imputeColumns=["B"],
            imputeMethods=["sampleNormal"],
            seed=seed,
        )
        x.fit()
        y.fit()

        assert_frame_equal(x.imputeData, y.imputeData)
