# File:tests/multiverse/system/inner/conftest.py
# Last Updated: 06/02/24
# Updated By: JW

import os
import yaml
import pytest
import shutil
import tempfile
import pandas as pd

from tests import test_utils as ut


@pytest.fixture
def temp_planet_dir():
    with tempfile.TemporaryDirectory() as temp_planet_dir:
        yield temp_planet_dir
    if os.path.isdir(temp_planet_dir):
        shutil.rmtree(temp_planet_dir)


@pytest.fixture
def temp_planet_dir_1():
    with tempfile.TemporaryDirectory() as temp_planet_dir_1:
        yield temp_planet_dir_1
    if os.path.isdir(temp_planet_dir_1):
        shutil.rmtree(temp_planet_dir_1)


@pytest.fixture
def temp_yaml_0():
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_dataFile:
        ut._test_data_0.to_pickle(tmp_dataFile.name)
        data = tmp_dataFile.name
        runName = "test"
        with tempfile.TemporaryDirectory() as tmpdir:
            outDir = tmpdir
            cleaning = {
                "scaler": "standard",
                "encoding": "one_hot",
                "numSamples": 3,
                "seeds": [42, 41, 40],
                "dropColumns": None,
                "imputeColumns": ["B"],
                "imputeMethods": ["sampleNormal"],
            }
            parameters = {
                "runName": runName,
                "data": data,
                "outDir": outDir,
                "Planet": cleaning,
            }

            with tempfile.NamedTemporaryFile(
                suffix=".yaml",
                mode="w",
            ) as yaml_temp_file:
                yaml.dump(parameters, yaml_temp_file, default_flow_style=False)
                yield yaml_temp_file
            yaml_temp_file.close()
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)
    tmp_dataFile.close()


@pytest.fixture
def temp_csv_yaml():
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp_dataFile:
        ut._test_data_0.to_csv(tmp_dataFile.name, index=False)
        data = tmp_dataFile.name
        runName = "test"
        with tempfile.TemporaryDirectory() as tmpdir:
            outDir = tmpdir
            cleaning = {
                "scaler": "standard",
                "encoding": "one_hot",
                "numSamples": 3,
                "seeds": [42, 41, 40],
                "dropColumns": None,
                "imputeColumns": ["B"],
                "imputeMethods": ["sampleNormal"],
            }
            parameters = {
                "runName": runName,
                "data": data,
                "outDir": outDir,
                "Planet": cleaning,
            }

            with tempfile.NamedTemporaryFile(
                suffix=".yaml",
                mode="w",
            ) as yaml_temp_file:
                yaml.dump(parameters, yaml_temp_file, default_flow_style=False)
                yield yaml_temp_file
            yaml_temp_file.close()
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)
    tmp_dataFile.close()


@pytest.fixture
def temp_xlsx_yaml():
    with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp_dataFile:
        ut._test_data_0.to_excel(tmp_dataFile.name, index=False)
        data = tmp_dataFile.name
        runName = "test"
        with tempfile.TemporaryDirectory() as tmpdir:
            outDir = tmpdir
            cleaning = {
                "scaler": "standard",
                "encoding": "one_hot",
                "numSamples": 3,
                "seeds": [42, 41, 40],
                "dropColumns": None,
                "imputeColumns": ["B"],
                "imputeMethods": ["sampleNormal"],
            }
            parameters = {
                "runName": runName,
                "data": data,
                "outDir": outDir,
                "Planet": cleaning,
            }

            with tempfile.NamedTemporaryFile(
                suffix=".yaml",
                mode="w",
            ) as yaml_temp_file:
                yaml.dump(parameters, yaml_temp_file, default_flow_style=False)
                yield yaml_temp_file
            yaml_temp_file.close()
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)
    tmp_dataFile.close()


@pytest.fixture
def temp_dataFile_0():
    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp_file:
        ut._test_data_0.to_pickle(temp_file.name)
        yield temp_file
    temp_file.close()


@pytest.fixture
def temp_csv_dataFile():
    with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
        ut._test_data_0.to_csv(temp_file.name, index=False)
        yield temp_file
    temp_file.close()


@pytest.fixture
def temp_excel_dataFile():
    with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
        ut._test_data_0.to_csv(temp_file.name, index=False)
        yield temp_file
    temp_file.close()


@pytest.fixture
def temp_dataFile_1():
    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp_file:
        ut._test_data_1.to_pickle(temp_file.name)
        yield temp_file
    temp_file.close()


@pytest.fixture
def temp_dataFile_2():
    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp_file:
        ut._test_data_2.to_pickle(temp_file.name)
        yield temp_file
    temp_file.close()


@pytest.fixture
def temp_dataFile_3():
    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp_file:
        ut._test_data_3.to_pickle(temp_file.name)
        yield temp_file
    temp_file.close()


@pytest.fixture
def temp_dataFile_4():
    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp_file:
        ut._test_data_4.to_pickle(temp_file.name)
        yield temp_file
    temp_file.close()


@pytest.fixture
def test_data_0_missingData_summary():
    return ut._test_data_0_missingData_summary


@pytest.fixture
def test_data_1_missingData_summary():
    return ut._test_data_1_missingData_summary


@pytest.fixture
def test_data_2_missingData_summary():
    return ut._test_data_2_missingData_summary


@pytest.fixture
def test_data_0():
    return ut._test_data_0


@pytest.fixture
def test_data_1():
    return ut._test_data_1


@pytest.fixture
def test_data_2():
    return ut._test_data_2


@pytest.fixture
def test_data_3():
    return ut._test_data_3
