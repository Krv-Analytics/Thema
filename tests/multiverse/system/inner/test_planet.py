# File: tests/multiverse/system/inner/test_planet.py
# Lasted Updated: 04-07-24
# Updated By: SHI & SW

import os
import pytest
import pickle
import random
import tempfile
import numpy as np
from thema.multiverse import Planet
from pandas.testing import assert_frame_equal


class Test_Planet:
    """
    A PyTest class for Planet
    """

    def test_init_empty(self):
        with pytest.raises(ValueError):
            Planet()

    def test_init_missingYamlFile(self):
        with pytest.raises(AssertionError):
            Planet(YAML_PATH="../a/very/junk/file")

    def test_init_incorrectYamlFile(self, temp_dataFile_0):
        with pytest.raises(ValueError):
            Planet(YAML_PATH=temp_dataFile_0.name)

    def test_init_defaults(self, temp_planet_dir, temp_dataFile_0, test_data_0):
        my_Planet = Planet(data=temp_dataFile_0.name, outDir=temp_planet_dir)

        assert_frame_equal(my_Planet.data, test_data_0)
        assert my_Planet.scaler == "standard"
        assert my_Planet.encoding == "one_hot"
        assert my_Planet.dropColumns == []
        assert my_Planet.get_data_path() == temp_dataFile_0.name
        assert my_Planet.imputeColumns == []
        assert my_Planet.imputeMethods == []

    def test_init_dataPath_pkl(self, temp_planet_dir, temp_dataFile_0, test_data_0):
        my_Planet = Planet(data=temp_dataFile_0.name, outDir=temp_planet_dir)

        assert_frame_equal(my_Planet.data, test_data_0)
        assert my_Planet.scaler == "standard"
        assert my_Planet.encoding == "one_hot"
        assert my_Planet.dropColumns == []
        assert my_Planet.get_data_path() == temp_dataFile_0.name
        assert my_Planet.imputeColumns == []
        assert my_Planet.imputeMethods == []

    def test_init_yaml_dataPath_pkl(self, temp_yaml_0, test_data_0):
        my_Planet = Planet(YAML_PATH=temp_yaml_0.name)
        assert_frame_equal(my_Planet.data, test_data_0)
        assert my_Planet.scaler == "standard"
        assert my_Planet.encoding == "one_hot"
        assert my_Planet.dropColumns == []
        assert my_Planet.imputeColumns == ["B"]
        assert my_Planet.imputeMethods == ["sampleNormal"]

    def test_init_dataPath_csv(self, temp_planet_dir, temp_csv_dataFile, test_data_0):
        my_Planet = Planet(data=temp_csv_dataFile.name, outDir=temp_planet_dir)
        assert_frame_equal(my_Planet.data, test_data_0)
        assert my_Planet.scaler == "standard"
        assert my_Planet.encoding == "one_hot"
        assert my_Planet.dropColumns == []
        assert my_Planet.get_data_path() == temp_csv_dataFile.name
        assert my_Planet.imputeColumns == []
        assert my_Planet.imputeMethods == []

    def test_init_yaml_dataPath_csv(self, temp_csv_yaml, test_data_0):
        my_Planet = Planet(YAML_PATH=temp_csv_yaml.name)
        assert_frame_equal(my_Planet.data, test_data_0)
        assert my_Planet.scaler == "standard"
        assert my_Planet.encoding == "one_hot"
        assert my_Planet.dropColumns == []
        assert my_Planet.imputeColumns == ["B"]
        assert my_Planet.imputeMethods == ["sampleNormal"]

    def test_init_dataPath_xlsx(
        self, temp_planet_dir, temp_excel_dataFile, test_data_0
    ):
        my_Planet = Planet(data=temp_excel_dataFile.name, outDir=temp_planet_dir)

        assert_frame_equal(my_Planet.data, test_data_0)
        assert my_Planet.scaler == "standard"
        assert my_Planet.encoding == "one_hot"
        assert my_Planet.dropColumns == []
        assert my_Planet.get_data_path() == temp_excel_dataFile.name
        assert my_Planet.imputeColumns == []
        assert my_Planet.imputeMethods == []

    def test_init_yaml_dataPath_xlsx(self, temp_xlsx_yaml, test_data_0):
        my_Planet = Planet(YAML_PATH=temp_xlsx_yaml.name)
        assert_frame_equal(my_Planet.data, test_data_0)
        assert my_Planet.scaler == "standard"
        assert my_Planet.encoding == "one_hot"
        assert my_Planet.dropColumns == []
        assert my_Planet.imputeColumns == ["B"]
        assert my_Planet.imputeMethods == ["sampleNormal"]

    def test_init_scaler(self, temp_planet_dir, temp_dataFile_0):
        with pytest.raises(AssertionError):
            Planet(
                data=temp_dataFile_0.name,
                outDir=temp_planet_dir,
                scaler="a good one",
            )

    def test_init_encoding(self, temp_planet_dir, temp_dataFile_0):

        x = Planet(
            data=temp_dataFile_0.name,
            outDir=temp_planet_dir,
            encoding="one_hot",
        )
        y = Planet(
            data=temp_dataFile_0.name,
            outDir=temp_planet_dir,
            encoding="integer",
        )
        z = Planet(data=temp_dataFile_0.name, outDir=temp_planet_dir, encoding="hash")

        assert x.encoding == "one_hot"
        assert y.encoding == "integer"
        assert z.encoding == "hash"

    def test_init_dropColumns(self, temp_planet_dir, temp_dataFile_0):
        with pytest.raises(AssertionError):
            Planet(
                data=temp_dataFile_0.name,
                outDir=temp_planet_dir,
                dropColumns="all",
            )

        my_Planet2 = Planet(
            data=temp_dataFile_0.name, outDir=temp_planet_dir, dropColumns=["A"]
        )
        assert my_Planet2.dropColumns == ["A"]

    def test_init_imputeColumns(self, temp_planet_dir, temp_dataFile_0):
        w = Planet(
            data=temp_dataFile_0.name,
            outDir=temp_planet_dir,
            imputeColumns="ABCs",
        )
        x = Planet(
            data=temp_dataFile_0.name,
            outDir=temp_planet_dir,
            imputeColumns="auto",
        )
        y = Planet(
            data=temp_dataFile_0.name,
            outDir=temp_planet_dir,
            imputeColumns="None",
        )
        z = Planet(
            data=temp_dataFile_0.name,
            outDir=temp_planet_dir,
            imputeColumns=["A"],
        )

        assert w.imputeColumns == []
        assert x.imputeColumns == ["B"]
        assert y.imputeColumns == []
        assert z.imputeColumns == ["A"]

    def test_init_imputeMethods(self, temp_planet_dir, temp_dataFile_0):
        with pytest.raises(AssertionError):
            Planet(
                data=temp_dataFile_0.name,
                outDir=temp_planet_dir,
                imputeColumns=["A", "B"],
                imputeMethods=["mean", "median", "drop"],
            )

        w = Planet(
            data=temp_dataFile_0.name,
            outDir=temp_planet_dir,
            imputeColumns="auto",
            imputeMethods="wrong",
        )
        x = Planet(
            data=temp_dataFile_0.name,
            outDir=temp_planet_dir,
            imputeColumns="auto",
            imputeMethods="sampleNormal",
        )
        y = Planet(
            data=temp_dataFile_0.name,
            outDir=temp_planet_dir,
            imputeColumns=["A", "B"],
            imputeMethods=None,
        )
        z = Planet(
            data=temp_dataFile_0.name,
            outDir=temp_planet_dir,
            imputeColumns=["A"],
            imputeMethods=["mean"],
        )

        assert w.imputeMethods == ["drop"]
        assert x.imputeMethods == ["sampleNormal"]
        assert y.imputeMethods == ["drop", "drop"]
        assert z.imputeMethods == ["mean"]

    def test_get_column_summary(
        self,
        temp_planet_dir,
        temp_dataFile_0,
        temp_dataFile_1,
        temp_dataFile_2,
        test_data_0_missingData_summary,
        test_data_1_missingData_summary,
        test_data_2_missingData_summary,
    ):
        x = Planet(data=temp_dataFile_0.name, outDir=temp_planet_dir)
        y = Planet(data=temp_dataFile_1.name, outDir=temp_planet_dir)
        z = Planet(data=temp_dataFile_2.name, outDir=temp_planet_dir)

        assert x.get_missingData_summary() == test_data_0_missingData_summary
        assert y.get_missingData_summary() == test_data_1_missingData_summary
        assert z.get_missingData_summary() == test_data_2_missingData_summary

    def test_get_na_as_list(
        self,
        temp_planet_dir,
        temp_dataFile_0,
        temp_dataFile_1,
        temp_dataFile_2,
    ):
        x = Planet(data=temp_dataFile_0.name, outDir=temp_planet_dir)
        y = Planet(data=temp_dataFile_1.name, outDir=temp_planet_dir)
        z = Planet(data=temp_dataFile_2.name, outDir=temp_planet_dir)

        assert x.get_na_as_list() == ["B"]
        assert y.get_na_as_list() == ["A", "B", "C", "D", "E", "F"]
        assert z.get_na_as_list() == ["Y", "Z", "V"]

    def test_get_recomended_sampling_method(
        self,
        temp_planet_dir,
        temp_dataFile_0,
        temp_dataFile_1,
        temp_dataFile_2,
    ):
        x = Planet(data=temp_dataFile_0.name, outDir=temp_planet_dir)
        y = Planet(data=temp_dataFile_1.name, outDir=temp_planet_dir)
        z = Planet(data=temp_dataFile_2.name, outDir=temp_planet_dir)

        assert x.get_recomended_sampling_method() == ["sampleNormal"]
        assert y.get_recomended_sampling_method() == [
            "sampleNormal",
            "sampleCategorical",
            "sampleCategorical",
            "sampleNormal",
            "sampleCategorical",
            "sampleCategorical",
        ]
        assert z.get_recomended_sampling_method() == [
            "sampleCategorical",
            "sampleNormal",
            "sampleCategorical",
        ]

    def test_save(self, temp_planet_dir, temp_dataFile_0):
        x = Planet(data=temp_dataFile_0.name, outDir=temp_planet_dir)
        x.fit()
        with tempfile.NamedTemporaryFile() as temp_file:
            x.save(temp_file.name)
            with open(temp_file.name, "rb") as f:
                y = pickle.load(f)
                assert_frame_equal(x.data, y.data)
                assert x.dropColumns == y.dropColumns
                assert x.outDir == y.outDir
        temp_file.close()

    def test_fit(self, temp_planet_dir, temp_dataFile_1):
        numSamples = np.random.randint(1, 51)
        seeds = [random.randint(0, 100) for _ in range(numSamples)]
        clear_temporary_directory(temp_planet_dir)

        x = Planet(
            data=temp_dataFile_1.name,
            outDir=temp_planet_dir,
            imputeColumns="auto",
            imputeMethods="auto",
            numSamples=numSamples,
            seeds=seeds,
            encoding="one_hot",
        )
        x.fit()
        assert x.numSamples == numSamples
        assert x.seeds == seeds
        assert len(os.listdir(temp_planet_dir)) == numSamples

    def test_determinism(self, temp_planet_dir, temp_planet_dir_1, temp_dataFile_3):
        numSamples = 1
        seeds = [random.randint(0, 100) for _ in range(numSamples)]
        print(seeds)
        clear_temporary_directory(temp_planet_dir)
        clear_temporary_directory(temp_planet_dir_1)

        x = Planet(
            data=temp_dataFile_3.name,
            outDir=temp_planet_dir,
            imputeColumns="auto",
            numSamples=numSamples,
            seeds=seeds,
        )

        x.imputeMethods = x.get_recomended_sampling_method()

        assert x.seeds == seeds
        x_models = []
        x.fit()
        files_in_directory = os.listdir(temp_planet_dir)
        absolute_paths = sorted(
            [os.path.join(temp_planet_dir, filename) for filename in files_in_directory]
        )
        for path in absolute_paths:
            with open(path, "rb") as f:
                x_models.append(pickle.load(f))

        y = Planet(
            data=temp_dataFile_3.name,
            outDir=temp_planet_dir_1,
            imputeColumns="auto",
            numSamples=numSamples,
            seeds=seeds,
        )

        y.imputeMethods = y.get_recomended_sampling_method()

        assert y.seeds == seeds
        y_models = []
        y.fit()
        files_in_directory = os.listdir(temp_planet_dir_1)
        absolute_paths = sorted(
            [
                os.path.join(temp_planet_dir_1, filename)
                for filename in files_in_directory
            ]
        )
        for path in absolute_paths:
            with open(path, "rb") as f:
                y_models.append(pickle.load(f))

        for index in range(len(y_models)):
            assert_frame_equal(x_models[index].imputeData, y_models[index].imputeData)


def clear_temporary_directory(tempDir):
    if os.listdir(tempDir):
        for filename in os.listdir(tempDir):
            file_path = os.path.join(tempDir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
        assert len(os.listdir(tempDir)) == 0
