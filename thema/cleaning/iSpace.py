# File: src/cleaning/iSpace.py 
# Last Update: 03-12-24
# Updated by: SW 

import os
import pickle
import pandas as pd
import numpy as np 

from omegaconf import OmegaConf, ListConfig
from sklearn.preprocessing import StandardScaler
import category_encoders as ce 

from . import cleaning_utils
from .cleaning_utils import clean_data_filename, integer_encoder
from .cleaning_utils import add_imputed_flags
from ..utils import function_scheduler

class iSpace: 
    """
    Clean Data Generator 
    TODO: Update Doc String 
    """

    def __init__(self, data=None, scaler:str="standard", encoding:str="one_hot", dropColumns=None, imputeMethods=None, imputeColumns=None, num_samples:int=1, verbose: bool=True, YAML_PATH=None): 
        """
        TODO: Update Doc String
        """
        
        if YAML_PATH is None and data is None:
            raise ValueError("Please provide config parameters or a path to a yaml configuration file.")
        
        self.verbose = verbose
        self.YAML_PATH = None 
        if YAML_PATH is not None: 
            assert os.path.isfile(YAML_PATH), "params.yaml file not found!"
            try: 
                self.YAML_PATH = YAML_PATH
                with open(YAML_PATH, "r") as f:
                    params = OmegaConf.load(f)
                data = params.data
                scaler = params.cleaning.scaler 
                encoding = params.cleaning.encoding 
                dropColumns = params.cleaning.dropColumns
                imputeColumns = params.cleaning.imputeColumns
                imputeMethods = params.cleaning.imputeMethods
                num_samples = params.cleaning.num_samples 

            except Exception as e: 
                print(e)
        
        if type(data) == str:
            try:
                if data.endswith('.csv') or data.endswith('.xlsx'):
                    assert os.path.isfile(data), "\n Invalid path to Data"
                    if data.endswith('.csv'):
                        self.data = pd.read_csv(data)
                    elif data.endswith('.xlsx'):
                        self.data = pd.read_excel(data)
                    self.data_path = data
                elif data.endswith('.pkl'):
                    assert os.path.isfile(data), "\n Invalid path to Data"
                    self.data = pd.read_pickle(data)
                    self.data_path = data
                else:
                    raise ValueError("Unsupported file format. Supported formats: .csv, .xlsx, .pkl")
            except:
                raise ValueError("There was an issue opening your data file. Please make sure it exists and is a supported file format.")
        
        elif isinstance(data, pd.DataFrame):
            self.data = data 
            self.data_path = None 
        else: 
            raise ValueError("'data' must be a pd.DataFrame object, OR a path to a csv, xlxs, or pickle file")

        # HARD CODED SUPPORTED TYPED 
        supported_imputeMethods = ["random_sampling", "drop", "mean", "median", "mode"]
        
        self.scaler = scaler 
        self.encoding = encoding 
        self.num_samples = num_samples
        
        assert self.scaler in ["standard"]
        assert self.encoding in ["one_hot", "integer", "hash"], "Only one_hot, integer, and hash encoding are supported."
        
        
        if dropColumns is None:
            self.dropColumns = []
        else:
            assert type(dropColumns) == list, "dropColumns must be a list" 
            self.dropColumns = dropColumns

        assert num_samples > 0, "Please Specify a postive number of Samples"


        if imputeColumns is None or imputeColumns=="None": 
            self.imputeColumns = []
        elif imputeColumns == "all":
            self.imputeColumns = self.data.isna().any()[self.data.isna().any()].index.tolist()
        elif type(imputeColumns) == ListConfig or type(imputeColumns) == list:
            self.imputeColumns = imputeColumns
            for c in imputeColumns:
                if c not in self.data.columns:
                    print("Invalid impute column. Defaulting to 'None'") 
                    self.imputeColumns =[] 
        else: 
            self.imputeColumns=[]

        if imputeMethods is None or imputeMethods == "None": 
            self.imputeMethods = ["drop" for _ in range(len(self.imputeColumns))] 
        
        elif type(imputeMethods) == str:
            if not imputeMethods in supported_imputeMethods: 
                print("Invalid impute methods. Defaulting to 'drop'")
                imputeMethods = "drop" 
            self.imputeMethods = [imputeMethods for _ in range(len(self.imputeColumns))]
            self.num_samples = 1 
        else: 
            assert len(imputeMethods) == len(imputeColumns), "Lengh of imputeMethods must match length of imputeColumns"
            for index, method in enumerate(imputeMethods):
                if not method in supported_imputeMethods: 
                    print("Invalid impute methods. Defaulting to 'drop'")
                    imputeMethods[index] = "drop"
            self.imputeMethods = imputeMethods
        
        self.imputed_data = None 
        
        
    def get_missingData_summary(self):
        """
        Returns a dictionary containing a breakdown of columns from 'data' that are:
        - 'numericMissing': Numeric columns with missing values
        - 'numericComplete': Numeric columns without missing values
        - 'categoricalMissing': Categorical columns with missing values
        - 'categoricalComplete': Categorical columns without missing values
        """

        numeric_missing = []
        numeric_not_missing = []
        categorical_missing = []
        categorical_complete = []

        for column in self.data.columns:
            if self.data[column].dtype.kind in 'biufc':
                if self.data[column].isna().any():
                    numeric_missing.append(column)
                else:
                    numeric_not_missing.append(column)
            else:
                if self.data[column].isna().any():
                    categorical_missing.append(column)
                else:
                    categorical_complete.append(column)

        summary = {
            'numericMissing': numeric_missing,
            'numericComplete': numeric_not_missing,
            'categoricalMissing': categorical_missing,
            'categoricalComplete': categorical_complete
        }

        return summary
    

    def get_dropped_ratio(self, axis=0):
        """
        Calculates the ratio of the number of rows after fit over the total number of rows in 'self.data'.
    
        Returns:
        float: The ratio of rows after NaNs are dropped to the total number of rows.
        """
        if self.imputed_data is None: 
            print("Make sure to run 'fit' before looking at the dropped ratio")
            return 0 
        return self.imputed_data.shape[axis]/self.data[axis]

    
    
    def get_na_as_list(self):
        return self.data.columns[self.data.isna().any()].tolist()
    


    def get_reccomended_sampling_method(self):
        methods = []
        for column in self.data.columns[self.data.isna().any()].tolist():
            if pd.api.types.is_numeric_dtype(self.data[column]):
                methods.append("random_sampling")
            else: 
                methods.append("mode")
        
        return methods 

    

    def fit_space(self, out_dir, numSamples=None): 
        """
        """
        if(numSamples):
            self.num_samples = numSamples
        subprocesses = []
        for i in range(self.num_samples):
            cmd = (self.fit_space_helper, 
                   out_dir,
                   i)
            subprocesses.append(cmd)
            # Handles Process scheduling
        function_scheduler(
                subprocesses, max_workers=min(4, self.num_samples), out_message="SUCCESS: Imputation(s)", resilient=True
                )


    def fit_space_helper(self, out_dir, id):
        """
        """
        self.fit()
        self.dump(out_dir=out_dir, id=id)   


    def fit(self):
        """
        """
        if not self.dropColumns == [] and all(column in self.data.columns for column in self.dropColumns): 
            self.imputed_data = self.data.drop(columns=self.dropColumns)    
        else:
            self.imputed_data = self.data

        self.imputed_data = add_imputed_flags(self.data, self.imputeColumns)
        for index, column in enumerate(self.imputeColumns):
            impute_function = getattr(cleaning_utils, self.imputeMethods[index])
            self.imputed_data[column] = impute_function(self.data[column])

        # Drops unaccounted columns
        self.imputed_data.dropna(axis=1, inplace=True)

        # Encoding 
        assert self.encoding in ["one_hot", "integer", "hash"], "Only one_hot, integer, and hash encoding are supported."

        if self.encoding == "one_hot":
            non_numeric_columns = self.imputed_data.select_dtypes(exclude=["number"]).columns
            for column in non_numeric_columns:
                self.imputed_data = pd.get_dummies(
                    self.imputed_data, prefix=f"OH_{column}", columns=[column]
                )

        if self.encoding == "integer":
            categorical_variables = self.imputed_data.select_dtypes(
                exclude=["number"]
            ).columns.tolist()
            # Rewrite Columns
            for column in categorical_variables:
                vals = self.imputed_data[column].values
                self.imputed_dataed[column] = integer_encoder(vals)

        if self.encoding == "hash":
            categorical_variables = self.imputed_data.select_dtypes(
                exclude=["number"]
            ).columns.tolist()
            hashing_encoder = ce.HashingEncoder(cols=categorical_variables, n_components=10)
            self.imputed_data = hashing_encoder.fit_transform(self.imputed_data)

        # Scaling 
        assert self.scaler in ["standard"], "Invalid Scaler"
        if self.scaler == "standard":
            scaler = StandardScaler()
            self.imputed_data = pd.DataFrame(
                scaler.fit_transform(self.imputed_data), columns=list(self.imputed_data.columns)
            )

    
    def dump(self, out_dir, id=None): 
        """
        
        """
        if id is None:
            id = np.random.randint(0, 100)
        try:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            if self.data_path is None: 
                data_name = "my_dataset"
            else: 
                filename_without_extension, extension = os.path.splitext(self.data_path)
                data_name = filename_without_extension.split('/')[-1]
            file_name = clean_data_filename(data_name=data_name, id=id, scaler=self.scaler, encoding=self.encoding)
            output_filepath = os.path.join(out_dir,file_name)

            results = {"data": self.imputed_data, 
                        "description": {"raw": self.data_path, 
                                        "scaler": self.scaler,
                                        "encoding": self.encoding,
                                        "dropColumns": self.dropColumns,
                                        "imputeColumns": self.imputeColumns,
                                        "imputeMethods": self.imputeMethods}
                            }
            with open(output_filepath, "wb") as f:
                pickle.dump(results, f)
        except Exception as e:
            print(e)


    def save(self, file_path): 
        """
        Save the current object instance to a file using pickle serialization.

        Parameters:
            file_path (str): The path to the file where the object will be saved.

        """
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            print(e)