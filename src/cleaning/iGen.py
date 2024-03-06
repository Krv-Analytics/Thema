import os
import pickle
import pandas as pd
import numpy as np 
from types import FunctionType as function

from .cleaning_utils import clean_data_filename, integer_encoder
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
import category_encoders as ce 

from . import cleaning_utils
from .cleaning_utils import add_imputed_flags

class iGen: 
    """
    Clean Data Generator 
    TODO: Update Doc String 
    """

    def __init__(self, data=None, scaler="standard_scaler", encoding="one_hot", drop_columns=[], impute_methods=None, impute_columns=None, verbose=True, YAML_PATH=None): 
        """
        TODO: Update Doc String
        """
        
        assert YAML_PATH is not None or data is not None, "Please provide config parameters or a path to a yaml configuration file."
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
                drop_columns = params.cleaning.drop_columns
                impute_columns = params.cleaning.impute_columns
                impute_methods = params.cleaning.impute_methods

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
                    assert os.path.isfile(data), "\n Invalid path to Clean Data"
                    self.data = pd.read_pickle(data)
                    self.data_path = data
                else:
                    raise ValueError("Unsupported file format. Supported formats: .csv, .xlsx, .pkl")
            except:
                raise ValueError("There was an issue opening your data file. Please make sure it exists and is a supported file format.")
        
        elif isinstance(data, pd.DataFrame):
            self.data = data 
            self.data_path = -1 
        else: 
            raise ValueError("'data' must be a pd.DataFrame object, OR a path to a csv, xlxs, or pickle file")

        # HARD CODED SUPPORTED TYPED 
        supported_impute_methods = ["random_sampling", "drop", "mean", "median", "mode"]
        
        self.scaler = scaler 
        self.encoding = encoding 
        self.drop_columns = drop_columns

        if impute_columns is None or impute_columns=="None": 
            self.impute_columns = self.data.isna().any().columns
        else: 
            self.impute_columns = impute_columns

        if impute_methods is None or impute_methods == "None": 
            self.impute_methods = ["drop" for _ in range(len(impute_columns))] 
        
        elif type(impute_methods) == str:
            if not impute_methods in supported_impute_methods: 
                print("Invalid impute methods. Defaulting to 'drop'")
                impute_methods = "drop"
            impute_methods = [impute_methods for _ in range(len(impute_columns))]
        else: 
            for index, method in enumerate(impute_methods):
                if not method in supported_impute_methods: 
                    print("Invalid impute methods. Defaulting to 'drop'")
                    impute_methods[index] = "drop"
            self.impute_methods = impute_methods
        
    
    def get_column_summary(self):
        """
        Prints a breakdown of columns from the DataFrame 'data' that are 'numeric', 'categorical', and 'missing values'.
        """
        
        columns_with_na = self.data.columns[self.data.isna().any()].tolist()
        non_numeric = self.data.select_dtypes(exclude='number').columns
        numeric = self.data.select_dtypes(include='number').columns
        print("Non-Numeric Columns")
        print("----------------")
        print()
        for column in non_numeric: 
            print(column)
        print()

        print("Missing Values")
        print("----------------")
        print()
        for column in columns_with_na:
            num_missing_values = self.data[column].isna().sum()
            print(f"Column '{column}' has {num_missing_values} missing values.")

        print("Numeric Columns")
        print("----------------")
        print()
        for column in numeric:
            print(column)
        print()
    
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

    

    def fit(self): 
        """
        """
        try:
            self.imputed_data = self.data.drop(columns=self.drop_columns)
        except:
            print(
                    "WARNING: Invalid Dropped Columns in Parameter file: Defaulting to no dropped columns.",
                    "yellow",
            )
            self.imputed_data = self.data

        self.imputed_data = add_imputed_flags(self.data)
        
        for index, column in enumerate(self.impute_columns):
            impute_function = getattr(cleaning_utils, self.impute_methods[index])
            impute_column = self.data[column]
            self.imputed_data[column] = impute_column.apply(impute_function, axis=1)

        # Drops unaccounted columns
        self.imputed_data.dropna(axis=1, inplace=True)

        # Encoding 
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
        assert self.scaler in ["standard_scaler"], "Invalid Scaler"
        if self.scaler == "standard_scaler":
            scaler = StandardScaler()
            self.imputed_data = pd.DataFrame(
                scaler.fit_transform(self.imputed_data), columns=list(self.imputed_data.columns)
            )

    
    def save_to_file(self, out_dir, id=None): 
        """
        
        """
        if id is None:
            id = np.random.randint(0, 100)
        try:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            if self.data_path == -1: 
                data_name = "my_dataset"
            else: 
                filename_without_extension, extension = os.path.splitext(self.data_path)
                data_name = filename_without_extension.split('/')[-1]
            file_name = clean_data_filename(data_name=data_name, id=id, scaler=self.scaler, encoding=self.encoding)
            output_filepath = os.path.join(out_dir,file_name)

            results = {"data": self.imputed_data, 
                        "description": {"raw": self.data_path, 
                                        "scaler": self.scaler,
                                        "ecoding": self.encoding,
                                        "drop_columns": self.drop_columns,
                                        "impute_columns": self.impute_columns,
                                        "impute_methods": self.impute_methods}
                            }
            with open(output_filepath, "wb") as f:
                pickle.dump(results, f)
        except Exception as e:
            print(e)
