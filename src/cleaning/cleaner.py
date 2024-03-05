# File: src/cleaning/cleaner.py 
# Last Updated: 03-04-24
# Updated by: SW 

import os
import pickle
import pandas as pd


from .cleaning_utils import clean_data_filename, integer_encoder
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from termcolor import colored
from ..utils import env 

root, src = env()  # Load .env

class cleaner: 
    """TODO: Update Doc String """

    def __init__(self, data=None, scaler=None, encoding=None, drop_columns=None, verbose=True, YAML_PATH=None): 
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
                data = params.raw_data
                scaler = params.scaler 
                encoding = params.encoding 
                drop_columns = params.cleaning_remove_columns
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
                    with open(data, "rb") as clean:
                        reference = pickle.load(clean)
                        self.data = reference["clean_data"]
                    self.data_path = data
                else:
                    raise ValueError("Unsupported file format. Supported formats: .csv, .xlsx, .pkl")
            except:
                print("There was an issue opening your data file. Please make sure it exists and is a supported file format.")
        
        elif isinstance(data, pd.DataFrame):
            self.data = data 
            self.data_path = -1 
        else: 
            raise ValueError("'data' must be a pd.DataFrame object, OR a path to a csv, xlxs, or pickle file")

        self.scaler = scaler 
        self.encoding = encoding 
        self.drop_columns = drop_columns


    def fit(self):   
        try:
            self.clean = self.data.drop(columns=self.drop_columns)
        except:
            print(
                colored(
                    " \n WARNING: Invalid Dropped Columns in Parameter file: Defaulting to no dropped columns.",
                    "yellow",
                )
            )
            self.clean = self.data

        # Encoding
        assert self.encoding in ["integer", "one_hot", "hash"], colored(
            "\n ERROR: Invalid Encoding. We currently only support `integer`,`one_hot` and `hash` encodings",
            "red",
        )
        if self.encoding == "one_hot":
            non_numeric_columns = cleaned_data.select_dtypes(exclude=["number"]).columns
            for column in non_numeric_columns:
                cleaned_data = pd.get_dummies(
                    self.clean, prefix=f"OH_{column}", columns=[column]
                )

        if self.encoding == "integer":
            categorical_variables = cleaned_data.select_dtypes(
                exclude=["number"]
            ).columns.tolist()
            # Rewrite Columns
            for column in categorical_variables:
                vals = cleaned_data[column].values
                self.cleaned[column] = integer_encoder(vals)

        if self.encoding == "hash":
            categorical_variables = cleaned_data.select_dtypes(
                exclude=["number"]
            ).columns.tolist()
            hashing_encoder = ce.HashingEncoder(cols=categorical_variables, n_components=10)
            self.clean = hashing_encoder.fit_transform(cleaned_data)

        # Scaling 
        if self.scaler is not None:
            self.scaler = StandardScaler()
            self.clean = pd.DataFrame(
                self.scaler.fit_transform(cleaned_data), columns=list(cleaned_data.columns)
            )

    def save_to_file(self, out_dir): 
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        if self.data_path == -1: 
            data_name = "my_dataset"
        else: 
            filename_without_extension, extension = os.path.splitext(self.data_path)
            data_name = filename_without_extension.split('/')[-1]
        
        output_file = clean_data_filename(
            run_name=data_name,
            scaler=self.scaler,
            encoding=self.encoding,
            filter=False if self.drop_columns else True,
        ) 
        output_file = os.path.join(out_dir, output_file)
        rel_outfile = "/".join(output_file.split("/")[-4:])

        results = {"clean_data": self.clean, 
                   "description": {"raw_data": self.data, "scaler": self.scaler, "encoding":self.encoding, "drop_columns":self.drop_columns}}
        
        # Write to pickle
        with open(output_file, "wb") as f:
            pickle.dump(results, f)

        # If YAML Config File was specified, write back with location of clean data
        try: 
            if self.YAML_PATH is not None: 
                with open(self.YAML_PATH, "r") as f:
                    params = OmegaConf.load(f)
                params["clean_data"] = rel_outfile

                with open(self.YAML_PATH, "w") as f:
                    OmegaConf.save(params, f)
        except:
            print(
                colored(
                "ERROR: Unable to write to params.yaml file. \n Make sure it exists and you have set appropriate file permissions.  ",
                "red",
                )
            )

        if self.verbose:
            print(
                "\n\n-------------------------------------------------------------------------------- \n\n"
            )
            print(
                colored(f"SUCCESS: Completed data Cleaning.", "green"),
                f"Written to `{rel_outfile}`",
            )

            print(
                "\n\n-------------------------------------------------------------------------------- "
        )
