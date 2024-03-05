import os
from types import FunctionType as function
import pickle
import pandas as pd
from . import imputing_utils as iu


class iGen: 
    """
    Impution Generator 
    TODO: Update Doc String 
    """

    def __init__(self, data, impute_columns, impute_methods, id): 
        """
        iGen init 
         TODO: Update Doc String 
        """

        if type(data) == str:
            try:
                if data.endswith('.pkl'):
                    assert os.path.isfile(data), "\n Invalid path to Clean Data"
                    with open(data, "rb") as clean:
                        reference = pickle.load(clean)
                        self.data = reference["clean_data"]
                    self.data_path = data
                else:
                    raise ValueError("Unsupported file format. Supported formats: '.pkl' ")
            except:
                raise ValueError("There was an issue opening your data file. Please make sure it exists and is a supported file format.")
        
        elif isinstance(data, pd.DataFrame):
            self.data = data 
            self.data_path = -1 
        else: 
            raise ValueError("'data' must be a pd.DataFrame object, OR a path to a csv, xlxs, or pickle file")

        self.imputed_data = self.data.copy()
        self.impute_columns = impute_columns
        self.impute_methods = impute_methods 
        self.id = id 

    def get_columns_w_NA(self):
        """
        Returns a list of columns from the DataFrame 'data' that contain missing values (NA).
        """
        columns_with_na = self.data.columns[self.data.isna().any()].tolist()
        return columns_with_na

    def fit(self): 
        """
        """
        for index, column in enumerate(self.impute_columns):
            impute_function = getattr(iu, self.impute_methods[index])
            impute_column = self.data[column]
        self.imputed_data[column] = impute_column.apply(impute_function, axis=1)
        self.imputed_data.dropna(axis=1, inplace=True)
    
    def save_to_file(self, out_dir): 
        """
        
        """
        try:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            if self.data_path == -1: 
                data_name = "my_dataset"
            else: 
                filename_without_extension, extension = os.path.splitext(self.data_path)
                data_name = filename_without_extension.split('/')[-1]
            
            file_name = iu.imputed_filename(data_name, id=self.id)
            output_filepath = os.path.join(out_dir,file_name)

            # TODO: log random sample seed
            results = {"imputed_data": self.imputed_data, 
                        "description": {"clean_data": self.data_path, 
                                        "impute_columns": self.impute_columns,
                                        "impute_methods": self.impute_methods}
                            }
            with open(output_filepath, "wb") as f:
                pickle.dump(results, f)
        except Exception as e:
            print(e)
