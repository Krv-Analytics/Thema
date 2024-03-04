import os 
from utils import env
root, src = env()
from src.projecting import pGrid

if __name__ == "__main__":
    YAML_PATH = os.getenv("params")
    my_pGrid = pGrid(YAML_PATH=YAML_PATH)
    my_pGrid.fit()