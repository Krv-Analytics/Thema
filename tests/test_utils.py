# File:tests/test_utils.py
# Last Updated: 04-05-24
# Updated By: SW

import pandas as pd
import pandas as pd
import numpy as np
import random
import networkx as nx


def generate_dataframe(num_rows=100, seed=42):
    "Generates a random dataFrame with num_rows Rows and 15 Columns containing Null Values"
    random.seed(seed)

    categories = ["A", "B", "C", "D", "E"]
    cat_data = [random.choice(categories) for _ in range(num_rows)]

    numeric_data = [random.random() for _ in range(num_rows)]

    for i in range(5):
        random_indices = random.sample(range(num_rows), random.randint(5, 15))
        for idx in random_indices:
            numeric_data[idx] = None

    data = {
        "Cat1": cat_data,
        "Cat2": cat_data,
        "Cat3": cat_data,
        "Cat4": cat_data,
        "Cat5": cat_data,
        "Num1": numeric_data,
        "Num2": numeric_data,
        "Num3": numeric_data,
        "Num4": numeric_data,
        "Num5": numeric_data,
        "Num6": numeric_data,
        "Num7": numeric_data,
        "Num8": numeric_data,
        "Num9": numeric_data,
        "Num10": numeric_data,
    }

    df = pd.DataFrame(data)
    return df


_test_data_0 = pd.DataFrame(
    {"A": [1, 2, 3], "B": [4, 5, None], "C": ["a", "b", "c"]}
)


_test_data_1 = pd.DataFrame(
    {
        "A": [1, 2, None, 4, 5],
        "B": ["a", "b", None, "d", "d"],
        "C": ["x", "y", "z", None, "w"],
        "D": [None, 10, 20, 30, 40],
        "E": ["p", "q", "r", "r", None],
        "F": ["u", "v", None, "x", "y"],
    }
)

_test_data_2 = pd.DataFrame(
    {
        "X": [1, 2, 3, 4, 5],
        "Y": ["a", None, "c", "d", "e"],
        "Z": [None, 10, 20, 30, 40],
        "W": ["p", "q", "r", "s", "t"],
        "V": ["u", "v", "w", "x", None],
        "U": ["i", "j", "k", "l", "m"],
    }
)

_test_data_3 = pd.DataFrame(
    {
        "A": [1, None, 3, None, 5],
        "B": [None, 2, None, 4, None],
        "C": [6, None, 8, None, 10],
        "D": [None, 1.1, 2.0, None, 2.3],
    }
)

_test_data_0_missingData_summary = {
    "numericMissing": ["B"],
    "numericComplete": ["A"],
    "categoricalMissing": [],
    "categoricalComplete": ["C"],
}

_test_data_1_missingData_summary = {
    "numericMissing": ["A", "D"],
    "numericComplete": [],
    "categoricalMissing": ["B", "C", "E", "F"],
    "categoricalComplete": [],
}

_test_data_2_missingData_summary = {
    "numericMissing": ["Z"],
    "numericComplete": ["X"],
    "categoricalMissing": ["Y", "V"],
    "categoricalComplete": ["W", "U"],
}


_test_cleanData_0 = pd.DataFrame(
    {
        "A": [-1.224745, 0.000000, 1.224745],
        "B": [-1.400710, 0.869202, 0.531507],
        "impute_B": [-0.707107, -0.707107, 1.414214],
        "OH_C_a": [1.414214, -0.707107, -0.707107],
        "OH_C_b": [-0.707107, 1.414214, -0.707107],
        "OH_C_c": [-0.707107, -0.707107, 1.414214],
    }
)


