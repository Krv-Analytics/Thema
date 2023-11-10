import os
import sys

from dotenv import load_dotenv

################################################################################################
#  Handling Local Imports
################################################################################################


def env():
    """
    This function loads the .env file and adds necessary folders to the system path.

    Returns:
    -----------
    str
        The root directory of the project.
    """

    load_dotenv()
    root = os.getenv("root")
    src = os.getenv("src")
    sys.path.append(src + "jmapping/selecting/")
    sys.path.append(root + "logging/")
    sys.path.append(src + "modeling/synopsis/")
    return root, src
