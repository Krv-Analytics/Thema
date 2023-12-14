# Needs to be run from the root directory!
import os
import shutil

root = os.getcwd()
src = os.path.join(root, "src/")
params = os.path.join(root, "params.yaml")


# Make `.env` file if it doesn't exist
env_file = os.path.join(root, ".env")
if not os.path.isfile(env_file):
    with open(env_file, "w") as f:
        f.write(f"root='{root}/'\n")
        f.write(f"src='{src}'\n")
        f.write("mongo_client='' #Your Mongo Client ID\n")
        f.write(
            "mongo_database='' #The name of the MongoDB database where the dataset is located.\n"
        )
        f.write(
            "mongo_collection='' #The name of the collection within the specified MongoDB database.\n"
        )

        f.write(f"params='{params}'")


# Make `data` directory if it doesn't exist
data = os.path.join(root, "data")
if not os.path.isdir(data):
    os.mkdir(data)

# Make `params.yaml` file if it doesn't exist
params = os.path.join(root, "params.yaml")
if not os.path.isfile(params):
    shutil.copy(os.path.join(root, "paramsSAMPLE.yaml"), params)


# Write .env file
with open(env_file, "w") as f:
    f.write(f"root='{root}/'\n")
    f.write(f"src='{src}'\n")
    f.write("mongo_client='' #Your Mongo Client ID\n")
    f.write(
        "mongo_database='' #The name of the MongoDB database where the dataset is located.\n"
    )
    f.write(
        "mongo_collection='' #The name of the collection within the specified MongoDB database.\n"
    )

    f.write(f"params='{params}'")
