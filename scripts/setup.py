# Needs to be run from the root directory!
import os

root = os.getcwd()
src = os.path.join(root, "src/")
params = os.path.join(root, "params.json")

env_file = os.path.join(root, ".env")
# Write .env file
with open(env_file, "w") as f:
    f.write(f"root='{root}/'\n")
    f.write(f"src='{src}'\n")
    f.write("mongo_client='' #Your Mongo Client ID\n")
    f.write(f"params='{params}'")

    f.write(
        "mongo_database='' #The name of the MongoDB database where the dataset is located.\n"
    )
    f.write(
        "mongo_collection='' #The name of the collection within the specified MongoDB database."
    )
