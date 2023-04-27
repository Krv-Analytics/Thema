import os

root = os.getcwd()
src = os.path.join(root, "src/")

env_file = os.path.join(root, ".env")
# Write .env file
with open(env_file, "w") as f:
    f.write(f"root={root}/\n")
    f.write(f"src={src}")
