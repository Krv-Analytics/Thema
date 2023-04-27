"Preprocess your data before applying JMapper."
import argparse
import os
import pickle
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

from cleaner_helper import data_cleaner, clean_data_filename

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    load_dotenv()
    root = os.getenv("root")

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=os.path.join(root, "data/raw/coal_plant_data_raw.pkl"),
        help="Select raw data to clean.",
    )
    parser.add_argument(
        "-s",
        "--scaler",
        default="standard_scaler",
        help="Select `sklearn` compatible method to scale data.",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        type=str,
        default="integer",
        help="Method for encoding categorical fields.",
    )

    parser.add_argument(
        "-r",
        "--remove_columns",
        type=list,
        default=[
            "ORISPL",
            "coal_FUELS",
            "NONcoal_FUELS",
            "ret_DATE",
            "PNAME",
            "FIPSST",
            "PLPRMFL",
            "FIPSCNTY",
            "LAT",
            "LON",
            "Utility ID",
            "Entity Type",
            "STCLPR",
            "STGSPR",
            "SECTOR",
        ],
        help="Specify list of columns to drop.",
    )
    parser.add_argument(
        "-v",
        "--Verbose",
        default=False,
        action="store_true",
        help="If set, will print messages detailing computation and output.",
    )

    args = parser.parse_args()

    # Read in Raw Data
    assert os.path.isfile(args.data), "Invalid path to Raw Data"

    with open(args.data, "rb") as raw_data:
        df = pickle.load(raw_data)

    assert (
        args.scaler == "standard_scaler"
    ), "Currently we only support `StandardScaler`"
    scaler = StandardScaler()

    # Clean
    clean_data = data_cleaner(
        data=df,
        scaler=scaler,
        column_filter=args.remove_columns,
        encoding=args.encoding,
    )

    output_dir = os.path.join(root, "data/clean/")

    column_filter = False
    if len(args.remove_columns) > 0:
        column_filter = True

    output_file = clean_data_filename(
        scaler=scaler, encoding=args.encoding, filter=column_filter
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, output_file)
    out_dir_message = "/".join(output_file.split("/")[-2:])

    output = {"clean_data": clean_data, "dropped_columns": args.remove_columns}
    # Write to pickle
    with open(output_file, "wb") as f:
        pickle.dump(output, f)

    if args.Verbose:
        print(
            "\n################################################################################## \n\n"
        )

        print(f"Finished cleaning data! Written to {out_dir_message}")

        print(
            "\n\n##################################################################################\n"
        )
