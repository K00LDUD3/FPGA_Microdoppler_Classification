# Add Base directory to Python path
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent))
from os import path as ospath, makedirs as osmakedirs
from json import dump as jsondump, load as jsonload
from Utils.paths import LOGS, NOTEBOOKS
from datetime import datetime
from pandas import json_normalize as pd_json_normalize

def save_config_to_json(config_dict, json_filename_abs, verbose=False):
    if not json_filename_abs.endswith(".json"):
        json_filename_abs += ".json"

    json_filename_abs = LOGS+"\\" + json_filename_abs
    osmakedirs(ospath.dirname(json_filename_abs) or ".", exist_ok=True)

    # Load existing data if file exists
    if ospath.exists(json_filename_abs):
        with open(json_filename_abs, "r", encoding="utf-8") as f:
            try:
                data = jsonload(f)
            except:
                data = {}
    else:
        data = {}

    
    run_id = f"{len(data)+1:03d}"
    config_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    

    file_path = config_dict.get("file_path") or config_dict.get("model_save_path")
    if file_path and ospath.exists(file_path):
        size_mb = ospath.getsize(file_path) / (1024 ** 2)
        if size_mb >= 100:
            size_gb = size_mb / 1024
            config_dict["size"] = f"{round(size_gb, 4)} GB"
        else:
            config_dict["size"] = f"{round(size_mb, 4)} MB"
    else:
        raise FileNotFoundError(f"File path does not exist: {file_path}")
    
    data[run_id] = config_dict

    
    with open(json_filename_abs, "w", encoding="utf-8") as f:
        jsondump(data, f, indent=4, ensure_ascii=False)

    if verbose:
        print(f"Config saved to '{json_filename_abs}'")



def read_json_config(config_path: str, normalize: bool = False):
    """
    Reads a JSON config file and returns it as a dictionary.
    
    Args:
        config_path (str): Path to the JSON config file.
        normalize (bool): If True, returns a flattened pandas DataFrame.

    Returns:
        dict | pandas.DataFrame
    """
    if not ospath.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = jsonload(f)

    if not isinstance(config, dict):
        raise ValueError("JSON config must contain a single dictionary at the top level.")

    if normalize:
        return pd_json_normalize(config)

    return config


def new_entry_version(filename):
    if not filename.endswith(".json"):
        filename += ".json"

    filepath = LOGS + "\\" + filename

    if ospath.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = jsonload(f)
                existing_ids = [int(k) for k in data.keys() if k.isdigit()]
                next_id = max(existing_ids) + 1 if existing_ids else 1
            except:
                next_id = 1
    else:
        next_id = 1

    return f"{next_id:03d}"

def load_config_as_dataframe(filename, fillna_with=""):
    if not filename.endswith(".json"):
        filename += ".json"

    filepath = LOGS + "\\" + filename

    if not ospath.exists(filepath):
        raise FileNotFoundError(f"{filepath} does not exist.")

    with open(filepath, "r", encoding="utf-8") as f:
        data = jsonload(f)

    # Remove global timestamp
    data.pop("timestamp", None)

    # Extract run_ids and configs separately
    run_ids = list(data.keys())
    configs = list(data.values())

    # Flatten nested dictionaries
    df = pd_json_normalize(configs, sep=".")

    # Insert run_id as first column
    df.insert(0, "run_id", run_ids)

    if fillna_with is not None:
        df = df.fillna(fillna_with)

    return df




def get_config_by_id(json_path: str, run_id: str, as_dataframe: bool = False):
    """
    Retrieve a configuration entry from JSON file by run_id.

    Args:
        json_path (str): Absolute path to JSON file.
        run_id (str | int): ID key to retrieve (e.g., "004").
        as_dataframe (bool): If True, return as single-row DataFrame.

    Returns:
        dict | pandas.DataFrame
    """

    if not ospath.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = jsonload(f)

    run_id = str(run_id)

    if run_id not in data:
        raise KeyError(f"Run ID '{run_id}' not found in JSON file.")

    result = data[run_id]

    if as_dataframe:
        df = pd_json_normalize(result)
        df.insert(0, "run_id", run_id)
        return df

    return result

print(new_entry_version("experiments_metadata.json"))