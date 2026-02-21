from os import path

# Parent Directory of where THIS file lives
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

OUTPUTS = path.join(BASE_DIR, "Outputs")
NOTEBOOKS = path.join(BASE_DIR, "Notebooks")

ASSETS = path.join(BASE_DIR, "Assets")
DATASET = path.join(ASSETS, "Dataset")
MODELS = path.join(ASSETS, "Models")

LOGS = path.join(BASE_DIR, "Logs")


PATHS = {
    name: value
    for name, value in globals().items()
    if name.isupper() and isinstance(value, str)
}

# missing_paths = {}
for name, p in PATHS.items():
    if not path.exists(p):
        
        raise FileNotFoundError(
            f"\nRequired path does not exist. Please create it.\n"
            f"\tKey  -- {name}\n"
            f"\tPath -- {p}"
        )