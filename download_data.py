
from pathlib import Path
from urllib.request import urlretrieve

base_url = "https://physionet.org/files/ptb-xl/1.0.3/"

files = [
    "ptbxl_database.csv",
    "scp_statements.csv",
    "records500/00000/00001_hr.hea",
    "records500/00000/00001_hr.dat",
]

for filename in files:
    destination = Path("data/ptb-xl") / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    urlretrieve(base_url + filename, destination)
    print("Downloaded:", filename)
