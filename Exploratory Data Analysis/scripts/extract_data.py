import tarfile
import os

def extract_tgz(tgz_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with tarfile.open(tgz_path, "r:gz") as tar:
        print(f"Extracting {tgz_path} to {extract_to}...")
        tar.extractall(path=extract_to)
        print("Extraction complete.")