import urllib.request
import zipfile
import os

url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
zip_path = "coco128.zip"

print("Downloading coco128...")
urllib.request.urlretrieve(url, zip_path)
print("Download complete!")

print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(".")
print("Done! Check the 'coco128' folder.")

os.remove(zip_path)