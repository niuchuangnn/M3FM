import zipfile

zip_file_path = 'demo_data.zip'
extraction_directory = './'

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents into the directory
    zip_ref.extractall(extraction_directory)

print(f"All files have been extracted")
