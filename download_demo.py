import requests
import zipfile
import os


def download_and_unzip(url, extract_to='.'):
    """
    Download a zip file from the given URL and unzip it into the specified directory.

    Args:
    url (str): The URL of the zip file to download.
    extract_to (str): The directory to extract the zip file into.
    """
    # Download the file
    response = requests.get(url)
    zip_filename = os.path.join(extract_to, 'temp.zip')

    # Write the downloaded file to a new file on disk
    with open(zip_filename, 'wb') as f:
        f.write(response.content)

    # Unzip the file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Clean up the temporary zip file
    os.remove(zip_filename)
    print(f'File extracted to {extract_to}')


# Use the direct download URL
download_url = 'https://drive.google.com/uc?export=download&id=1QJer00vxumElsZIvdpdJLcD_6jeCbX4j'
download_and_unzip(download_url, './demo_data/')