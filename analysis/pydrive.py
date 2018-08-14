from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from tqdm import tqdm


CADP_FRAME_FOLDER_ID="1BRIv9h4c_zbqJ6Ye4rVyFSo3K20BQt_v"

def authenticate():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return gauth


def list_folder(drive, folder_id):
    _q = {'q': "'{}' in parents and trashed=false".format(folder_id)}
    return drive.ListFile(_q).GetList()


def download_cadp(gauth, output_path):
    drive = GoogleDrive(gauth)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    subfolders = list_folder(drive, CADP_FRAME_FOLDER_ID)
    for sf in subfolders:
        print("Downloading {} ...".format(sf["title"]))
        dir_name = "{}/{}".format(output_path,sf["title"])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fid = sf["id"]
        files = list_folder(drive, fid)
        for f in tqdm(files):
            f.GetContentFile("{}/{}".format(dir_name, f["title"]))


