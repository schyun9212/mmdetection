import os
import requests
from zipfile import ZipFile
import math
from typing import Optional

from tqdm import tqdm


COCO_2017_ANNOTATION_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_2017_IMAGE_URL = "http://images.cocodataset.org/zips/{}2017.zip"


# TODO: move common function
def download_file(
    url: str,
    output_dir: str,
    chunk_size: Optional[int] = 4096
):
    name = os.path.basename(url)
    output_path = os.path.join(output_dir, name)
    
    if not os.path.exists(output_path):
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        with open(output_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=math.ceil(total_size / chunk_size)):
                f.write(chunk)

    return output_path


def download_coco2017_annotation(data_root: str):
    print("Start to download annotations")
    zipfile_path = download_file(COCO_2017_ANNOTATION_URL, data_root)

    with ZipFile(zipfile_path, "r") as f:
        f.extractall(data_root)


def download_coco2017_image(
    data_root: str,
    data_type: Optional[str] = "val",
) -> None:
    print(f"Start to download {data_type} images")
    url = COCO_2017_IMAGE_URL.format(data_type)
    zipfile_path = download_file(url, data_root)

    with ZipFile(zipfile_path, "r") as f:
        f.extractall(data_root)


def main(data_root: str, val_only: Optional[bool] = True):
    if not os.path.exists("{}/{}".format(data_root, "annotations")):
        download_coco2017_annotation(data_root)
    
    download_coco2017_image(data_root, "val")
    if not val_only:
        download_coco2017_image(data_root, "train")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/coco")
    parser.add_argument("--val-only", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
