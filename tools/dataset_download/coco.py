import os
import requests
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
import math
from typing import Optional

from tqdm import tqdm
from pycocotools.coco import COCO


COCO_2017_ANNOTATION_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


# TODO: move common function
def download_file(
    url: str,
    output_dir: str,
    chunk_size: Optional[int] = 4096
):
    name = os.path.basename(url)
    output_path = os.path.join(output_dir, name)

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    with open(output_path, "wb") as f:
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=math.ceil(total_size / chunk_size)):
            f.write(chunk)

    return f.name


def download_coco2017_annotation(
    data_root: str,
    remove: Optional[bool] = False
):
    print("Start to download annotations")
    zipfile_path = download_file(COCO_2017_ANNOTATION_URL, data_root)

    with ZipFile(zipfile_path, "r") as f:
        f.extractall(data_root)

    print("Complete to download annotations")
    if remove:
        os.remove(zipfile_path)


def download_coco2017(
    data_root: str,
    data_type: Optional[str] = "val",
    task: Optional[str] = "instances",
) -> None:
    ann_file='{}/annotations/{}_{}2017.json'.format(data_root, task, data_type)
    data_dir = "{}/{}2017".format(data_root, data_type)
    coco = COCO(ann_file)

    images = list(coco.imgs.keys())
    chunk_size = int(len(images) / os.cpu_count())
    image_chunks = [images[i:i+chunk_size] for i in range(0, len(images), chunk_size)]
    task = lambda x: coco.download(data_dir, x)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(task, image_chunks)


def main(data_root: str, task: str, val_only: Optional[bool] = True):
    if not os.path.exists("{}/{}".format(data_root, "annotations")):
        download_coco2017_annotation(data_root)
    
    download_coco2017(data_root, "val", task)
    if not val_only:
        download_coco2017(data_root, "train", task)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/coco")
    parser.add_argument("--task", type=str, default="instances")
    parser.add_argument("--val-only", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
