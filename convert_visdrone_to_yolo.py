# # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# # VisDrone2019-DET dataset https://github.com/VisDrone/VisDrone-Dataset by Tianjin University
# # Documentation: https://docs.ultralytics.com/datasets/detect/visdrone/
# # Example usage: yolo train data=VisDrone.yaml
# # parent
# # ├── ultralytics
# # └── datasets
# #     └── VisDrone ← downloads here (2.3 GB)

# # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
# path: VisDrone # dataset root dir
# train: images/train # train images (relative to 'path') 6471 images
# val: images/val # val images (relative to 'path') 548 images
# test: images/test # test-dev images (optional) 1610 images

from pathlib import Path
import shutil

from ultralytics.utils.downloads import download
from ultralytics.utils import TQDM
import fiftyone as fo

# # Classes
dataset_map = {
    "0": "person",
    "1": "car",
    "2": "motorcycle",
    "3": "airplane",
    "4": "bus",
    "5": "boat",
    "6": "stop sign",
    "7": "snowboard",
    "8": "umbrella",
    "9": "soccer ball",
    "10": "basketball",
    "11": "volleyball",
    "12": "football",
    "13": "baseball bat",
    "14": "bed",
    "15": "tennis racket",
    "16": "suitcase",
    "17": "skis",
}

visdrone_map = {
    "0": "pedestrian",
    "1": "people",
    "2": "bicycle",
    "3": "car",
    "4": "van",
    "5": "truck",
    "6": "tricycle",
    "7": "awning-tricycle",
    "8": "bus",
    "9": "motor",
}


def visdrone2yolo(dir, split, source_name=None):
    """Convert VisDrone annotations to YOLO format with images/{split} and labels/{split} structure."""
    from PIL import Image

    source_dir = dir / (source_name or f"VisDrone2019-DET-{split}")
    images_dir = dir / "images" / split
    labels_dir = dir / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Move images to new structure
    if (source_images_dir := source_dir / "images").exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        for img in source_images_dir.glob("*.jpg"):
            img.rename(images_dir / img.name)

    for f in TQDM(
        (source_dir / "annotations").glob("*.txt"), desc=f"Converting {split}"
    ):
        img_size = Image.open(images_dir / f.with_suffix(".jpg").name).size
        dw, dh = 1.0 / img_size[0], 1.0 / img_size[1]
        lines = []

        with open(f, encoding="utf-8") as file:
            for row in [x.split(",") for x in file.read().strip().splitlines()]:
                if row[4] != "0":  # Skip ignored regions
                    x, y, w, h = map(int, row[:4])
                    cls = int(row[5]) - 1
                    # Convert to YOLO format
                    x_center, y_center = (x + w / 2) * dw, (y + h / 2) * dh
                    w_norm, h_norm = w * dw, h * dh
                    lines.append(
                        f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                    )

        (labels_dir / f.name).write_text("".join(lines), encoding="utf-8")


def visdrone2SAETA(dpath):
    """Convert VisDrone class IDs to SAETA class IDs based on class name mapping."""
    # Create mapping from visdrone to SAETA class IDs
    visdrone_to_saeta = {}

    # Map based on class names
    # visdrone: 0=pedestrian, 1=people -> SAETA: 0=person
    visdrone_to_saeta[0] = 0  # pedestrian -> person
    visdrone_to_saeta[1] = 0  # people -> person

    # visdrone: 3=car -> SAETA: 1=car
    visdrone_to_saeta[3] = 1  # car -> car

    # visdrone: 8=bus -> SAETA: 4=bus
    visdrone_to_saeta[8] = 4  # bus -> bus

    # visdrone: 9=motor -> SAETA: 2=motorcycle
    visdrone_to_saeta[9] = 2  # motor -> motorcycle

    # Classes without direct mapping are ignored: bicycle, van, truck, tricycle, awning-tricycle

    splits = ["test", "train", "val"]
    for split in splits:
        split_path = dpath / "labels" / split
        if not split_path.exists():
            print(f"Directory not found: {split_path}")
            continue

        for txt_file in split_path.glob("*.txt"):
            new_lines = []
            with open(txt_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        visdrone_class = int(parts[0])

                        # Only keep classes that map to SAETA dataset
                        if visdrone_class in visdrone_to_saeta:
                            saeta_class = visdrone_to_saeta[visdrone_class]
                            parts[0] = str(saeta_class)
                            new_lines.append(" ".join(parts) + "\n")

            # Write back the converted annotations
            with open(txt_file, "w") as f:
                f.writelines(new_lines)


def load_datasets_from_dirs(dataset_name, base_dirs, splits=["train", "valid", "test"]):
    """
    Load multiple COCO datasets from base directories into a single FiftyOne dataset.

    Args:
        dataset_name: Name for the combined FiftyOne dataset
        base_dirs: List of base directory paths (e.g., ["test-2", "2026_SUAS-2"])
        splits: List of split names to load (default: ["train", "valid", "test"])

    Returns:
        FiftyOne Dataset
    """
    # Delete existing dataset
    try:
        fo.delete_dataset(dataset_name)
    except:
        pass

    dataset = fo.Dataset(name=dataset_name)

    # Construct splits from base directories
    all_splits = []
    for base_dir in base_dirs:
        for split in splits:
            data_path = f"{base_dir}/{split}"
            labels_path = f"{base_dir}/{split}/_annotations.coco.json"
            all_splits.append((data_path, labels_path, split))

    # Load all splits
    for data_path, labels_path, tag in all_splits:
        print(f"Loading {tag} from {data_path}...")
        dataset.add_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=data_path,
            labels_path=labels_path,
            tags=tag,
            progress=True,
        )

    print(f"\n{dataset}")
    print(f"Total samples: {len(dataset)}")

    return dataset


if __name__ == "__main__":
    # Download (ignores test-challenge split)
    # dir = Path(yaml["path"])  # dataset root dir
    dir = Path("/home/cole/cole_scripts/vision/SAETA/datasets")
    urls = [
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
        # "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-challenge.zip",
    ]
    download(urls, dir=dir, threads=4)

    # Convert
    splits = {
        "VisDrone2019-DET-train": "train",
        "VisDrone2019-DET-val": "val",
        "VisDrone2019-DET-test-dev": "test",
    }
    for folder, split in splits.items():
        visdrone2yolo(dir, split, folder)  # convert VisDrone annotations to YOLO labels
        shutil.rmtree(dir / folder)  # cleanup original directory
