import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image


def load_rows(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "super_class", "sub_class"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns in {path}: {required}")
        for row in reader:
            rows.append(row)
    return rows


def check_files(rows, root: Path, limit: int):
    missing = []
    inspected = 0
    sizes = defaultdict(int)
    for row in rows:
        img_path = root / row["image"]
        if not img_path.exists():
            missing.append(row["image"])
            continue
        if inspected < limit:
            with Image.open(img_path) as im:
                sizes[im.mode] += 1
            inspected += 1
    return missing, sizes


def summarize(rows):
    total = len(rows)
    super_counter = Counter(r["super_class"] for r in rows)
    sub_counter = Counter(r["sub_class"] for r in rows)
    pair_counter = Counter((r["super_class"], r["sub_class"]) for r in rows)
    return total, super_counter, sub_counter, pair_counter


def print_counter(title, counter: Counter, top: int = 10):
    print(title)
    for name, count in counter.most_common(top):
        print(f"  {name}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--train", type=str, default="data/train.csv")
    parser.add_argument("--val", type=str, default="data/val.csv")
    parser.add_argument("--inspect-images", type=int, default=10, help="open up to N images to check modes")
    args = parser.parse_args()

    root = Path(args.root)
    train_path = Path(args.train)
    val_path = Path(args.val)

    print(f"Root: {root}")
    print(f"Train CSV: {train_path}")
    print(f"Val CSV: {val_path}")

    train_rows = load_rows(train_path)
    val_rows = load_rows(val_path)

    print(f"Train samples: {len(train_rows)}")
    print(f"Val samples: {len(val_rows)}")

    train_total, train_super, train_sub, train_pairs = summarize(train_rows)
    val_total, val_super, val_sub, val_pairs = summarize(val_rows)

    print_counter("Train super-class counts", train_super)
    print_counter("Train sub-class counts", train_sub)
    print_counter("Val super-class counts", val_super)
    print_counter("Val sub-class counts", val_sub)

    train_missing, train_modes = check_files(train_rows, root, args.inspect_images)
    val_missing, val_modes = check_files(val_rows, root, args.inspect_images)

    if train_missing or val_missing:
        print("Missing files:")
        for m in train_missing[:10]:
            print(f"  train: {m}")
        for m in val_missing[:10]:
            print(f"  val: {m}")
        if len(train_missing) > 10 or len(val_missing) > 10:
            print("  ...")
    else:
        print("No missing image files detected.")

    if train_modes or val_modes:
        print("Image modes observed (first files only):")
        for mode, count in (train_modes | val_modes).items():
            print(f"  {mode}: {count}")

    shared_pairs = set(train_pairs) & set(val_pairs)
    print(f"Shared super/sub combinations between splits: {len(shared_pairs)}")


if __name__ == "__main__":
    main()
