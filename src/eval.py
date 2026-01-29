from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.model import DINOModel
from pathlib import Path
import pandas as pd
import argparse
import pickle
import torch
import torch.nn.functional as F
import numpy as np


class ImageFolderWithPaths(datasets.ImageFolder):
    """Returns image tensors alongside their source paths."""

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embeddings for image retrieval."
    )
    parser.add_argument(
        "--data-dir", required=True, type=Path, help="Directory with image folders."
    )
    parser.add_argument(
        "--ground-truth", required=True, type=Path, help="Path to ground truth file."
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="Path to the trained checkpoint."
    )
    parser.add_argument(
        "--output",
        default=Path("embeddings.csv"),
        type=Path,
        help="File where the dataframe will be stored (CSV).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def build_dataloader(data_dir: Path, batch_size: int, num_workers: int) -> DataLoader:
    normalize = transforms.Normalize(
        # Google Landmarks Dataset v2 mean and std (estimated with 1000 sample images)
        mean=(0.51736622, 0.51440692, 0.49375241),
        std=(0.28642876, 0.28386362, 0.30015979),
    )
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )
    dataset = ImageFolderWithPaths(root=str(data_dir), transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def main():
    pl.seed_everything(42, workers=True)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = build_dataloader(args.data_dir, args.batch_size, args.num_workers)
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])
    model = DINOModel.load_from_checkpoint(str(args.checkpoint))
    model.eval()
    model.to(device)

    embeddings, paths = [], []
    with torch.no_grad():
        for images, img_paths in dataloader:
            images = images.to(device)
            feats = model.teacher(images)
            embeddings.append(feats.cpu())
            paths.extend(img_paths)

    embedding_tensor = torch.cat(embeddings)
    df = pd.DataFrame(embedding_tensor.numpy())
    df.insert(0, "path", paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    with open(args.ground_truth, "rb") as f:
        ground_truth = pickle.load(f)

    path_keys = [Path(p).stem for p in paths]
    key_to_idx = {key: idx for idx, key in enumerate(path_keys)}
    imlist = list(ground_truth["imlist"])
    qimlist = list(ground_truth["qimlist"])
    missing = [name for name in imlist + qimlist if name not in key_to_idx]
    if missing:
        raise ValueError(f"Missing embeddings for: {', '.join(missing[:5])}")

    features = F.normalize(embedding_tensor, dim=1)
    gallery_indices = torch.tensor(
        [key_to_idx[name] for name in imlist], dtype=torch.long
    )
    query_indices = torch.tensor(
        [key_to_idx[name] for name in qimlist], dtype=torch.long
    )
    gallery_feats = features[gallery_indices]
    query_feats = features[query_indices]

    im_to_pos = {name: idx for idx, name in enumerate(imlist)}
    rankings = []
    for qi, q_feat in enumerate(query_feats):
        sims = torch.mv(gallery_feats, q_feat)
        q_name = qimlist[qi]
        if q_name in im_to_pos:
            sims[im_to_pos[q_name]] = float("-inf")
        rankings.append(torch.argsort(sims, descending=True).tolist())

    def as_int_list(values):
        return [int(v) for v in values]

    configs = {
        "easy": lambda g: (
            as_int_list(g["easy"]),
            as_int_list(g["hard"]) + as_int_list(g["junk"]),
        ),
        "medium": lambda g: (
            as_int_list(g["easy"]) + as_int_list(g["hard"]),
            as_int_list(g["junk"]),
        ),
        "hard": lambda g: (
            as_int_list(g["hard"]),
            as_int_list(g["easy"]) + as_int_list(g["junk"]),
        ),
    }

    aggregated = {cfg: {"ap": [], "p10": [], "r_prec": [], "f1": []} for cfg in configs}

    for qi, order in enumerate(rankings):
        g = ground_truth["gnd"][qi]
        for cfg_name, selector in configs.items():
            positives, ignore = selector(g)
            if not positives:
                continue
            pos_set = set(positives)
            ignore_set = set(ignore)
            filtered = [idx for idx in order if idx not in ignore_set]
            if not filtered:
                continue
            ap_hits = 0
            ap_sum = 0.0
            for rank, im_idx in enumerate(filtered):
                if im_idx in pos_set:
                    ap_hits += 1
                    ap_sum += ap_hits / (rank + 1)
                    if ap_hits == len(pos_set):
                        break
            ap = ap_sum / len(pos_set)
            top10 = filtered[:10]
            if top10:
                hits_at10 = sum(1 for idx in top10 if idx in pos_set)
                k = len(top10)
                precision10 = hits_at10 / k
                recall10 = hits_at10 / len(pos_set)
                f1 = (
                    2 * precision10 * recall10 / (precision10 + recall10)
                    if (precision10 + recall10) > 0
                    else 0.0
                )
            else:
                precision10 = 0.0
                f1 = 0.0
            r_size = len(pos_set)
            top_r = filtered[:r_size]
            hits_r = sum(1 for idx in top_r if idx in pos_set)
            r_precision = hits_r / r_size

            aggregated[cfg_name]["ap"].append(ap)
            aggregated[cfg_name]["p10"].append(precision10)
            aggregated[cfg_name]["r_prec"].append(r_precision)
            aggregated[cfg_name]["f1"].append(f1)

    for cfg_name, scores in aggregated.items():
        if not scores["ap"]:
            continue
        count = len(scores["ap"])
        mean_ap = sum(scores["ap"]) / count
        mean_p10 = sum(scores["p10"]) / count
        mean_r = sum(scores["r_prec"]) / count
        mean_f1 = sum(scores["f1"]) / count
        print(
            f"{cfg_name.capitalize()}: "
            f"mAP={mean_ap:.4f}, mP@10={mean_p10:.4f}, "
            f"R-Precision={mean_r:.4f}, F1={mean_f1:.4f}"
        )


if __name__ == "__main__":
    main()
