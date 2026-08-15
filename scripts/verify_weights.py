from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


EXPECTED = {
    "weights/metric_encoder_all_params.pth": "fc91c12ee7ea7b0859dfe9946fc9345748dab795663954240ed245aada6edce9",
    "weights/association_gnn/checkpoints/best.ckpt": "f7bed4a893ad4a85efec03d1672744c5ec32ae79221882d0b95679bf05a17d1b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    import torch

    observed = {}
    for relative_path, expected in EXPECTED.items():
        path = ROOT / relative_path
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {relative_path}")
        observed[relative_path] = actual

    metric = torch.load(ROOT / "weights/metric_encoder_all_params.pth", map_location="cpu")
    required = {"trunk_state_dict", "embedder_state_dict", "roi", "mlp_dims"}
    if not required.issubset(metric):
        raise RuntimeError(f"Metric checkpoint is missing {sorted(required.difference(metric))}")

    from src.models.celltrack_plmodel import CellTrackLitModel

    model = CellTrackLitModel.load_from_checkpoint(
        str(ROOT / "weights/association_gnn/checkpoints/best.ckpt"),
        map_location="cpu",
    )
    model.eval()
    print(
        json.dumps(
            {
                "status": "ok",
                "checksums": observed,
                "metric_embedding_dim": int(metric["mlp_dims"][-1]),
                "gnn_input_mode": model.hparams.model_params["kwargs"]["input_mode"],
                "gnn_layers": model.hparams.model_params["kwargs"]["message_passing"]["kwargs"]["num_layers"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
