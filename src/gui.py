from pathlib import Path
import sys
import tempfile
import os
from typing import List, Optional

os.environ.setdefault("PANDAS_DTYPE_BACKEND", "numpy")
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.vision_transformer import vit_small  # noqa: E402


st.set_page_config(page_title="Image Retrieval GUI", layout="wide")
if hasattr(pd.options.mode, "string_storage"):
    pd.options.mode.string_storage = "python"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "webp"]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.51736622, 0.51440692, 0.49375241),
            std=(0.28642876, 0.28386362, 0.30015979),
        ),
    ]
)


def _persist_uploaded_file(uploaded_file) -> Path:
    data = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix or ".ckpt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return Path(tmp.name)


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str):
    model = vit_small()
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model


def _prepare_tensors(images: List[Image.Image]) -> torch.Tensor:
    tensors = [TRANSFORM(img) for img in images]
    return torch.stack(tensors)


def _compute_attention_map(model: torch.nn.Module, image: Image.Image) -> Image.Image:
    tensor = TRANSFORM(image)
    height = tensor.shape[1] - tensor.shape[1] % 16
    width = tensor.shape[2] - tensor.shape[2] % 16
    tensor = tensor[:, :height, :width].unsqueeze(0).to(DEVICE)
    w_featmap = tensor.shape[-2] // 16
    h_featmap = tensor.shape[-1] // 16

    with torch.no_grad():
        attentions = model.get_last_selfattention(tensor)

    nh = attentions.shape[1]
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attention_grid = F.interpolate(
        attentions.unsqueeze(0), scale_factor=16, mode="nearest"
    )[0].cpu()
    attention_map = attention_grid.mean(0, keepdim=True)
    attention_map = attention_map - attention_map.min()
    attention_map = attention_map / attention_map.max().clamp_min(1e-6)
    return to_pil_image(attention_map)


def _similarity_dataframe(embeddings: torch.Tensor, labels: List[str]) -> pd.DataFrame:
    features = F.normalize(embeddings, dim=1)
    sim = torch.matmul(features, features.T).cpu().numpy()
    return pd.DataFrame(sim, index=labels, columns=labels)


def main() -> None:
    st.title("Image Retrieval Playground")
    st.caption("Upload images, select a checkpoint, and inspect cosine similarities.")

    checkpoint_upload = st.file_uploader(
        "Select checkpoint file",
        type=["ckpt", "pth", "pt"],
        accept_multiple_files=False,
    )
    checkpoint_path_input = st.text_input(
        "…or type a checkpoint path",
        value="",
        placeholder="checkpoints/final.pth",
    )

    checkpoint_path: Optional[Path] = None
    session_key = "checkpoint_temp_path"

    if checkpoint_upload is not None:
        checkpoint_path = _persist_uploaded_file(checkpoint_upload)
        st.session_state[session_key] = str(checkpoint_path)
    elif session_key in st.session_state:
        persisted = Path(st.session_state[session_key])
        if persisted.exists():
            checkpoint_path = persisted

    if not checkpoint_path and checkpoint_path_input:
        candidate = Path(checkpoint_path_input).expanduser()
        if candidate.exists():
            checkpoint_path = candidate
        else:
            st.warning("Provided checkpoint path does not exist.")

    if checkpoint_path:
        st.info(f"Using checkpoint: {checkpoint_path}")

    uploads = st.file_uploader(
        "Upload images",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        help="Add at least two images to see similarities.",
    )

    images: List[Image.Image] = []
    labels: List[str] = []
    attention_maps: List[Optional[Image.Image]] = []
    attention_error: Optional[str] = None
    if uploads:
        for upload in uploads:
            image = Image.open(upload).convert("RGB")
            images.append(image)
            labels.append(upload.name)

        attention_model: Optional[torch.nn.Module] = None
        if checkpoint_path:
            try:
                attention_model = load_model(str(checkpoint_path))
            except Exception as exc:
                attention_error = f"Failed to load checkpoint for attention maps: {exc}"
        else:
            attention_error = "Provide a checkpoint to visualize attention maps."

        if attention_model is not None:
            for image in images:
                try:
                    attention_maps.append(
                        _compute_attention_map(attention_model, image)
                    )
                except Exception as exc:  # pragma: no cover - bubble up to UI
                    attention_maps.append(None)
                    attention_error = f"Failed to compute attention maps: {exc}"
        else:
            attention_maps = [None] * len(images)

        if attention_error:
            if checkpoint_path:
                st.warning(attention_error)
            else:
                st.info(attention_error)

        base_cols = min(6, len(images)) or 1
        cols_per_row = min(6, max(2, base_cols + base_cols % 2))
        cols = st.columns(cols_per_row)
        slot_idx = 0

        for idx, (image, label) in enumerate(zip(images, labels)):
            if slot_idx >= cols_per_row:
                cols = st.columns(cols_per_row)
                slot_idx = 0
            with cols[slot_idx]:
                st.image(image, caption=label, use_column_width=True)
            slot_idx += 1

            if slot_idx >= cols_per_row:
                cols = st.columns(cols_per_row)
                slot_idx = 0
            with cols[slot_idx]:
                attention_image = (
                    attention_maps[idx] if idx < len(attention_maps) else None
                )
                if attention_image is not None:
                    st.image(
                        attention_image,
                        caption=f"{label} attention",
                        use_column_width=True,
                    )
                else:
                    st.caption("Attention unavailable")
            slot_idx += 1

    compute = st.button(
        "Compute similarities", disabled=not (checkpoint_path and len(images) >= 2)
    )

    if compute:
        if not checkpoint_path:
            st.error("Please provide a checkpoint before computing similarities.")
            return
        if len(images) < 2:
            st.error("Upload at least two images to build a similarity matrix.")
            return

        try:
            with st.spinner("Loading model…"):
                model = load_model(str(checkpoint_path))
        except Exception as exc:  # pragma: no cover - surface load errors in UI
            st.error(f"Failed to load checkpoint: {exc}")
            return

        tensors = _prepare_tensors(images).to(DEVICE)
        with torch.no_grad():
            embeddings = model(tensors)

        df = _similarity_dataframe(embeddings, labels)
        matrix = df.to_numpy(dtype="float32", copy=False)
        index = pd.Index([str(label) for label in labels], dtype="object", name="image")
        df = pd.DataFrame(matrix, index=index, columns=index, copy=False)
        st.dataframe(df.style.format("{:.3f}"), use_container_width=True)


if __name__ == "__main__":
    main()
