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
        transforms.Resize((224, 224)),
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
    if uploads:
        cols = st.columns(min(3, len(uploads)))
        for idx, upload in enumerate(uploads):
            image = Image.open(upload).convert("RGB")
            images.append(image)
            labels.append(upload.name)
            with cols[idx % len(cols)]:
                st.image(image, caption=upload.name, use_column_width=True)

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
