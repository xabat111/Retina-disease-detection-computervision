"""Utilidades generales: semillas, dispositivo, rutas (Kaggle vs. local) y dependencias."""

from __future__ import annotations

import importlib
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

KAGGLE_INPUT_DIR = Path("/kaggle/input")
KAGGLE_WORKING_DIR = Path("/kaggle/working")
OUTPUT_DIR_ENV_VARIABLE = "RETINA_OUTPUT_DIR"
DATA_ROOT_ENV_VARIABLE = "RETINA_DATA_ROOT"

# Carpeta de cada split en la copia local del repositorio original.
LOCAL_SPLIT_FOLDERS = {"train": "Training_Set", "val": "Evaluation_Set", "test": "Test_Set"}


def set_seed(seed: int) -> None:
    """Fija las semillas de Python, NumPy y PyTorch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"GPU: {torch.cuda.get_device_name(device)}"
    return "CPU (sin GPU disponible)"


def is_kaggle() -> bool:
    return KAGGLE_INPUT_DIR.exists()


def resolve_output_dir() -> Path:
    """Carpeta de salida: `/kaggle/working` en Kaggle, `outputs/` del proyecto en local."""
    configured = os.environ.get(OUTPUT_DIR_ENV_VARIABLE)
    if configured:
        output_dir = Path(configured)
    elif KAGGLE_WORKING_DIR.exists():
        output_dir = KAGGLE_WORKING_DIR
    else:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_dataset_search_roots() -> dict[str, Path]:
    """Carpeta donde buscar los ficheros de cada split del dataset RFMiD.

    En Kaggle se busca en todo `/kaggle/input` (sirve cualquier copia adjunta de RFMiD).
    En local se usan las carpetas `Training_Set`, `Evaluation_Set` y `Test_Set` del repositorio.
    """
    splits = tuple(LOCAL_SPLIT_FOLDERS)
    configured = os.environ.get(DATA_ROOT_ENV_VARIABLE)
    if configured:
        return {split: Path(configured) for split in splits}
    if is_kaggle():
        return {split: KAGGLE_INPUT_DIR for split in splits}

    for folder in [Path.cwd(), *Path.cwd().parents]:
        if (folder / LOCAL_SPLIT_FOLDERS["train"]).is_dir():
            return {split: folder / LOCAL_SPLIT_FOLDERS[split] for split in splits}
    raise FileNotFoundError(
        "No se encontró el dataset RFMiD. En Kaggle, añade el dataset al notebook "
        "(Add Input); en local, ejecuta desde una carpeta que contenga "
        f"'{LOCAL_SPLIT_FOLDERS['train']}/' o define {DATA_ROOT_ENV_VARIABLE}."
    )


def locate_artifact(filename: str, output_dir: Path | None = None) -> Path:
    """Busca un fichero producido por un notebook anterior.

    Orden: carpeta de salida actual y después cualquier input adjunto de Kaggle
    (por ejemplo "Add Input -> Notebook Output" del notebook 02).
    """
    output_dir = output_dir or resolve_output_dir()
    candidate = output_dir / filename
    if candidate.exists():
        return candidate
    if KAGGLE_INPUT_DIR.exists():
        matches = sorted(KAGGLE_INPUT_DIR.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No se encontró '{filename}'. Ejecuta antes el notebook que lo genera o, en Kaggle, "
        "añade su salida como input (Add Input -> Notebook Output) o sube el fichero como dataset."
    )


def ensure_packages(required: list[str], optional: list[str] | None = None) -> None:
    """Importa los paquetes necesarios e intenta instalarlos con pip si faltan.

    Los paquetes opcionales (p. ej. captum) solo avisan si no se pueden instalar.
    """
    for package in required:
        _import_or_install(package, mandatory=True)
    for package in optional or []:
        _import_or_install(package, mandatory=False)


def _import_or_install(package: str, mandatory: bool) -> None:
    try:
        importlib.import_module(package)
        return
    except ImportError:
        pass
    print(f"Instalando '{package}'...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", package],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        importlib.invalidate_caches()
        return
    message = f"No se pudo instalar '{package}': {result.stderr[-300:]}"
    if mandatory:
        raise ImportError(message)
    print(f"AVISO (opcional): {message}")


def resolve_hf_token(secret_name: str = "HF_TOKEN") -> str | None:
    """Obtiene el token de Hugging Face desde un Kaggle Secret o una variable de entorno.

    Nunca se imprime ni se guarda; devuelve None si no hay token.
    """
    try:
        from kaggle_secrets import UserSecretsClient  # solo existe en Kaggle

        return UserSecretsClient().get_secret(secret_name)
    except Exception:
        return os.environ.get(secret_name) or None
