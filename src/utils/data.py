from pathlib import Path
from typing import List, Tuple
import soundfile as sf

def pair_files(folder: Path, audio_ext: str = ".wav", text_ext: str = ".txt") -> List[Tuple[Path, Path]]:
    """Return sorted list of (audio_path, text_path) pairs where both files exist."""
    pairs = []
    for audio in sorted(folder.glob(f"*{audio_ext}")):
        txt = audio.with_suffix(text_ext)
        if txt.exists():
            pairs.append((audio, txt))
    return pairs

def load_audio(path: Path):
    data, sr = sf.read(str(path), always_2d=False)
    return data, sr

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()
