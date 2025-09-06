from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn.functional as F
import librosa
from tqdm import trange
from transformers import WhisperProcessor, WhisperForConditionalGeneration, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from src.config import DATA_DIR, BASE_MODEL_ID, ASR_MODEL_DIR, MAX_MEL_FRAMES


def pair_files(folder: Path):
    pairs = []
    for wav in sorted(folder.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.exists():
            pairs.append((wav, txt))
    return pairs

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()

SR = 16000

class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], processor: WhisperProcessor):
        self.pairs = pairs
        self.processor = processor
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        wav, txt = self.pairs[idx]
        y, _ = librosa.load(str(wav), sr=SR, mono=True)
        inputs = self.processor(y, sampling_rate=SR, return_tensors="pt")
        input_features = inputs.input_features[0]
        T = input_features.size(-1)
        if T < MAX_MEL_FRAMES:
            input_features = F.pad(input_features, (0, MAX_MEL_FRAMES - T))
        else:
            input_features = input_features[:, :MAX_MEL_FRAMES]
        labels = self.processor.tokenizer(
            read_text(txt),
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids[0]
        labels = labels.to(torch.long)
        return {"input_features": input_features, "labels": labels}

def collate_fn(batch):
    feats = torch.stack([b["input_features"] for b in batch], dim=0)
    labels = [b["labels"] for b in batch]
    max_len = max(x.size(0) for x in labels)
    padded = torch.full((len(labels), max_len), fill_value=-100, dtype=torch.long)
    for i, x in enumerate(labels):
        padded[i, :x.size(0)] = x
    return {"input_features": feats, "labels": padded}

def main():
    pairs = pair_files(Path(DATA_DIR))
    if not pairs:
        print("No .wav/.txt pairs in DATA_DIR.")
        return
    try:
        processor = WhisperProcessor.from_pretrained(str(ASR_MODEL_DIR))
        model = WhisperForConditionalGeneration.from_pretrained(str(ASR_MODEL_DIR))
    except Exception:
        processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, language="pa", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="pa", task="transcribe")
    ds = PairDataset(pairs, processor)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    total_steps = len(dl) * 1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps//10), num_training_steps=total_steps)
    pbar = trange(total_steps, desc="training")
    step = 0
    for epoch in range(1):
        for batch in dl:
            if step >= total_steps:
                break
            feats = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_features=feats, labels=labels)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": float(loss.detach().cpu())})
            pbar.update(1)
            step += 1
    out_dir = Path("checkpoints") / "trained_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()
