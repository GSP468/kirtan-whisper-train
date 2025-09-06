from pathlib import Path
from dataclasses import dataclass
from typing import List
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer
from src.config import TEST_EVAL_DIR, ASR_MODEL_DIR

@dataclass
class Row:
    clip: str
    ref: str
    hyp: str
    WER: float
    CER: float

def transcribe(model, processor, audio, sr) -> str:
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(**inputs)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

def main():
    print(f"Evaluating in: {TEST_EVAL_DIR}")
    wavs = sorted(TEST_EVAL_DIR.glob("*.wav"))
    if not wavs:
        print("No .wav files found. Add .wav/.txt pairs to test_evaluate/ and retry.")
        return

    try:
        processor = WhisperProcessor.from_pretrained(str(ASR_MODEL_DIR))
        model = WhisperForConditionalGeneration.from_pretrained(str(ASR_MODEL_DIR))
    except Exception:
        processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="pa", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="pa", task="transcribe")

    rows: List[Row] = []
    for wav in wavs:
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            print(f"Warning: missing transcript for {wav.name}; skipping.")
            continue
        ref = txt.read_text(encoding="utf-8").strip()
        audio, sr = sf.read(str(wav), always_2d=False)
        hyp = transcribe(model, processor, audio, sr)
        rows.append(Row(clip=wav.stem, ref=ref, hyp=hyp, WER=wer(ref, hyp), CER=cer(ref, hyp)))

    from tabulate import tabulate
    table = [[r.clip, r.WER, r.CER] for r in rows]
    print(tabulate(table, headers=["Clip", "WER", "CER"], floatfmt=".3f"))

    if rows:
        avg_wer = sum(r.WER for r in rows)/len(rows)
        avg_cer = sum(r.CER for r in rows)/len(rows)
        print(f"\nAverages — WER: {avg_wer:.3f}, CER: {avg_cer:.3f}")

if __name__ == "__main__":
    main()
