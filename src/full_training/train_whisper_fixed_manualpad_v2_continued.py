import os
import torch
import torchaudio
import wandb
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ========== WANDB SETUP ==========
wandb.init(
    project="whisper-medium-kirtan-fixpadding",
    name="whisper-fixed-continue-to-20"
)

# ========== CONFIG ==========
MODEL_DIR = "checkpoints/epoch10"
DATA_DIR = "C:/Users/Gurjeevan/OneDrive/Desktop/Kirtan Corpus/Transcripts"
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10  # 10 more to reach epoch 20

# ========== LOAD PROCESSOR AND MODEL ==========
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)


# ========== CUSTOM DATASET ==========
class KirtanDataset(Dataset):
    def __init__(self, data_dir):
        self.paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav")])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio_path = self.paths[idx]
        txt_path = audio_path.replace(".wav", ".txt")

        # Load and resample audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        waveform = waveform.mean(dim=0)  # Mono


        input_features = \
        processor.feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="np").input_features[0]


        if input_features.shape[-1] < 3000:
            pad_width = 3000 - input_features.shape[-1]
            input_features = np.pad(input_features, ((0, 0), (0, pad_width)))
        else:
            input_features = input_features[:, :3000]


        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        labels = \
        processor.tokenizer(text, return_tensors="pt", padding="max_length", max_length=448, truncation=True).input_ids[
            0]

        return {
            "input_features": torch.tensor(input_features, dtype=torch.float32),
            "labels": labels
        }


# ========== COLLATE FUNCTION ==========
def collate_fn(batch):
    input_features = torch.stack([item["input_features"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_features": input_features, "labels": labels}


# ========== LOAD DATA ==========
dataset = KirtanDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ========== TRAINING ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=processor.tokenizer.pad_token_id)

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0

    for batch in dataloader:
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} (continued) - Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})

# ========== SAVE ==========
model.save_pretrained("checkpoints/epoch_20")
processor.save_pretrained("checkpoints/epoch_20")
print("✅ Training complete. Model saved to checkpoints/epoch20")
