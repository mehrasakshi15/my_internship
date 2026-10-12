"""
Fast training for a SMALL dataset (like COCO128's ~100-600 captions).

Why this exists: with only ~128 images and the default 5000-word vocabulary,
20 epochs at lr=1e-4 is not enough signal for the decoder to move past
predicting the most frequent words ("a", "of", "and" ...). This script:
  1) Uses a SMALLER vocabulary (matched to how much data you actually have)
  2) Extracts each image's pretrained-CNN embedding ONCE and caches it
     (the CNN is frozen, so this avoids re-running VGG16 every epoch --
     makes 300+ epochs finish in a couple of minutes on CPU)
  3) Trains the decoder alone, much longer, with a slightly higher LR

Usage (run from your show_and_tell_pytorch folder):
    python quick_train_local.py --image_dir ../coco128/images/train2017 \
        --caption_file ../captions_coco128.json --epochs 300
"""
import argparse
import json
import os

import torch
from PIL import Image

from config import Config
from model import ShowAndTell
from vocabulary import Vocabulary
from dataset import build_transform

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', required=True)
parser.add_argument('--caption_file', required=True)
parser.add_argument('--vocab_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=1200)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--max_len', type=int, default=16)
args = parser.parse_args()

cfg = Config()
cfg.vocabulary_size = args.vocab_size
cfg.max_caption_length = args.max_len
cfg.pretrained_cnn = True   # use real ImageNet weights (you have internet locally)
cfg.train_cnn = False       # frozen CNN -> embeddings can be cached
cfg.lstm_drop_rate = 0.0    # dropout hurts memorization on a small dataset -- turn it off

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ---- vocab + raw (filename, caption) pairs ----
with open(args.caption_file) as f:
    coco_json = json.load(f)
sentences = [ann['caption'] for ann in coco_json['annotations']]
vocab = Vocabulary(cfg.vocabulary_size).build(sentences)
vocab.save('./vocabulary_local.pkl')
print(f'Vocabulary size: {len(vocab)} (built from {len(sentences)} real captions)')

id_to_file = {im['id']: im['file_name'] for im in coco_json['images']}
samples = []
for ann in coco_json['annotations']:
    fname = id_to_file[ann['image_id']]
    ids, mask = vocab.encode(ann['caption'], cfg.max_caption_length)
    samples.append((fname, ids, mask))
print(f'Total (image, caption) training pairs: {len(samples)}')

# ---- model + cached pretrained-CNN embeddings (computed once) ----
model = ShowAndTell(cfg, len(vocab)).to(device)
transform = build_transform(cfg.image_size, train=False)

unique_files = sorted(set(s[0] for s in samples))
print(f'Precomputing pretrained CNN embeddings for {len(unique_files)} unique images (once)...')
model.eval()
cached = {}
with torch.no_grad():
    for i, fname in enumerate(unique_files):
        path = os.path.join(args.image_dir, fname)
        img = Image.open(path).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        cached[fname] = model.encoder(x).squeeze(0)
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(unique_files)} done')

image_embs = torch.stack([cached[fn] for fn, _, _ in samples]).to(device)
captions = torch.tensor([ids for _, ids, _ in samples], dtype=torch.long, device=device)
masks = torch.tensor([m for _, _, m in samples], dtype=torch.float, device=device)

optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)

print(f'\nTraining decoder for {args.epochs} epochs on cached embeddings...')
model.decoder.train()
for epoch in range(args.epochs):
    logits, _ = model.decoder(image_embs, captions, cfg.max_caption_length, teacher_forcing=True)
    loss = ShowAndTell.masked_cross_entropy(logits, captions, masks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0 or epoch == args.epochs - 1:
        print(f'epoch {epoch}: loss {loss.item():.4f}')

torch.save({'model_state': model.state_dict(), 'vocab_file': './vocabulary_local.pkl'},
           './models/quick_model.pt')
print('\nSaved checkpoint: ./models/quick_model.pt')

print('\nSample generated captions on training images:')
model.decoder.eval()
with torch.no_grad():
    for fname in unique_files[:8]:
        emb = cached[fname].unsqueeze(0)
        _, preds = model.decoder(emb, None, cfg.max_caption_length, teacher_forcing=False)
        print(f'  {fname}: {vocab.decode(preds[0].tolist())}')