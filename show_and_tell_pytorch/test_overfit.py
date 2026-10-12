"""
Train for many epochs on the 3-image mini real dataset and check the model
can overfit / reproduce captions close to training data -- the standard
"can it memorize a tiny dataset" sanity check before full-scale COCO
training. Since the CNN is frozen (no pretrained weights available in this
sandbox), image embeddings are precomputed once and cached, so only the
decoder is actually trained here -- this keeps the demo fast on CPU while
still exercising the exact same forward/backward code path used for full
training.
"""
import torch
from PIL import Image

from config import Config
from model import ShowAndTell
from dataset import build_vocab_from_coco, build_transform

cfg = Config()
cfg.caption_file_train = '/home/claude/mini_coco/captions_mini.json'
cfg.image_dir_train = '/home/claude/mini_coco/images'
cfg.vocabulary_size = 60
cfg.max_caption_length = 12
cfg.pretrained_cnn = False   # no internet access to ImageNet weights in this sandbox
cfg.train_cnn = False        # CNN frozen -> embeddings can be cached

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- vocab + raw captions (image_id -> list of (ids, mask)) ----
vocab = build_vocab_from_coco(cfg.caption_file_train, cfg.vocabulary_size)
import json
with open(cfg.caption_file_train) as f:
    coco_json = json.load(f)
id_to_file = {im['id']: im['file_name'] for im in coco_json['images']}
samples = []  # (file_name, ids, mask)
for ann in coco_json['annotations']:
    ids, mask = vocab.encode(ann['caption'], cfg.max_caption_length)
    samples.append((id_to_file[ann['image_id']], ids, mask))
print(f'{len(samples)} (image, caption) training pairs, vocab size {len(vocab)}')

# ---- model + cached frozen-CNN embeddings (computed once) ----
model = ShowAndTell(cfg, len(vocab)).to(device)
transform = build_transform(cfg.image_size, train=False)

print('Precomputing frozen CNN embeddings for the 3 training images (once)...')
model.eval()
cached = {}
with torch.no_grad():
    for fname in set(s[0] for s in samples):
        img = Image.open(f'{cfg.image_dir_train}/{fname}').convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        cached[fname] = model.encoder(x).squeeze(0)

image_embs = torch.stack([cached[fn] for fn, _, _ in samples]).to(device)
captions = torch.tensor([ids for _, ids, _ in samples], dtype=torch.long, device=device)
masks = torch.tensor([m for _, _, m in samples], dtype=torch.float, device=device)

optimizer = torch.optim.Adam(model.decoder.parameters(), lr=3e-4)

print('Overfitting decoder on cached embeddings for 300 steps...')
model.decoder.train()
for step in range(300):
    logits, _ = model.decoder(image_embs, captions, cfg.max_caption_length, teacher_forcing=True)
    loss = ShowAndTell.masked_cross_entropy(logits[:, :-1, :], captions[:, 1:], masks[:, 1:])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0 or step == 299:
        print(f'step {step}: loss {loss.item():.4f}')

print('\nGenerating captions on the (memorized) training images:')
model.decoder.eval()
with torch.no_grad():
    for fname in ['1.jpg', '2.jpg', '3.jpg']:
        emb = cached[fname].unsqueeze(0)
        _, preds = model.decoder(emb, None, cfg.max_caption_length, teacher_forcing=False)
        print(f'  {fname}: {vocab.decode(preds[0].tolist())}')

print('\nGround-truth captions for reference:')
for fname, ids, _ in samples:
    print(f'  {fname}: {vocab.decode(ids)}')
