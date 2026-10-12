"""
Validate dataset.py + vocabulary.py against a *real* COCO-format annotation
file (proper images[]/annotations[] JSON schema, as pycocotools expects) and
real JPEG images, then run one real training step through the full model.

Full MS COCO (train2014.zip is ~13GB of images) cannot be downloaded inside
this sandboxed environment, so this uses a hand-built mini annotation file
(captions_mini.json) pointing at 3 real images, exercising the exact same
code path (pycocotools COCO() parsing, Vocabulary.build, DataLoader batching,
model forward/backward) that will run against the full dataset.
"""
import torch
from config import Config
from model import ShowAndTell
from dataset import build_vocab_from_coco, get_loader

cfg = Config()
cfg.image_dir_train = '/home/claude/mini_coco/images'
cfg.caption_file_train = '/home/claude/mini_coco/captions_mini.json'
cfg.vocabulary_size = 60
cfg.batch_size = 3
cfg.max_caption_length = 12
cfg.pretrained_cnn = False  # no internet access to ImageNet weights in this sandbox

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('--- Building vocabulary from real COCO-format captions ---')
vocab = build_vocab_from_coco(cfg.caption_file_train, cfg.vocabulary_size)
print(f'Vocabulary size: {len(vocab)}')
print('Sample words:', list(vocab.word2idx.keys())[:15])

print('\n--- Testing encode/decode roundtrip ---')
sentence = "A group of people standing on a city street."
ids, mask = vocab.encode(sentence, cfg.max_caption_length)
print('original :', sentence)
print('token ids:', ids)
print('mask     :', mask)
print('decoded  :', vocab.decode(ids))

print('\n--- Loading real images + captions via DataLoader (pycocotools) ---')
loader = get_loader(cfg.image_dir_train, cfg.caption_file_train, vocab,
                     batch_size=cfg.batch_size, max_caption_length=cfg.max_caption_length,
                     image_size=cfg.image_size, train=True, num_workers=0)
print(f'Dataset size (num captions): {len(loader.dataset)}')

images, captions, masks = next(iter(loader))
print('images batch shape  :', tuple(images.shape))
print('captions batch shape:', tuple(captions.shape))
print('masks batch shape   :', tuple(masks.shape))

print('\n--- Running one real forward + backward pass through the model ---')
model = ShowAndTell(cfg, len(vocab)).to(device)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

images, captions, masks = images.to(device), captions.to(device), masks.to(device)
logits, preds = model(images, captions, teacher_forcing=True)
loss = ShowAndTell.masked_cross_entropy(logits[:, :-1, :], captions[:, 1:], masks[:, 1:])
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(f'loss on real mini-batch: {loss.item():.4f}')
print('backward pass OK on real images + real annotation JSON')

print('\nREAL DATA PIPELINE TEST PASSED.')
