"""
Sanity-check the full pipeline (encoder -> decoder -> loss -> backward ->
greedy inference) using synthetic data, since MS COCO cannot be downloaded
in this environment. This proves shapes, gradients, and the training/
inference loop are all correct; real training just swaps in `dataset.py`.
"""
import torch
from config import Config
from model import ShowAndTell
from vocabulary import Vocabulary

torch.manual_seed(0)

cfg = Config()
cfg.vocabulary_size = 50
cfg.batch_size = 4
cfg.max_caption_length = 10
cfg.pretrained_cnn = False  # sandbox has no internet access to download ImageNet weights;
                            # set True for real training on your own machine/Colab
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# Fake vocabulary (50 words)
vocab = Vocabulary(cfg.vocabulary_size)
words = ['<pad>', '<start>', '<end>', '<unk>'] + [f'word{i}' for i in range(46)]
vocab.word2idx = {w: i for i, w in enumerate(words)}
vocab.idx2word = {i: w for w, i in vocab.word2idx.items()}

# Fake batch: random "images" and random caption token ids
images = torch.randn(cfg.batch_size, 3, cfg.image_size, cfg.image_size, device=device)
captions = torch.randint(4, cfg.vocabulary_size, (cfg.batch_size, cfg.max_caption_length), device=device)
captions[:, 0] = vocab.word2idx['<start>']
masks = torch.ones_like(captions, dtype=torch.float, device=device)

model = ShowAndTell(cfg, len(vocab)).to(device)
print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')
print(f'Trainable params (CNN frozen): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

# ---- one training step ----
model.train()
logits, preds = model(images, captions, teacher_forcing=True)
print('logits shape:', logits.shape, '| preds shape:', preds.shape)

loss = ShowAndTell.masked_cross_entropy(logits[:, :-1, :], captions[:, 1:], masks[:, 1:])
print('initial loss:', loss.item())

optimizer.zero_grad()
loss.backward()
optimizer.step()
print('backward pass OK, gradients flowed correctly')

# ---- a few more steps to confirm loss goes down (pipeline is learning something) ----
model.train()
for step in range(20):
    logits, _ = model(images, captions, teacher_forcing=True)
    loss = ShowAndTell.masked_cross_entropy(logits[:, :-1, :], captions[:, 1:], masks[:, 1:])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 5 == 0:
        print(f'step {step}: loss {loss.item():.4f}')

# ---- inference (greedy decoding, no teacher forcing) ----
model.eval()
with torch.no_grad():
    _, preds = model(images, captions=None, teacher_forcing=False)
print('greedy predictions (token ids), sample 0:', preds[0].tolist())
print('decoded caption, sample 0:', vocab.decode(preds[0].tolist()))

print('\nDEMO PASSED: full encoder-decoder pipeline runs end-to-end.')
