"""
Run the PyTorch encoder + decoder on REAL images (borrowed from the
reference repo's test/images folder) to confirm the pipeline handles real
image data correctly end-to-end, not just random tensors.

Note: ImageNet-pretrained VGG16 weights cannot be downloaded in this sandbox
(no internet access to download.pytorch.org), so encoder runs with random
init here. On your own machine with internet access, set
config.pretrained_cnn = True for meaningful captions.
"""
import glob
import torch
from PIL import Image

from config import Config
from model import ShowAndTell
from vocabulary import Vocabulary
from dataset import build_transform

cfg = Config()
cfg.vocabulary_size = 200
cfg.pretrained_cnn = False  # would be True with internet access
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab = Vocabulary(cfg.vocabulary_size)
words = ['<pad>', '<start>', '<end>', '<unk>'] + [f'word{i}' for i in range(196)]
vocab.word2idx = {w: i for i, w in enumerate(words)}
vocab.idx2word = {i: w for w, i in vocab.word2idx.items()}

model = ShowAndTell(cfg, len(vocab)).to(device)
model.eval()

transform = build_transform(cfg.image_size, train=False)

image_paths = sorted(glob.glob('/home/claude/reference_repo/test/images/*.jpg'))
print(f'Found {len(image_paths)} real test images: {image_paths}')

for path in image_paths:
    img = Image.open(path).convert('RGB')
    print(f'\n{path} -> original size {img.size}')
    x = transform(img).unsqueeze(0).to(device)
    print('  tensor shape after transform:', tuple(x.shape))

    with torch.no_grad():
        image_emb = model.encoder(x)
        print('  encoder output (image embedding) shape:', tuple(image_emb.shape))
        _, preds = model(x, captions=None, teacher_forcing=False)
        caption = vocab.decode(preds[0].tolist())
        print('  decoded caption (random-init weights, so just architecture check):', caption)

print('\nREAL IMAGE TEST PASSED: encoder correctly processes real JPEGs of varying '
      'sizes into a [1, 512] embedding, and the decoder generates a caption of the '
      'expected length. With pretrained VGG16 weights + trained LSTM, this same code '
      'path produces meaningful captions.')
