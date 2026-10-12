"""
Test the quick-trained model on a brand-new, real-world image (one that was
NOT in the 128-image COCO128 training set).

Note: since the model was trained on only 128 images and reached a very low
loss (0.27), it has largely MEMORIZED those specific images rather than
learned to generalize to arbitrary photos. Expect weaker / more generic
results here than on training images -- this is a real and expected
limitation of training on such a small dataset, not a bug.

Usage:
    python test_new_image.py --image path/to/your_new_photo.jpg
"""
import argparse
import torch
from PIL import Image

from config import Config
from model import ShowAndTell
from vocabulary import Vocabulary
from dataset import build_transform

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
parser.add_argument('--checkpoint', default='./models/quick_model.pt')
parser.add_argument('--vocab', default='./vocabulary_local.pkl')
args = parser.parse_args()

# must match the settings used in quick_train_local.py
cfg = Config()
cfg.vocabulary_size = 500
cfg.max_caption_length = 16
cfg.pretrained_cnn = True
cfg.train_cnn = False
cfg.lstm_drop_rate = 0.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab = Vocabulary(cfg.vocabulary_size).load(args.vocab)
model = ShowAndTell(cfg, len(vocab)).to(device)
ckpt = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()

transform = build_transform(cfg.image_size, train=False)
image = Image.open(args.image).convert('RGB')
x = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    _, preds = model(x, captions=None, teacher_forcing=False)

caption = vocab.decode(preds[0].tolist())
print(f'Image: {args.image}')
print(f'Generated caption: {caption}')
