"""
Generate a caption for a single image using a trained checkpoint.

Usage:
    python inference.py --image path/to/image.jpg --checkpoint ./models/model_epoch10.pt
"""
import argparse
import torch
from PIL import Image

from config import Config
from model import ShowAndTell
from vocabulary import Vocabulary
from dataset import build_transform


def generate_caption(image_path, checkpoint_path, vocab_path, config=None):
    config = config or Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocabulary(config.vocabulary_size).load(vocab_path)
    model = ShowAndTell(config, len(vocab)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    transform = build_transform(config.image_size, train=False)
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        _, preds = model(image, captions=None, teacher_forcing=False)

    caption = vocab.decode(preds[0].tolist())
    return caption


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--vocab', default='./data/vocabulary.pkl')
    args = parser.parse_args()
    caption = generate_caption(args.image, args.checkpoint, args.vocab)
    print('Generated caption:', caption)
