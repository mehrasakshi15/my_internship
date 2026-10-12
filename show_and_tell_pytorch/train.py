"""
Training script.

Usage:
    python train.py --phase train
    python train.py --phase eval --model_file ./models/model_epoch10.pt
    python train.py --phase test  --model_file ./models/model_epoch10.pt --image_dir ./test/images
"""
import argparse
import os
import torch
from torch.optim import Adam

from config import Config
from model import ShowAndTell
from dataset import get_loader, build_vocab_from_coco
from vocabulary import Vocabulary


def train(config):
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if os.path.exists(config.vocab_file):
        vocab = Vocabulary(config.vocabulary_size).load(config.vocab_file)
    else:
        print('Building vocabulary from training captions...')
        vocab = build_vocab_from_coco(config.caption_file_train, config.vocabulary_size)
        os.makedirs(os.path.dirname(config.vocab_file), exist_ok=True)
        vocab.save(config.vocab_file)

    train_loader = get_loader(config.image_dir_train, config.caption_file_train, vocab,
                               batch_size=config.batch_size,
                               max_caption_length=config.max_caption_length,
                               image_size=config.image_size, train=True)

    model = ShowAndTell(config, len(vocab)).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable, lr=config.initial_learning_rate,
                      betas=(config.beta1, config.beta2), eps=config.epsilon,
                      weight_decay=config.weight_decay)

    os.makedirs(config.save_dir, exist_ok=True)
    step = 0
    for epoch in range(config.num_epochs):
        model.train()
        for images, captions, masks in train_loader:
            images, captions, masks = images.to(device), captions.to(device), masks.to(device)

            logits, _ = model(images, captions, teacher_forcing=True)
            # step t sees the image + tokens[:t-1] and predicts tokens[t] -> NO shift needed
            # (step 0 sees only the image and learns to emit <start>)
            loss = ShowAndTell.masked_cross_entropy(logits, captions, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.clip_gradients)
            optimizer.step()

            if step % config.log_every == 0:
                print(f'epoch {epoch} step {step} loss {loss.item():.4f}')
            step += 1

        ckpt_path = os.path.join(config.save_dir, f'model_epoch{epoch}.pt')
        torch.save({'model_state': model.state_dict(), 'vocab_file': config.vocab_file}, ckpt_path)
        print(f'Saved checkpoint: {ckpt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train', 'eval', 'test'])
    args = parser.parse_args()
    cfg = Config()
    if args.phase == 'train':
        train(cfg)
    else:
        print('Use inference.py for eval/test phases (BLEU evaluation / caption generation).')