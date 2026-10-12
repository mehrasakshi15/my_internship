"""
MS COCO caption dataset loader (PyTorch Dataset/DataLoader), mirroring
`prepare_train_data` / `prepare_eval_data` in the reference repo's dataset.py.

Expects the standard COCO layout:
    <image_dir>/COCO_train2014_000000XXXXXX.jpg
    <caption_file> = captions_train2014.json  (COCO annotation format)

Download instructions (http://cocodataset.org/#download):
    train2014.zip, val2014.zip                 (images)
    annotations_trainval2014.zip                (captions_train2014.json, captions_val2014.json)
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from pycocotools.coco import COCO

from vocabulary import Vocabulary

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size, train=True):
    if train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CocoCaptionDataset(Dataset):
    def __init__(self, image_dir, caption_file, vocab: Vocabulary,
                 max_caption_length=20, image_size=224, train=True):
        self.image_dir = image_dir
        self.coco = COCO(caption_file)
        self.ann_ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.max_caption_length = max_caption_length
        self.transform = build_transform(image_size, train)

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, idx):
        ann_id = self.ann_ids[idx]
        ann = self.coco.anns[ann_id]
        caption = ann['caption']
        image_id = ann['image_id']
        file_name = self.coco.loadImgs(image_id)[0]['file_name']

        image = Image.open(os.path.join(self.image_dir, file_name)).convert('RGB')
        image = self.transform(image)

        ids, mask = self.vocab.encode(caption, self.max_caption_length)
        return image, torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float)


def build_vocab_from_coco(caption_file, vocabulary_size=5000):
    coco = COCO(caption_file)
    sentences = [ann['caption'] for ann in coco.anns.values()]
    return Vocabulary(vocabulary_size).build(sentences)


def get_loader(image_dir, caption_file, vocab, batch_size=32, max_caption_length=20,
               image_size=224, train=True, num_workers=4):
    dataset = CocoCaptionDataset(image_dir, caption_file, vocab, max_caption_length,
                                  image_size, train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                       num_workers=num_workers, drop_last=train)
