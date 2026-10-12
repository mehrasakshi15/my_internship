"""
PyTorch reproduction of the Show and Tell architecture.

Encoder: VGG16 (torchvision, pretrained on ImageNet) -> fc7 (4096-d) ->
         reshaped to [batch, num_ctx=8, cnn_feature_dim=512] -> mean pooled
         to a single 512-d image embedding (this mirrors build_vgg16 +
         reduce_mean in the reference model.py, where the 8x512 "pseudo
         spatial" features are averaged before entering the LSTM).

Decoder: single-layer LSTMCell. Step 0 receives the image embedding as
         input (no word). From step 1 onward it receives the embedding of
         the previous word (teacher forcing at train time, its own greedy
         prediction at inference time). Each LSTM output is decoded to
         vocabulary logits through 2 FC layers (tanh -> linear), matching
         `decode()` in the reference implementation.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, cnn_feature_dim=512, num_ctx=8, train_cnn=False, pretrained=True):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg16(weights=weights)
        # Keep convolutional feature extractor + first two FC layers (fc6, fc7),
        # drop the final 1000-way ImageNet classifier (fc8) -- same idea as
        # the reference code, which only uses fc2 (=fc7 output) as features.
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc6 = vgg.classifier[0]
        self.relu6 = vgg.classifier[1]
        self.fc7 = vgg.classifier[3]
        self.relu7 = vgg.classifier[4]

        self.cnn_feature_dim = cnn_feature_dim
        self.num_ctx = num_ctx

        if not train_cnn:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, images):
        x = self.features(images)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu6(self.fc6(x))
        x = self.relu7(self.fc7(x))          # [batch, 4096]
        x = x.view(x.size(0), self.num_ctx, self.cnn_feature_dim)  # [batch, 8, 512]
        image_emb = x.mean(dim=1)             # [batch, 512]  (mirrors tf.reduce_mean)
        return image_emb


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, dim_embedding=512, num_lstm_units=512,
                 num_decode_layers=2, dim_decode_layer=1024, lstm_drop_rate=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_embedding, padding_idx=0)
        self.lstm_cell = nn.LSTMCell(dim_embedding, num_lstm_units)
        self.dropout = nn.Dropout(lstm_drop_rate)

        if num_decode_layers == 1:
            self.decode = nn.Sequential(nn.Linear(num_lstm_units, vocab_size))
        else:
            self.decode = nn.Sequential(
                nn.Linear(num_lstm_units, dim_decode_layer),
                nn.Tanh(),
                nn.Dropout(lstm_drop_rate),
                nn.Linear(dim_decode_layer, vocab_size),
            )
        self.num_lstm_units = num_lstm_units

    def init_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.num_lstm_units, device=device)
        c = torch.zeros(batch_size, self.num_lstm_units, device=device)
        return h, c

    def forward(self, image_emb, captions=None, max_len=20, teacher_forcing=True):
        """
        image_emb: [batch, dim_embedding]  (from EncoderCNN)
        captions:  [batch, max_len] token ids, required if teacher_forcing=True
        Returns logits: [batch, max_len, vocab_size], predictions: [batch, max_len]
        """
        batch_size = image_emb.size(0)
        device = image_emb.device
        h, c = self.init_state(batch_size, device)

        logits_list, preds_list = [], []
        last_input = image_emb  # step 0 input is the image embedding itself

        for t in range(max_len):
            h, c = self.lstm_cell(last_input, (h, c))
            logits = self.decode(self.dropout(h))
            pred = logits.argmax(dim=1)
            logits_list.append(logits)
            preds_list.append(pred)

            if teacher_forcing and captions is not None:
                next_word = captions[:, t]
            else:
                next_word = pred
            last_input = self.embedding(next_word)

        logits = torch.stack(logits_list, dim=1)
        preds = torch.stack(preds_list, dim=1)
        return logits, preds


class ShowAndTell(nn.Module):
    """Wraps encoder + decoder, analogous to CaptionGenerator in the reference repo."""
    def __init__(self, config, vocab_size):
        super().__init__()
        pretrained = getattr(config, 'pretrained_cnn', True)
        self.encoder = EncoderCNN(config.cnn_feature_dim, config.num_ctx, config.train_cnn, pretrained)
        self.decoder = DecoderRNN(vocab_size, config.dim_embedding, config.num_lstm_units,
                                   config.num_decode_layers, config.dim_decode_layer,
                                   config.lstm_drop_rate)
        self.max_caption_length = config.max_caption_length

    def forward(self, images, captions=None, teacher_forcing=True):
        image_emb = self.encoder(images)
        logits, preds = self.decoder(image_emb, captions, self.max_caption_length, teacher_forcing)
        return logits, preds

    @staticmethod
    def masked_cross_entropy(logits, targets, masks):
        """logits: [B,T,V], targets: [B,T], masks: [B,T] (1=real token, 0=pad)."""
        B, T, V = logits.shape
        loss = nn.functional.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction='none')
        loss = loss.reshape(B, T) * masks
        return loss.sum() / masks.sum().clamp(min=1.0)
