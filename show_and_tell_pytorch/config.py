"""
Configuration for the PyTorch reproduction of "Show and Tell: A Neural
Image Caption Generator" (Vinyals et al., 2015).

Mirrors the hyperparameters used in the reference TensorFlow implementation
(https://github.com/nikhilmaram/Show_and_Tell/blob/master/config.py) so that
results are comparable.
"""

class Config(object):
    # ---------------- Data ----------------
    image_dir_train = '../coco128/images/train2017'
    caption_file_train = '../captions_coco128.json'
    image_dir_val = './data/val2014/images'
    caption_file_val = './data/val2014/captions_val2014.json'
    vocab_file = './data/vocabulary.pkl'

    image_size = 224              # VGG16 input size
    max_caption_length = 20       # sentences truncated/padded to this length
    vocabulary_size = 5000        # top-K most frequent words kept

    # ---------------- CNN Encoder ----------------
    cnn = 'vgg16' 
    pretrained_cnn = True
    train_cnn = False             # if True, fine-tune CNN jointly with LSTM
    cnn_feature_dim = 512         # channel dim of the feature map fed to LSTM
    num_ctx = 8                   # "spatial" positions after reshaping fc7 (8 x 512 = 4096)

    # ---------------- RNN Decoder ----------------
    dim_embedding = 512
    num_lstm_units = 512
    num_lstm_layers = 1
    lstm_drop_rate = 0.3
    num_decode_layers = 1          # reference repo actually uses 1 (commented "changed from 2 to 1")
    dim_decode_layer = 1024

    # ---------------- Training ----------------
    batch_size = 32
    num_epochs = 20
    optimizer = 'Adam'
    initial_learning_rate = 1e-4
    learning_rate_decay_factor = 0.9
    num_steps_per_decay = 5000
    clip_gradients = 5.0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-6
    weight_decay = 1e-4

    # ---------------- Misc ----------------
    save_dir = './models'
    log_every = 50
    seed = 42
    device = 'cuda'  # falls back to 'cpu' automatically if no GPU is found
