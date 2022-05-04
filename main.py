import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertModel, AutoTokenizer, AutoConfig

from utils.dataLoader import generate_data_loader
from models.ganbert import Generator, Discriminator
from src.train import train

import io
import random
import numpy as np
import pandas as pd
import time

import logging


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.debug('This is debug')

    #verify GPU or CPU settings
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # --------------------------------
    #  Transformer parameters
    # --------------------------------
    max_seq_length = 100
    batch_size = 4

    # --------------------------------
    #  GAN-BERT specific parameters
    # --------------------------------
    # number of hidden layers in the generator,
    # each of the size of the output space
    num_hidden_layers_g = 1
    # number of hidden layers in the discriminator,
    # each of the size of the input space
    num_hidden_layers_d = 1
    # size of the generator's input noisy vectors
    noise_size = 100
    # dropout to be applied to discriminator's input vectors
    out_dropout_rate = 0.2

    # Replicate labeled data to balance poorly represented datasets,
    # e.g., less than 1% of labeled material
    apply_balance = True

    # --------------------------------
    #  Optimization parameters
    # --------------------------------
    learning_rate_discriminator = 1e-5
    learning_rate_generator = 1e-5
    epsilon = 1e-8
    num_train_epochs = 10
    multi_gpu = True
    # Scheduler
    apply_scheduler = False
    warmup_proportion = 0.1
    # Print
    print_each_n_step = 10

    # --------------------------------
    #  Adopted Tranformer model
    # --------------------------------
    # Since this version is compatible with Huggingface transformers, you can uncomment
    # (or add) transformer models compatible with GAN

    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    # model_name = "bert-base-cased"
    # model_name = "bert-base-uncased"
    # model_name = "roberta-base"
    # model_name = "albert-base-v2"
    # model_name = "xlm-roberta-base"
    # model_name = "amazon/bort"

    #model loading
    transformers = BertModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #data loading
    train_label = pd.read_csv('data/news_train_label.csv', header=None).values.tolist()
    train_unlabel = pd.read_csv('data/news_train_unlabel.csv', header=None).values.tolist()
    test_label = pd.read_csv('data/news_test.csv', header=None).values.tolist()
    test_unlabel = []

    label_list = ['UNK_UNK', 8, 7, 6, 5, 4, 3, 2, 1, 0]
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    #train_dataloader = generate_data_loader(label_map, train_label, train_unlabel, do_shuffle=True,
    #                                        balance_label_examples=apply_balance)
    train_dataloader = generate_data_loader(label_map, tokenizer, max_seq_length, batch_size, train_label,
                         train_unlabel, do_shuffle=False, balance_label_examples=False)
    #test_dataloader = generate_data_loader(label_map, test, test_unlabel=False, do_shuffle=False,
    #                                       balance_label_examples=False)
    test_dataloader = generate_data_loader(label_map, tokenizer, max_seq_length, batch_size, test_label,
                         test_unlabel, do_shuffle=False, balance_label_examples=False)

    # The config file is required to get the dimension of the vector produced by
    # the underlying transformer
    config = AutoConfig.from_pretrained(model_name)
    hidden_size = int(config.hidden_size)
    # Define the number and width of hidden layers
    hidden_levels_g = [hidden_size for i in range(0, num_hidden_layers_g)]
    hidden_levels_d = [hidden_size for i in range(0, num_hidden_layers_d)]

    # -------------------------------------------------
    #   Instantiate the Generator and Discriminator
    # -------------------------------------------------
    generator = Generator(noise_size=noise_size, output_size=hidden_size, hidden_sizes=hidden_levels_g,
                          dropout_rate=out_dropout_rate)
    discriminator = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_d, num_labels=len(label_list),
                                  dropout_rate=out_dropout_rate)

    # Put everything in the GPU if available
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        transformers.cuda()
        if multi_gpu:
            transformer = torch.nn.DataParallel(transformers)

    train(transformers, discriminator, generator, train_dataloader, test_dataloader, device,
          num_train_epochs, print_each_n_step, learning_rate_discriminator, learning_rate_generator,
          apply_scheduler, batch_size, warmup_proportion, noise_size, epsilon, label_list)

    logger.debug("This is debug")
