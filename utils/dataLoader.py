import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import math
import datetime
#max_seq_lengthとbatch_sizeはhydraで管理したい。

def generate_data_loader(label_map, tokenizer, max_seq_length, batch_size, train_label,
                         train_unlabel, do_shuffle=False, balance_label_examples=False):

    # The labeled (train) dataset is assigned with a mask set to True
    label_masks = np.ones(len(train_label), dtype=bool)

    # If unlabel examples are available
    if len(train_unlabel) > 0:
        train_examples = train_label + train_unlabel
        # The unlabeled (train) dataset is assigned with a mask set to False
        tmp_masks = np.zeros(len(train_unlabel), dtype=bool)
        label_masks = np.concatenate([label_masks, tmp_masks])
        # label_masksはラベルがあるかないかを表すブール値、[True,True,False]ならラベル付き2つラベルなし一つ


    '''
    Generate a Dataloader given the input examples, eventually masked if they are
    to be considered NOT labeled.
    '''
    examples = []

    # Count the percentage of labeled examples
    num_labeled_examples = 0
    for label_mask in label_masks:
        if label_mask:
            num_labeled_examples += 1
    label_mask_rate = num_labeled_examples / len(train_label)

    # if required it applies the balance
    for index, ex in enumerate(train_label):
        if label_mask_rate == 1 or not balance_label_examples:
            examples.append((ex, label_masks[index]))
            #ex:（text, 正解label),label_masks[index]はlabelありかなしかのbool
        else:
            # balance>1ならラベル付きデータを複製
            if label_masks[index]:
                balance = int(1 / label_mask_rate)
                balance = int(math.log(balance, 2))
                if balance < 1:
                    balance = 1
                for b in range(0, int(balance)):
                    examples.append((ex, label_masks[index]))
            else:
                examples.append((ex, label_masks[index]))

    # -----------------------------------------------
    # Generate input examples to the Transformer
    # -----------------------------------------------
    input_ids = []  # textを入力可能な数字に変換したもの
    input_mask_array = []  # paddingしていない部分を知らせる
    label_mask_array = []  # textにlabelがあるかないかを知らせる配列
    label_id_array = []  # textに割り当てられたlabelの辞書の番号を知らせる配列
    # ex:[text,label_name], examples:(ex, label_mask)
    # Tokenization
    for (text, label_mask) in examples:
        encoded_sent = tokenizer.encode(text[0], add_special_tokens=True, max_length=max_seq_length,
                                        padding="max_length", truncation=True)
        input_ids.append(encoded_sent)
        label_id_array.append(label_map[text[1]])
        label_mask_array.append(label_mask)

    # Attention to token (to ignore padded input wordpieces)
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        input_mask_array.append(att_mask)

    # Convertion to Tensor
    input_ids = torch.tensor(input_ids)
    input_mask_array = torch.tensor(input_mask_array)
    label_id_array = torch.tensor(label_id_array, dtype=torch.long)
    label_mask_array = torch.tensor(label_mask_array)

    # Building the TensorDataset
    dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)

    if do_shuffle:
        sampler = RandomSampler
    else:
        sampler = SequentialSampler

    # Building the DataLoader
    return DataLoader(
        dataset,  # The training samples.
        sampler=sampler(dataset),
        batch_size=batch_size)  # Trains with this batch size.
