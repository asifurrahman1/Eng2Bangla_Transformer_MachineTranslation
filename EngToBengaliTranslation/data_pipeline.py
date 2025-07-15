import tensorflow as tf
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from indicnlp.tokenize import indic_tokenize
from basetokenizer import BaseTokenizer
import pandas as pd

class DataPipeline:
    def __init__(self, dataset_path, src_lang='en', trgt_lang='bn', tokenizerclass = None):
        self.dataset_df =  pd.read_csv(dataset_path)
        self.SRC_LANGUAGE = src_lang
        self.TGT_LANGUAGE = trgt_lang
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.tokenizercls = tokenizerclass if tokenizerclass is not None else BaseTokenizer()
        self.token_transform = {self.SRC_LANGUAGE: self.tokenizercls.src_tokenizer,
                                self.TGT_LANGUAGE: self.tokenizercls.target_tokenizer
                               }

    def get_special_token_idx(self, token_name ='PAD'):
        if token_name.upper()=='UNK':
          return self.UNK_IDX
        elif token_name.upper()=='BOS':
          return self.BOS_IDX
        elif token_name.upper()=='EOS':
          return self.EOS_IDX
        else:
          return self.PAD_IDX

    def build_vocab(self, texts, tokenizer, min_freq=1):
        counter = Counter()
        for text in texts:
            counter.update(tokenizer(text))
        vocab = {tok: idx + len(self.special_symbols) for idx, (tok, freq) in enumerate(counter.items()) if freq >= min_freq}
        for i, sym in enumerate(self.special_symbols):
            vocab[sym] = i
        inv_vocab = {idx: tok for tok, idx in vocab.items()}
        return vocab, inv_vocab

    def encode_tokens(self, tokens, vocab, flip=False):
        ids = [vocab.get(tok, self.UNK_IDX) for tok in tokens]
        if flip:
            ids = [self.BOS_IDX] + list(reversed(ids)) + [self.EOS_IDX]
        else:
            ids = [self.BOS_IDX] + ids + [self.EOS_IDX]
        return ids

    def encode_pair(self, src, tgt, src_lang, tgt_lang, src_vocab, tgt_vocab, flip=False):
        src_tokens = self.token_transform[src_lang](src)
        tgt_tokens = self.token_transform[tgt_lang](tgt)
        src_ids = self.encode_tokens(src_tokens, src_vocab, flip=flip)
        tgt_ids = self.encode_tokens(tgt_tokens, tgt_vocab)
        return src_ids, tgt_ids

    def pad_sequence(self, seq, pad_idx, max_len=None):
        if max_len is None:
            return seq
        padded = seq + [pad_idx] * (max_len - len(seq))
        return padded[:max_len]

    def index_to_text(self, indices, inv_vocab):
        if isinstance(indices, tf.Tensor):
            indices = indices.numpy().flatten()
        return " ".join([inv_vocab.get(int(idx), '<unk>') for idx in indices if int(idx) != self.PAD_IDX])


    def prepare_tf_dataset_from_dataframe(self, df, src_lang, tgt_lang, src_vocab, tgt_vocab, batch_size=4, flip=False):
        encoded_data = [self.encode_pair(row[src_lang], row[tgt_lang], src_lang, tgt_lang, src_vocab, tgt_vocab, flip) for _, row in df.iterrows()]
        encoded_data.sort(key=lambda x: len(x[0]))

        def gen():
            for src, tgt in encoded_data:
                yield tf.constant(src, dtype=tf.int64), tf.constant(tgt, dtype=tf.int64)

        output_signature = (
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64)
        )

        dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None], [None]),
            padding_values=(tf.constant(self.PAD_IDX, dtype=tf.int64), tf.constant(self.PAD_IDX, dtype=tf.int64)),
            drop_remainder=True
        )

        return dataset

    def get_tf_datasets(self, batch_size=128, test_split_size = 0.2, validation_split_size = 0.5, flip=False):
        train_df, temp_df = train_test_split(self.dataset_df, test_size=test_split_size, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=(1-validation_split_size), random_state=42)

        src_vocab, inv_src_vocab = self.build_vocab(train_df[self.SRC_LANGUAGE], self.token_transform[self.SRC_LANGUAGE])
        tgt_vocab, inv_tgt_vocab = self.build_vocab(train_df[self.TGT_LANGUAGE], self.token_transform[self.TGT_LANGUAGE])

        train_dataset = self.prepare_tf_dataset_from_dataframe(train_df, self.SRC_LANGUAGE, self.TGT_LANGUAGE, src_vocab, tgt_vocab, batch_size, flip)
        val_dataset = self.prepare_tf_dataset_from_dataframe(val_df, self.SRC_LANGUAGE, self.TGT_LANGUAGE, src_vocab, tgt_vocab, batch_size, flip)

        return train_dataset, val_dataset, src_vocab, tgt_vocab, inv_src_vocab, inv_tgt_vocab

