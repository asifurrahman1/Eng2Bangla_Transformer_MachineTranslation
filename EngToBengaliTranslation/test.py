import argparse
from tqdm import tqdm
from util import masked_loss_fn, create_mask, greedy_decode
import tensorflow as tf
import os
import json
from collections import defaultdict
from data_pipeline import DataPipeline
from basetokenizer import BaseTokenizer
from model import SeqToSeqTransformer
import pandas as pd
from timeit import default_timer as timer

def create_src_padding_mask(src, PAD_IDX):
    src_padding_mask = tf.cast(tf.equal(src, PAD_IDX), tf.float32)  # shape: (batch, src_len)
    src_padding_mask = src_padding_mask[:, tf.newaxis, tf.newaxis, :]  # shape: (batch, 1, 1, src_len)
    return src_padding_mask

def translate(model, src_token, inv_tgt_vocab, datapipeline_obj):
    PAD_IDX = datapipeline_obj.get_special_token_idx()
    BOS_IDX = datapipeline_obj.get_special_token_idx('bos')
    EOS_IDX = datapipeline_obj.get_special_token_idx('eos')

    if len(src_token.shape) == 1:
        src_token = tf.expand_dims(src_token, axis=0)

    src_token = tf.cast(src_token, tf.int32)  # (1, seq_len)
    
    # padding mask (1, 1, 1, seq_len)
    src_mask = tf.cast(tf.math.equal(src_token, PAD_IDX), tf.float32)
    src_mask = src_mask[:, tf.newaxis, tf.newaxis, :]  # (1, 1, 1, seq_len)

    max_len = tf.shape(src_token)[1] + 5

    tgt_tokens = greedy_decode(
        model,
        src_token,  # (1, seq_len)
        src_mask,   # (1, 1, 1, seq_len)
        max_len=max_len,
        start_symbol=BOS_IDX,
        pad_idx=PAD_IDX,
        eos_idx=EOS_IDX
    )

    output = datapipeline_obj.index_to_text(tgt_tokens, inv_tgt_vocab)
    return output.replace("<bos>", "").replace("<eos>", "").strip()

def test_loop(transformer, val_dataloader, inv_src_vocab, inv_tgt_vocab, datapipeline_obj):
    test_itr = iter(val_dataloader)
    for _ in range(arg.testsample):
        src, trgt= next(test_itr)
        print("Input :",datapipeline_obj.index_to_text(src, inv_src_vocab).replace("<bos>", "").replace("<eos>", "").replace("<unk>", ""))
        print("Target Translation :", datapipeline_obj.index_to_text(trgt, inv_tgt_vocab).replace("<bos>", "").replace("<eos>", "").replace("<unk>", ""))
        print("Target Translation :", translate(transformer, src, inv_src_vocab, datapipeline_obj))


def main(arg):
    TokenizerClass =  None
    if not arg.default_tokenizer:
        # Define the custome tokenizer class by inheriting the BaseTokenizer here
        assert TokenizerClass is not None, "Define your custom tokeninzer class"
    
    datapipeline = DataPipeline(arg.dataset_path, arg.src_lang, arg.trgt_lang, TokenizerClass)
    _, val_dataloader, src_vocab, tgt_vocab, inv_src_vocab, inv_tgt_vocab= datapipeline.get_tf_datasets(batch_size = arg.batchsize, test_split_size=arg.split)
    model_struct = defaultdict()
    with open(arg.model_structfile, 'r') as f:
        model_struct = json.load(f)
    # print(model_struct)
    # print(len(src_vocab))
    # print(len(tgt_vocab))
    transformer = SeqToSeqTransformer(num_encoder_layers = model_struct['NUM_ENCODER_LAYERS'],
                                      num_decoder_layers = model_struct['NUM_DECODER_LAYERS'],
                                      emb_size = model_struct['EMB_SIZE'],
                                      nhead = model_struct['NHEAD'],
                                      src_vocab_size = len(src_vocab),
                                      tgt_vocab_size = len(tgt_vocab),
                                      dim_feedforward = model_struct['FFN_HID_DIM'],
                                      dropout = model_struct['DROPOUT']
                                      )
    dummy_src = tf.zeros((1, 10), dtype=tf.int32)
    dummy_tgt = tf.zeros((1, 10), dtype=tf.int32)
    _ = transformer(dummy_src, dummy_tgt, training=False)
    
    transformer.load_weights(os.path.join(os.getcwd(), arg.modeldir))
    print("Model weights loaded successfully from checkpoint.")
    test_loop(transformer, val_dataloader, inv_src_vocab, inv_tgt_vocab, datapipeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLM machine translation model.")
    parser.add_argument('--dataset-path', type=str, default='Dataset/english_to_bangla.csv', help='Path to the dataset')
    parser.add_argument('--src-lang', type=str, default='en', help='Source Language')
    parser.add_argument('--trgt-lang', type=str, default='bn', help='Target Language')
    parser.add_argument('--default-tokenizer', type=bool, default=True, help='If False, define your custome tokenizer')
    parser.add_argument('--model-structfile', type=str, default='model_struct.json', help='Path to the file defining LLM model structure')
    parser.add_argument('--testsample', type=int, default=10, help='Batch size')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size')
    parser.add_argument('--split', type=float, default=0.2, help='Batch size')
    parser.add_argument('--modeldir', type=str, default='SavedModel_TF/transformer_en_to_bn_model_best.weights.h5', help='Path to the trained LLM model')
    arg = parser.parse_args()

    
    if not os.path.exists(arg.dataset_path):
       raise FileNotFoundError(f"Dataset file '{arg.dataset_path}' does not exist.")

    if not os.path.exists(os.path.join(os.getcwd(), arg.modeldir)):
      raise FileNotFoundError(f"Trained Transformer's model '{arg.model_structfile}' does not exist.")

    if not os.path.exists(arg.model_structfile):
        raise FileNotFoundError(f"Transformer's structure definition file '{arg.model_structfile}' does not exist.")
    main(arg)