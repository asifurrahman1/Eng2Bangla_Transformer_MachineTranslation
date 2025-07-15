import argparse
from tqdm import tqdm
from util import masked_loss_fn, create_mask
import tensorflow as tf
import os
import json
from collections import defaultdict
from data_pipeline import DataPipeline
from basetokenizer import BaseTokenizer
from model import SeqToSeqTransformer
import pandas as pd
from timeit import default_timer as timer


def evaluate(model, val_dataloader, pad_idx=0):
    total_loss = 0.0
    num_batches = 0

    for batch, (src, tgt) in enumerate(val_dataloader):
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input,
                       src_mask=src_mask,
                       tgt_mask=tgt_mask,
                       src_padding_mask=src_padding_mask,
                       tgt_padding_mask=tgt_padding_mask,
                       training=False)

        batch_loss = masked_loss_fn(tgt_out, logits, pad_idx=pad_idx)
        total_loss += batch_loss.numpy()
        num_batches += 1
    return total_loss / num_batches


def train_epoch(model, optimizer, train_dataloader, pad_idx=1):
    total_loss = 0.0
    num_batches = 0
    train_iterator = tqdm(train_dataloader, desc="Training", leave=False)
    for (batch, (src, tgt)) in enumerate(train_iterator):
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx)
        with tf.GradientTape() as tape:
            logits = model(src, tgt_input,
                           src_mask=src_mask,
                           tgt_mask=tgt_mask,
                           src_padding_mask=src_padding_mask,
                           tgt_padding_mask=tgt_padding_mask,
                           training=True)

            loss = masked_loss_fn(tgt_out, logits, pad_idx=pad_idx)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss.numpy()
        num_batches += 1
        train_iterator.set_postfix(loss=loss.numpy())
    return total_loss / num_batches

def training_loop(arg, transformer,  optimizer, train_dataloader, val_dataloader, pad_idx=1):
    model_save_dir = arg.model_savedir
    TrainLoss = []
    ValLoss = []

    NUM_EPOCHS = arg.epoch
    best_model_loss = float('inf')
    checkpoint_path = os.path.join(model_save_dir, 'transformer_en_to_bn_model_best.weights.h5')
    final_path = os.path.join(model_save_dir, 'transformer_en_to_bn_model_final.weights.h5')
    dummy_src = tf.zeros((1, 10), dtype=tf.int32)
    dummy_tgt = tf.zeros((1, 10), dtype=tf.int32)
    _ = transformer(dummy_src, dummy_tgt, training=False)

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at: {checkpoint_path}")
        transformer.load_weights(checkpoint_path)
        print("Model weights loaded from checkpoint.")
        best_model_loss = evaluate(transformer, val_dataloader, pad_idx=pad_idx)
        print(f"Resuming training from val loss: {best_model_loss:.4f}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader, pad_idx=pad_idx)
        TrainLoss.append(train_loss)
        end_time = timer()
        val_loss = evaluate(transformer, val_dataloader, pad_idx=pad_idx)
        ValLoss.append(val_loss)
        
        if val_loss < best_model_loss:
            best_model_loss = val_loss
            transformer.save_weights(checkpoint_path)
            print(f"Best model updated (val loss: {val_loss:.4f}) and saved.")

        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Time: {end_time - start_time:.2f}s")

    transformer.save_weights(final_path)
    print(f"Final model saved to {final_path}")


def main(arg):
    TokenizerClass =  None
    if not arg.default_tokenizer:
        # Define the custome tokenizer class by inheriting the BaseTokenizer here
        assert TokenizerClass is not None, "Define your custom tokeninzer class"
  
    datapipeline = DataPipeline(arg.dataset_path, arg.src_lang, arg.trgt_lang, TokenizerClass)
    train_dataloader, val_dataloader, src_vocab, tgt_vocab, inv_src_vocab, inv_tgt_vocab= datapipeline.get_tf_datasets(batch_size = arg.batchsize, test_split_size=arg.split)
  
    model_struct = defaultdict()
    with open(arg.model_structfile, 'r') as f:
        model_struct = json.load(f)
    transformer = SeqToSeqTransformer(num_encoder_layers = model_struct['NUM_ENCODER_LAYERS'],
                                      num_decoder_layers = model_struct['NUM_DECODER_LAYERS'],
                                      emb_size = model_struct['EMB_SIZE'],
                                      nhead = model_struct['NHEAD'],
                                      src_vocab_size = len(src_vocab),
                                      tgt_vocab_size = len(tgt_vocab),
                                      dim_feedforward = model_struct['FFN_HID_DIM'],
                                      dropout = model_struct['DROPOUT']
                                      )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    PAD_IDX = datapipeline.get_special_token_idx()
    training_loop(arg, transformer, optimizer, train_dataloader, val_dataloader, pad_idx=PAD_IDX)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM machine translation model.")
    parser.add_argument('--dataset-path', type=str, default='Dataset/english_to_bangla.csv', help='Path to the dataset')
    parser.add_argument('--src-lang', type=str, default='en', help='Source Language')
    parser.add_argument('--trgt-lang', type=str, default='bn', help='Target Language')
    parser.add_argument('--default-tokenizer', type=bool, default=True, help='If False, define your custome tokenizer')
    parser.add_argument('--model-structfile', type=str, default='model_struct.json', help='Path to the file defining LLM model structure')
    parser.add_argument('--batchsize', type=int, default=128, help='Batch size')
    parser.add_argument('--epoch', type=int, default=10, help='Batch size')
    parser.add_argument('--split', type=float, default=0.2, help='Batch size')
    parser.add_argument('--model-savedir', type=str, default='SavedModel_TF', help='Path to the file defining LLM model structure')
    arg = parser.parse_args()

    if not os.path.exists(os.path.join(os.getcwd(), arg.model_savedir)):
      os.makedirs(os.path.join(os.getcwd(), arg.model_savedir))

    if not os.path.exists(arg.dataset_path):
       raise FileNotFoundError(f"Dataset file '{arg.dataset_path}' does not exist.")

    if not os.path.exists(arg.model_structfile):
        raise FileNotFoundError(f"Transformer's structure definition file '{arg.model_structfile}' does not exist.")
    main(arg)