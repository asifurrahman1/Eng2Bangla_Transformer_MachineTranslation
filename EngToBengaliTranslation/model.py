import tensorflow as tf
import math
from tensorflow.keras.layers import LayerNormalization, Dropout, Dense, Embedding, MultiHeadAttention

class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def call(self, tokens):
        return self.embedding(tf.cast(tokens, tf.int32)) * tf.math.sqrt(tf.cast(self.emb_size, tf.float32))

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout_rate=0.1, max_len=5000):
        super().__init__()
        self.dropout = Dropout(dropout_rate)
        pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        self.pos_encoding = tf.cast(pos_encoding, tf.float32)

    def call(self, x, training=False):
        x = x + self.pos_encoding[:, :tf.shape(x)[1], :]
        return self.dropout(x, training=training)


def expand_padding_mask(mask, dtype=tf.float32):
    return tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype)  # shape: (batch, 1, 1, seq_len)


def FullyConnected(embed_dim, ffn_dim):
    return tf.keras.Sequential([
        Dense(ffn_dim, activation='relu'),
        Dense(embed_dim)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=nhead, key_dim=d_model)
        self.ffn = FullyConnected(d_model, dim_feedforward)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, x, training=False, mask=None):
        attn_output = self.mha(query=x, key=x, value=x, attention_mask=mask)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(num_heads=nhead, key_dim=d_model)
        self.mha2 = MultiHeadAttention(num_heads=nhead, key_dim=d_model)
        self.ffn = FullyConnected(d_model, dim_feedforward)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        out1 = self.layernorm1(x + self.dropout1(attn1, training=training))

        attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        out2 = self.layernorm2(out1 + self.dropout2(attn2, training=training))

        ffn_output = self.ffn(out2)
        out3 = self.layernorm3(out2 + self.dropout3(ffn_output, training=training))
        return out3

class Transformer(tf.keras.Model):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.encoder_layers = [EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)]
        self.decoder_layers = [DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)]

    def call(self, src, tgt, training=False, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        for layer in self.encoder_layers:
            src = layer(src, training=training, mask=src_padding_mask)
        memory = src
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, training=training, look_ahead_mask=tgt_mask, padding_mask=src_padding_mask)
        return tgt

class SeqToSeqTransformer(tf.keras.Model):
    def __init__(self,
                 num_encoder_layers,
                 num_decoder_layers,
                 emb_size,
                 nhead,
                 src_vocab_size,
                 tgt_vocab_size,
                 dim_feedforward=512,
                 dropout=0.1):
        super().__init__()
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.generator = Dense(tgt_vocab_size)

    def call(self, src, tgt,
             src_mask=None, tgt_mask=None,
             src_padding_mask=None, tgt_padding_mask=None,
             training=False):
        src_emb = self.positional_encoding(self.src_tok_emb(src), training=training)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt), training=training)
        output = self.transformer(src_emb, tgt_emb,
                                  training=training,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  src_padding_mask=src_padding_mask,
                                  tgt_padding_mask=tgt_padding_mask)
        return self.generator(output)

    # def encode(self, src, src_mask=None):
    #     return self.transformer.encoder_layers[0].call(
    #         self.positional_encoding(self.src_tok_emb(src)), mask=src_mask
    #     )
    def encode(self, src, src_mask=None, training=False):
        src_emb = self.positional_encoding(self.src_tok_emb(src), training=training)
        for layer in self.transformer.encoder_layers:
            src_emb = layer(src_emb, training=training, mask=src_mask)
        return src_emb

    def decode(self, tgt, memory, tgt_mask=None, src_padding_mask=None, training=False):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt), training=training)
        for layer in self.transformer.decoder_layers:
            tgt_emb = layer(tgt_emb, memory, training=training, look_ahead_mask=tgt_mask, padding_mask=src_padding_mask)
        return tgt_emb

    # def decode(self, tgt, memory, tgt_mask=None):
    #     return self.transformer.decoder_layers[0].call(
    #         self.positional_encoding(self.tgt_tok_emb(tgt)), memory, look_ahead_mask=tgt_mask
    #     )