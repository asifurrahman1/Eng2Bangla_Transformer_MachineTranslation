import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def masked_loss_fn(real, pred, pad_idx=0):
    loss_ = loss_object(real, pred)
    mask = tf.cast(tf.not_equal(real, pad_idx), dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def generate_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0) 
    return mask * -1e9  

def create_mask(src, tgt, PAD_IDX):
    batch_size = tf.shape(src)[0]
    src_len = tf.shape(src)[1]
    tgt_len = tf.shape(tgt)[1]

    src_padding_mask = tf.cast(tf.equal(src, PAD_IDX), tf.float32)
    src_padding_mask = src_padding_mask[:, tf.newaxis, tf.newaxis, :] 

    tgt_padding_mask = tf.cast(tf.equal(tgt, PAD_IDX), tf.float32)
    tgt_padding_mask = tgt_padding_mask[:, tf.newaxis, tf.newaxis, :]  

    look_ahead_mask = generate_look_ahead_mask(tgt_len)
    look_ahead_mask = look_ahead_mask[tf.newaxis, :, :]  

    decoder_target_mask = tf.maximum(look_ahead_mask, tgt_padding_mask[:, 0, 0, tf.newaxis, :]) 

    return None, decoder_target_mask, src_padding_mask, tgt_padding_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol=2, pad_idx=1, eos_idx=3):
    memory = model.encode(src, src_mask, training=False)  # src: (1, seq_len)
    ys = tf.constant([[start_symbol]], dtype=tf.int32)  # (1, 1)

    for i in range(max_len - 1):
        tgt_mask = generate_look_ahead_mask(tf.shape(ys)[1])  # (tgt_len, tgt_len)
        tgt_mask = tgt_mask[tf.newaxis, :, :]  # (1, tgt_len, tgt_len)

        out = model.decode(ys, memory, tgt_mask=tgt_mask)  # (1, tgt_len, d_model)
        out = model.generator(out)  # (1, tgt_len, vocab_size)

        next_word = tf.argmax(out[:, -1, :], axis=-1, output_type=tf.int32)  # (1,)
        ys = tf.concat([ys, tf.expand_dims(next_word, axis=1)], axis=-1)  # (1, len+1)

        if next_word[0].numpy() == eos_idx:
            break

    return ys[0]  




