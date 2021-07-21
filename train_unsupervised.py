from typing import Dict
import time

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
from tensorflow.keras import mixed_precision
from absl import app, flags, logging

from model import BertConfig, BertModelForSimCSE
from optimizer import AdamW, LinearWarmupAndDecayScheduler

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", "nsmc", help="dataset to train")
flags.DEFINE_string("config", "./configs/char_bert_base.json", help="bert config")
flags.DEFINE_string("pretrained_weight", "./model-checkpoints/char_bert_base/model", help="pretrained bert weight")
flags.DEFINE_string("vocab_path", "vocabs/vocab_char_16424.txt", help="Vocab path")
flags.DEFINE_boolean("mixed_precision", False, help="do mixed precision training")
flags.DEFINE_string("tensorboard_log_path", "logs", help='tensorboard log dir')
flags.DEFINE_string("model_checkpoints", "models", help='model checkpoint path')

flags.DEFINE_integer("batch_size", 64, help="batch size")
flags.DEFINE_integer("epochs", 3, help="epochs to train")
flags.DEFINE_integer("max_sequence_length", 64, help="max sequence length")
flags.DEFINE_float("temperature", 0.05, help='temperature for SimCSE')
flags.DEFINE_float("warmup_ratio", 0.05, help="warm up ratio")
flags.DEFINE_float("adam_beta1", 0.9, help="Adam Beta 1 value")
flags.DEFINE_float("adam_beta2", 0.999, help="Adam Beta 2 value")
flags.DEFINE_float("weight_decay", 0.001, help="Weight decay value")
flags.DEFINE_float("learning_rate", 5e-5, help="Learning rate")


def main(argv):
    if FLAGS.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
        logging.info(f"Compute dtype: {mixed_precision.global_policy().compute_dtype}")
        logging.info(f"Variable dtype: {mixed_precision.global_policy().variable_dtype}")

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        tokenizer = text.BertTokenizer(FLAGS.vocab_path, unknown_token="[UNK]")
        unk_id = tokenizer._wordpiece_tokenizer._vocab_lookup_table.lookup(tf.constant("[UNK]"))
        pad_id = tokenizer._wordpiece_tokenizer._vocab_lookup_table.lookup(tf.constant("[PAD]"))
        cls_id = tokenizer._wordpiece_tokenizer._vocab_lookup_table.lookup(tf.constant("[CLS]"))
        sep_id = tokenizer._wordpiece_tokenizer._vocab_lookup_table.lookup(tf.constant("[SEP]"))
        assert all([unk_id != id_ for id_ in [pad_id, cls_id, sep_id]])
        logging.info(f"PAD ID: {pad_id}, UNK ID: {unk_id} CLS ID: {cls_id}, SEP ID: {sep_id}")
        ds = _prepare_datasets(
            tokenizer=tokenizer,
            batch_size=FLAGS.batch_size,
            pad_id=pad_id,
            cls_id=cls_id,
            sep_id=sep_id,
            max_sequence_length=FLAGS.max_sequence_length,
        )
        logging.info(f"batch_size: {FLAGS.batch_size}, element_spec: {ds.element_spec}")

        bert_config = BertConfig.from_json(FLAGS.config)
        logging.info(f"Config: L{bert_config.num_hidden_layers}, A{bert_config.num_attention_heads}, H{bert_config.hidden_size}.")

        logging.info("Initialize Teacher BERT Model")
        bert_model = BertModelForSimCSE(bert_config, temperature=FLAGS.temperature, name="bert_model")
        if FLAGS.pretrained_weight:
            logging.info(f"Load pretrained weights from {FLAGS.pretrained_weight}")
            bert_model.load_weights(FLAGS.pretrained_weight)

        bert_model.build(
            {
                "input_word_ids": [None, None],
                "input_type_ids": [None, None],
                "input_mask": [None, None],
            }
        )
        bert_model.summary()

        no_decay = ["layer_normalization/gamma", "bias"]
        decay_var_list = [var for var in bert_model.trainable_variables if not any(name in var.name for name in no_decay)]
        total_steps = len([1 for _ in ds["train"]]) * FLAGS.epochs
        warmup_steps = int(FLAGS.warmup_ratio * total_steps)
        logging.info(f"total steps: {total_steps}, warmup steps: {warmup_steps}")
        optimizer = AdamW(
            beta_1=FLAGS.adam_beta1,
            beta_2=FLAGS.adam_beta2,
            weight_decay=LinearWarmupAndDecayScheduler(
                FLAGS.learning_rate * FLAGS.weight_decay,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            ),
            learning_rate=LinearWarmupAndDecayScheduler(
                FLAGS.learning_rate,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            ),
            decay_var_list=decay_var_list,
        )
        timestamp = int(time.time())
        bert_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[], optimizer=optimizer)
        bert_model.fit(
            ds["train"],
            validation_data=ds["dev"],
            epochs=FLAGS.epochs,
            callbacks=[
                tf.keras.callbacks.TensorBoard(os.path.join(FLAGS.tensorboard_log_path, f"unsupervised-{timestamp}", update_freq='batch'),
                tf.keras.callbacks.ModelCheckpoint(os.path.join(FLAGS.model_checkpoints, f"unsupervised-{timestamp}")),
            ],
        )


def _prepare_datasets(
    tokenizer: text.BertTokenizer,
    batch_size: int,
    pad_id: int,
    cls_id: int,
    sep_id: int,
    max_sequence_length: int,
) -> Dict[str, tf.data.Dataset]:
    ds = tfds.load("korean_wikipedia_corpus", split="train")
    bert_input_fn = _get_single_bert_input(
        tokenizer=tokenizer,
        pad_id=pad_id,
        cls_id=cls_id,
        sep_id=sep_id,
        max_sequence_length=max_sequence_length,
    )
    train_ds = (
        ds
        .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x['content']))
        .shuffle(100_000, reshuffle_each_iteration=True)
        .batch(batch_size)
        .map(bert_input_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .map(_create_label, num_parallel_calls=tf.data.AUTOTUNE)
    )

    return {"train": train_ds}


def _create_label(x):
    batch_size = tf.shape(x["input_word_ids"])[0]
    label = tf.expand_dims(tf.range(batch_size, dtype=tf.int64), -1)

    ctx = tf.distribute.get_replica_context()
    if ctx and ctx.num_replicas_in_sync != 1:
        label += ctx.replica_id_in_sync_group * batch_size

    return x, label


def _get_single_bert_input(tokenizer: text.BertTokenizer, pad_id: int, cls_id: int, sep_id: int, max_sequence_length: int):
    def _inner(x: tf.Tensor):
        # x: tf.TensorSpec([None], dtype=tf.string)
        batch_size = tf.size(x)
        sentence = tokenizer.tokenize(x).merge_dims(-2, -1)

        segments = tf.concat(
            [tf.expand_dims(tf.repeat(cls_id, batch_size), -1), sentence, tf.expand_dims(tf.repeat(sep_id, batch_size), -1)],
            axis=-1,
        )

        input_word_ids = segments.to_tensor(shape=[batch_size, max_sequence_length], default_value=pad_id)
        input_mask = tf.ragged.map_flat_values(tf.ones_like, segments, dtype=tf.bool).to_tensor(
            shape=[batch_size, max_sequence_length],
            default_value=False,
        )
        input_type_ids = tf.zeros([batch_size, max_sequence_length], dtype=tf.int64)

        return {
            "input_word_ids": input_word_ids,
            "input_mask": input_mask,
            "input_type_ids": input_type_ids,
        }

    return _inner


if __name__ == "__main__":
    app.run(main)
