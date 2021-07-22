import os
import time

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
from absl import app, flags, logging
from scipy import stats
from tensorflow.keras import mixed_precision
from tfds_korean import klue_sts, korean_wikipedia_corpus, korsts  # noqa

from model import BertConfig, BertModelForSimCSE
from optimizer import LinearWarmupAndDecayScheduler

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string("config", "./configs/char_bert_base.json", help="bert config")
    flags.DEFINE_string("pretrained_weight", "./model-checkpoints/char_bert_base/model", help="pretrained bert weight")
    flags.DEFINE_string("vocab_path", "vocabs/vocab_char_16424.txt", help="Vocab path")
    flags.DEFINE_boolean("mixed_precision", False, help="do mixed precision training")
    flags.DEFINE_string("tensorboard_log_path", "logs", help="tensorboard log dir")
    flags.DEFINE_string("model_checkpoints", "models", help="model checkpoint path")

    flags.DEFINE_integer("batch_size", 64, help="batch size")
    flags.DEFINE_integer("steps_per_epoch", 250, help="")
    flags.DEFINE_integer("epochs", 100, help="epochs to train")
    flags.DEFINE_integer("max_sequence_length", 64, help="max sequence length")
    flags.DEFINE_float("temperature", 0.05, help="temperature for SimCSE")
    flags.DEFINE_float("warmup_ratio", 0.05, help="warm up ratio")
    flags.DEFINE_float("learning_rate", 3e-5, help="Learning rate")


def main(argv):
    if FLAGS.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
        logging.info(f"Compute dtype: {mixed_precision.global_policy().compute_dtype}")
        logging.info(f"Variable dtype: {mixed_precision.global_policy().variable_dtype}")

    tokenizer = text.BertTokenizer(FLAGS.vocab_path, unknown_token="[UNK]")
    unk_id = tokenizer._wordpiece_tokenizer._vocab_lookup_table.lookup(tf.constant("[UNK]"))
    pad_id = tokenizer._wordpiece_tokenizer._vocab_lookup_table.lookup(tf.constant("[PAD]"))
    cls_id = tokenizer._wordpiece_tokenizer._vocab_lookup_table.lookup(tf.constant("[CLS]"))
    sep_id = tokenizer._wordpiece_tokenizer._vocab_lookup_table.lookup(tf.constant("[SEP]"))
    assert all([unk_id != id_ for id_ in [pad_id, cls_id, sep_id]])
    logging.info(f"PAD ID: {pad_id}, UNK ID: {unk_id} CLS ID: {cls_id}, SEP ID: {sep_id}")
    bert_input_fn = get_single_bert_input(
        tokenizer=tokenizer,
        pad_id=pad_id,
        cls_id=cls_id,
        sep_id=sep_id,
        max_sequence_length=FLAGS.max_sequence_length,
    )
    klue_sts_ds = (
        tfds.load("klue_sts", split="dev")
        .batch(FLAGS.batch_size)
        .map(lambda x: {"sentence1": bert_input_fn(x["sentence1"]), "sentence2": bert_input_fn(x["sentence2"]), "score": x["label"]})
    )
    korsts_ds = (
        tfds.load("korsts", split="dev")
        .batch(FLAGS.batch_size)
        .map(lambda x: {"sentence1": bert_input_fn(x["sentence1"]), "sentence2": bert_input_fn(x["sentence2"]), "score": x["score"]})
    )
    ds = (
        tfds.load("korean_wikipedia_corpus", split="train")
        .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x["content"]))
        .shuffle(1_000_000, reshuffle_each_iteration=True)
        .batch(FLAGS.batch_size)
        .map(bert_input_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .map(_create_label, num_parallel_calls=tf.data.AUTOTUNE)
        .repeat()
    )
    logging.info(f"batch_size: {FLAGS.batch_size}, element_spec: {ds.element_spec}")
    bert_config = BertConfig.from_json(FLAGS.config)
    logging.info(f"Config: L{bert_config.num_hidden_layers}, A{bert_config.num_attention_heads}, H{bert_config.hidden_size}.")

    logging.info("Initialize Teacher BERT Model")
    bert_model = BertModelForSimCSE(bert_config, temperature=FLAGS.temperature, name="bert_model")
    if FLAGS.pretrained_weight:
        logging.info(f"Load pretrained weights from {FLAGS.pretrained_weight}")
        bert_model.load_weights(FLAGS.pretrained_weight)

    for batch, _ in ds.take(1):
        bert_model(batch)
    bert_model.summary()

    if FLAGS.steps_per_epoch > 0:
        steps_per_epoch = FLAGS.steps_per_epoch
    else:
        steps_per_epoch = len([1 for _ in ds])
    total_steps = steps_per_epoch * FLAGS.epochs
    warmup_steps = int(FLAGS.warmup_ratio * total_steps)
    logging.info(f"total steps: {total_steps}, warmup steps: {warmup_steps}")
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LinearWarmupAndDecayScheduler(
            FLAGS.learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )
    timestamp = int(time.time())
    checkpoint_path = os.path.join(FLAGS.model_checkpoints, f"unsupervised-{timestamp}", "model-{epoch}")
    bert_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer)
    bert_model.fit(
        ds,
        epochs=FLAGS.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True),
            STSBenchmarkCallback(klue_sts_ds, korsts_ds),
        ],
    )


def _create_label(x):
    batch_size = tf.shape(x["input_word_ids"])[0]
    label = tf.expand_dims(tf.range(batch_size, dtype=tf.int64), -1)

    ctx = tf.distribute.get_replica_context()
    if ctx and ctx.num_replicas_in_sync != 1:
        label += ctx.replica_id_in_sync_group * batch_size

    return x, label


def get_single_bert_input(tokenizer: text.BertTokenizer, pad_id: int, cls_id: int, sep_id: int, max_sequence_length: int):
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


class STSBenchmarkCallback(tf.keras.callbacks.Callback):
    def __init__(self, klue_sts_ds, korsts_ds):
        self.klue_sts_ds = klue_sts_ds
        self.korsts_ds = korsts_ds

    def on_epoch_end(self, epoch, logs=None):
        klue_metric = stats.pearsonr(*get_scores_and_similarities(self.model, self.klue_sts_ds))[0]
        korsts_metric = stats.spearmanr(*get_scores_and_similarities(self.model, self.korsts_ds))[0]

        logging.info(f"KlueSTS PearsonR: {klue_metric}, KorSTS SpearmanR: {korsts_metric}")


def get_scores_and_similarities(model, ds):
    scores = []
    similarities = []

    for item in ds:
        similarities.append(model.calculate_similarity(item["sentence1"], item["sentence2"]))
        scores.append(item["score"])

    return tf.concat(scores, axis=0), tf.concat(similarities, axis=0)


if __name__ == "__main__":
    def_flags()
    app.run(main)
