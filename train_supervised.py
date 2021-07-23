import os
import time

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
from absl import app, flags, logging
from tensorflow.keras import mixed_precision
from tfds_korean import klue_sts, kornli, korsts  # noqa
from tqdm import tqdm

from model import BertConfig, BertModelForSimCSE
from optimizer import LinearWarmupAndDecayScheduler
from train_unsupervised import STSBenchmarkCallback, get_single_bert_input

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string("config", "./configs/char_bert_base.json", help="bert config")
    flags.DEFINE_string("pretrained_weight", "./model-checkpoints/char_bert_base/model", help="pretrained bert weight")
    flags.DEFINE_string("vocab_path", "vocabs/vocab_char_16424.txt", help="Vocab path")
    flags.DEFINE_boolean("mixed_precision", False, help="do mixed precision training")
    flags.DEFINE_string("tensorboard_log_path", "logs", help="tensorboard log dir")
    flags.DEFINE_string("model_checkpoints", "models", help="model checkpoint path")

    flags.DEFINE_integer("batch_size", 128, help="batch size")
    flags.DEFINE_integer("epochs", 3, help="epochs to train")
    flags.DEFINE_integer("max_sequence_length", 48, help="max sequence length")
    flags.DEFINE_integer("evaluation_frequency", 125, help="evaluation frequency (steps)")
    flags.DEFINE_float("temperature", 0.05, help="temperature for SimCSE")
    flags.DEFINE_float("warmup_ratio", 0.05, help="warm up ratio")
    flags.DEFINE_float("learning_rate", 5e-5, help="Learning rate")


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
    ds = get_supervised_dataset(bert_input_fn, FLAGS.batch_size)
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
    checkpoint_path = os.path.join(FLAGS.model_checkpoints, f"supervised-{timestamp}", "model-{epoch}")
    tensorboard_logdir = os.path.join(FLAGS.tensorboard_log_path, f"supervised-{timestamp}")
    bert_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
    )

    bert_model.fit(
        ds,
        epochs=FLAGS.epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard(tensorboard_logdir),
            STSBenchmarkCallback(klue_sts_ds, korsts_ds, model_save_dir=checkpoint_path, evaluation_frequency=FLAGS.evaluation_frequency),
        ],
    )


def get_supervised_dataset(bert_input_fn, batch_size):
    def _get_ds_from_split(split) -> tf.data.Dataset:
        ds = tfds.load("kornli", split=split)

        # label: [entailment, neutral, contradiction] => [0, 1, 2]
        sentences = {}
        for batch in tqdm(ds.as_numpy_iterator(), desc=f"reading {split}"):
            sentence1: str = batch["sentence1"].decode("utf8")
            sentence2: str = batch["sentence2"].decode("utf8")
            gold_label: int = batch["gold_label"]

            if sentence1 not in sentences:
                sentences[sentence1] = {}

            if gold_label != 1:  # not neutral
                sentences[sentence1][gold_label] = sentence2

        dataset_input = [(key, val[0], val[2]) for key, val in sentences.items() if 0 in val and 2 in val]
        logging.info(f"dataset length of split {split}: {len(dataset_input)}")
        return tf.data.Dataset.from_tensor_slices(dataset_input)

    return (
        _get_ds_from_split("mnli_train")
        .concatenate(_get_ds_from_split("snli_train"))
        .shuffle(500_000, reshuffle_each_iteration=True)
        .batch(batch_size)
        .map(lambda x: (bert_input_fn(x[:, 0]), bert_input_fn(x[:, 1]), bert_input_fn(x[:, 2])), num_parallel_calls=tf.data.AUTOTUNE)
        .map(create_label, num_parallel_calls=tf.data.AUTOTUNE)
    )


def create_label(x1, x2, z):
    batch_size = tf.shape(x1["input_word_ids"])[0]
    label = tf.expand_dims(tf.range(batch_size, dtype=tf.int64), -1)

    ctx = tf.distribute.get_replica_context()
    if ctx and ctx.num_replicas_in_sync != 1:
        label += ctx.replica_id_in_sync_group * batch_size

    return (x1, x2, z), label


if __name__ == "__main__":
    def_flags()
    app.run(main)
