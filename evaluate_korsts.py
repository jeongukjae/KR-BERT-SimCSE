import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
import tfds_korean.korsts  # noqa
from absl import app, flags, logging
from scipy import stats

from model import BertConfig, BertModel, CosineSimilarity
from train_unsupervised import get_single_bert_input

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string("config", "./configs/char_bert_base.json", help="bert config")
    flags.DEFINE_string("weight", "", help="bert weight")
    flags.DEFINE_string("vocab_path", "vocabs/vocab_char_16424.txt", help="Vocab path")
    flags.DEFINE_string("split", "dev", help="split name")
    flags.DEFINE_integer("batch_size", 64, help="batch size")
    flags.DEFINE_integer("max_sequence_length", 64, help="max sequence length")


def main(argv):
    tokenizer = text.BertTokenizer(FLAGS.vocab_path, unknown_token="[UNK]")
    pad_id = tokenizer._wordpiece_tokenizer._vocab_lookup_table.lookup(tf.constant("[PAD]"))
    cls_id = tokenizer._wordpiece_tokenizer._vocab_lookup_table.lookup(tf.constant("[CLS]"))
    sep_id = tokenizer._wordpiece_tokenizer._vocab_lookup_table.lookup(tf.constant("[SEP]"))

    bert_config = BertConfig.from_json(FLAGS.config)
    bert_model = BertModel(bert_config, name="bert_model")
    logging.info(f"Load weights from {FLAGS.weight}")
    bert_model.load_weights(FLAGS.weight)

    dataset = tfds.load("korsts", split=FLAGS.split).batch(FLAGS.batch_size)
    bert_input_fn = get_single_bert_input(
        tokenizer=tokenizer,
        pad_id=pad_id,
        cls_id=cls_id,
        sep_id=sep_id,
        max_sequence_length=FLAGS.max_sequence_length,
    )

    @tf.function
    def calculate_similarity(sentence1, sentence2):
        representation1 = bert_model(bert_input_fn(sentence1))["sequence_output"][:, 0]
        representation2 = bert_model(bert_input_fn(sentence2))["sequence_output"][:, 0]

        return CosineSimilarity()([representation1, representation2])

    label_score = []
    pred_score = []

    for item in dataset:
        label_score.append(item["score"])
        pred_score.append(calculate_similarity(item["sentence1"], item["sentence2"]))

    label_score = tf.concat(label_score, axis=0)
    pred_score = tf.concat(pred_score, axis=0)
    print(stats.spearmanr(label_score, pred_score))


if __name__ == "__main__":
    def_flags()
    app.run(main)
