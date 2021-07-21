import numpy as np
import tensorflow as tf
from absl import app, flags, logging

from model import BertConfig, BertModel

FLAGS = flags.FLAGS

flags.DEFINE_string("config", "./configs/char_bert_base.json", help="BERT config path")
flags.DEFINE_string("checkpoint", "./model-checkpoints/char_bert/model.ckpt-2000000", help="KR-BERT checkpoint path")
flags.DEFINE_string("output", "./model-checkpoints/char_bert_base/model", help="Output path")
flags.DEFINE_integer("vocab_size", 16424, help="Vocab size")


def main(argv):
    logging.info("Weights: \n" + "\n".join([str(v) for v in tf.train.list_variables(FLAGS.checkpoint) if "adam" not in v[0]]))

    bert_config = BertConfig.from_json(FLAGS.config, vocab_size=FLAGS.vocab_size)
    bert_model = BertModel(bert_config)
    bert_model(
        {
            "input_word_ids": tf.keras.Input([None], dtype=tf.int64),
            "input_type_ids": tf.keras.Input([None], dtype=tf.int64),
            "input_mask": tf.keras.Input([None], dtype=tf.int64),
        }
    )

    load_embedding_and_pooler(bert_model, FLAGS.checkpoint)
    for layer_index in range(bert_config.num_hidden_layers):
        load_transformer_encoder(bert_model, layer_index, FLAGS.checkpoint)
    bert_model.save_weights(FLAGS.output)


def load_embedding_and_pooler(bert_model, checkpoint_path):
    bert_model.bert_embedding.input_word_embeddings.set_weights(
        [tf.train.load_variable(checkpoint_path, "bert/embeddings/word_embeddings")]
    )
    bert_model.bert_embedding.input_type_embeddings.set_weights(
        [tf.train.load_variable(checkpoint_path, "bert/embeddings/token_type_embeddings")]
    )
    bert_model.bert_embedding.position_embeddings.set_weights(
        [tf.train.load_variable(checkpoint_path, "bert/embeddings/position_embeddings")]
    )
    bert_model.bert_embedding.layer_normalization.set_weights(
        [
            tf.train.load_variable(checkpoint_path, "bert/embeddings/LayerNorm/gamma"),
            tf.train.load_variable(checkpoint_path, "bert/embeddings/LayerNorm/beta"),
        ]
    )

    bert_model.pooler.set_weights(
        [
            tf.train.load_variable(checkpoint_path, "bert/pooler/dense/kernel"),
            tf.train.load_variable(checkpoint_path, "bert/pooler/dense/bias"),
        ]
    )


def load_transformer_encoder(bert_model, layer_index, checkpoint_path):
    bert_model.transformer_layers[layer_index].multihead_attention.qkv.set_weights(
        [
            np.concatenate(
                [
                    tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/attention/self/query/kernel"),
                    tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/attention/self/key/kernel"),
                    tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/attention/self/value/kernel"),
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/attention/self/query/bias"),
                    tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/attention/self/key/bias"),
                    tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/attention/self/value/bias"),
                ],
                axis=0,
            ),
        ]
    )
    bert_model.transformer_layers[layer_index].multihead_attention.output_projection.set_weights(
        [
            tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/attention/output/dense/kernel"),
            tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/attention/output/dense/bias"),
        ]
    )
    bert_model.transformer_layers[layer_index].mha_layer_normalization.set_weights(
        [
            tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/attention/output/LayerNorm/gamma"),
            tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/attention/output/LayerNorm/beta"),
        ]
    )

    bert_model.transformer_layers[layer_index].intermediate_layer.set_weights(
        [
            tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/intermediate/dense/kernel"),
            tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/intermediate/dense/bias"),
            tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/output/dense/kernel"),
            tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/output/dense/bias"),
        ]
    )
    bert_model.transformer_layers[layer_index].intermediate_layer_normalization.set_weights(
        [
            tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/output/LayerNorm/gamma"),
            tf.train.load_variable(checkpoint_path, f"bert/encoder/layer_{layer_index}/output/LayerNorm/beta"),
        ]
    )


if __name__ == "__main__":
    app.run(main)
