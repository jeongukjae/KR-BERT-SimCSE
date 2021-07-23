import json
import math
import warnings
from typing import Callable, Union

import tensorflow as tf


class BertConfig:
    def __init__(
        self,
        vocab_size: int,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        intermediate_activation: str = "gelu",
        num_attention_heads: int = 12,
        max_position_ids: int = 512,
        input_type_size: int = 2,
        dropout_rate: float = 0.1,
        attention_probs_dropout_rate: float = 0.1,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if len(kwargs) != 0:
            warnings.warn("Unused parameters found: " + kwargs)

        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_activation = get_activation_fn(intermediate_activation)
        self.num_attention_heads = num_attention_heads
        self.max_position_ids = max_position_ids
        self.input_type_size = input_type_size
        self.dropout_rate = dropout_rate
        self.attention_probs_dropout_rate = attention_probs_dropout_rate
        self.initializer_range = initializer_range

    @staticmethod
    def from_json(json_filename: str, **kwargs) -> "BertConfig":
        with open(json_filename, encoding="utf8") as f:
            jsondict = json.load(f)
            jsondict.update(kwargs)

        return BertConfig(**jsondict)


class BertModel(tf.keras.Model):
    def __init__(self, bert_config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.bert_config = bert_config

        self.bert_embedding = BertEmbedding(
            vocab_size=bert_config.vocab_size,
            hidden_size=bert_config.hidden_size,
            max_position_ids=bert_config.max_position_ids,
            input_type_size=bert_config.input_type_size,
            dropout_rate=bert_config.dropout_rate,
            initializer_range=bert_config.initializer_range,
            name="bert_embedding",
        )

        self.transformer_layers = [
            TransformerEncoder(
                hidden_size=bert_config.hidden_size,
                num_attention_heads=bert_config.num_attention_heads,
                intermediate_size=bert_config.intermediate_size,
                intermediate_activation=bert_config.intermediate_activation,
                dropout_rate=bert_config.dropout_rate,
                attention_probs_dropout_rate=bert_config.attention_probs_dropout_rate,
                initializer_range=bert_config.initializer_range,
                name=f"encoder_{i}",
            )
            for i in range(bert_config.num_hidden_layers)
        ]

        self.pooler = tf.keras.layers.Dense(
            bert_config.hidden_size,
            activation="tanh",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range),
            name="pooler",
        )

    def call(self, input_tensor):
        input_mask = input_tensor["input_mask"]
        hidden_state = self.bert_embedding(input_tensor)

        input_mask = tf.cast(input_mask, hidden_state.dtype)
        input_mask = (1.0 - input_mask[:, tf.newaxis, tf.newaxis, :]) * -10000.0

        for layer in self.transformer_layers:
            hidden_state = layer(hidden_state, input_mask=input_mask)

        pooled_output = self.pooler(hidden_state[:, 0])
        return {"sequence_output": hidden_state, "pooled_output": pooled_output}


class BertModelForSimCSE(BertModel):
    def __init__(self, *args, temperature=0.05, **kwargs):
        super().__init__(*args, **kwargs)

        self.temperature = temperature
        self.cosine_similarity = CosineSimilarity(dtype="float32")

    def call(self, input_tensor):
        if isinstance(input_tensor, dict):  # unsupervised
            r1 = super().call(input_tensor)["pooled_output"]
            r2 = super().call(input_tensor)["pooled_output"]
            z = None
        else:  # supervised
            r1 = super().call(input_tensor[0])["pooled_output"]
            r2 = super().call(input_tensor[1])["pooled_output"]

            z = super().call(input_tensor[2])["pooled_output"]
            z = self._reduce_representations(z)

        r2 = self._reduce_representations(r2)
        if z is not None:
            r2 = tf.concat([r2, z], axis=1)

        return self.cosine_similarity([tf.expand_dims(r1, 1), r2]) / self.temperature

    def _reduce_representations(self, representations):
        hidden_size = tf.shape(representations)[-1]

        ctx = tf.distribute.get_replica_context()
        if ctx and ctx.num_replicas_in_sync != 1:
            print(f"reduce reprsentations, num_replicas_in_sync: {ctx.num_replicas_in_sync}, and id: {ctx.replica_id_in_sync_group}")
            representations = tf.where(
                (tf.range(0, ctx.num_replicas_in_sync) == ctx.replica_id_in_sync_group)[:, tf.newaxis, tf.newaxis],
                tf.expand_dims(representations, 0),
                tf.expand_dims(tf.zeros_like(representations), 0),
            )
            [representations] = ctx.all_reduce(tf.distribute.ReduceOp.SUM, [representations])
            representations = tf.reshape(representations, [1, -1, hidden_size])
        else:
            representations = tf.expand_dims(representations, 0)

        return representations

    @tf.function
    def calculate_similarity(self, sentence1, sentence2):
        r1 = super().call(sentence1)["sequence_output"][:, 0]
        r2 = super().call(sentence2)["sequence_output"][:, 0]

        return self.cosine_similarity([r1, r2])


class CosineSimilarity(tf.keras.layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-12, **kwargs):
        super().__init__(**kwargs)

        self.axis = axis
        self.epsilon = epsilon

    def call(self, input_tensors):
        a, b = input_tensors
        input_dtype = a.dtype

        if input_dtype in (tf.float16, tf.bfloat16) and self.dtype == tf.float32:
            a = tf.cast(a, tf.float32)
            b = tf.cast(b, tf.float32)

        a = tf.nn.l2_normalize(a, axis=self.axis, epsilon=self.epsilon)
        b = tf.nn.l2_normalize(b, axis=self.axis, epsilon=self.epsilon)
        result = tf.reduce_sum(a * b, axis=self.axis)

        return tf.cast(result, input_dtype)


class BertEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_ids: int,
        input_type_size: int,
        dropout_rate: float,
        initializer_range: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_word_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="input_word_embeddings",
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=max_position_ids,
            output_dim=hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="position_embeddings",
        )
        self.input_type_embeddings = tf.keras.layers.Embedding(
            input_dim=input_type_size,
            output_dim=hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="input_type_embeddings",
        )

        self.layer_normalization = tf.keras.layers.LayerNormalization(name="layer_normalization", dtype="float32")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input_tensor):
        input_word_ids = input_tensor["input_word_ids"]
        input_type_ids = input_tensor["input_type_ids"]
        position_ids = tf.expand_dims(tf.range(tf.shape(input_word_ids)[-1]), 0)

        input_word_embedding = self.input_word_embeddings(input_word_ids)
        position_embedding = self.position_embeddings(position_ids)
        input_type_embedding = self.input_type_embeddings(input_type_ids)

        embeddings = tf.add(tf.add(input_word_embedding, position_embedding), input_type_embedding)

        embeddings = self.layer_normalization(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        intermediate_activation: Union[Callable, str],
        dropout_rate: float,
        attention_probs_dropout_rate: float,
        initializer_range: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.multihead_attention = MultiheadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout_rate=dropout_rate,
            attention_probs_dropout_rate=attention_probs_dropout_rate,
            initializer_range=initializer_range,
            name="multihead_attention",
        )

        self.mha_layer_normalization = tf.keras.layers.LayerNormalization(name="mha_layer_normalization", dtype="float32")

        self.intermediate_layer = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    intermediate_size,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
                    name="intermediate_1",
                ),
                tf.keras.layers.Activation(intermediate_activation, name="intermediate_act", dtype="float32"),
                tf.keras.layers.Dense(
                    hidden_size,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
                    name="intermediate_2",
                ),
                tf.keras.layers.Dropout(dropout_rate),
            ],
            name="intermediate_layer",
        )
        self.intermediate_layer_normalization = tf.keras.layers.LayerNormalization(name="intermediate_layer_normalization", dtype="float32")

    def call(self, hidden_state, input_mask=None):
        attention_output = self.multihead_attention(hidden_state, input_mask=input_mask)
        hidden_state = self.mha_layer_normalization(attention_output + hidden_state)
        hidden_state = tf.cast(hidden_state, attention_output.dtype)

        intermediate_output = self.intermediate_layer(hidden_state)
        hidden_state = self.intermediate_layer_normalization(intermediate_output + hidden_state)
        hidden_state = tf.cast(hidden_state, intermediate_output.dtype)
        return hidden_state


class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout_rate: float,
        attention_probs_dropout_rate: float,
        initializer_range: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert hidden_size % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // self.num_attention_heads
        self.scaling_factor = 1.0 / math.sqrt(float(self.attention_head_size))

        self.qkv = tf.keras.layers.Dense(
            hidden_size * 3,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="qkv",
        )
        self.attention_dropout = tf.keras.layers.Dropout(attention_probs_dropout_rate)
        self.output_projection = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="output_projection",
        )
        self.output_dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input_tensor, input_mask=None):
        query, key, value = tf.split(self.qkv(input_tensor), 3, axis=-1)
        batch_size = tf.shape(query)[0]

        query = self.transpose_for_scores(query, batch_size)
        key = self.transpose_for_scores(key, batch_size)
        value = self.transpose_for_scores(value, batch_size)

        attention_score = tf.matmul(query, key, transpose_b=True)
        attention_score *= tf.cast(self.scaling_factor, attention_score.dtype)

        if input_mask is not None:
            attention_score += input_mask

        attention_distribution = tf.keras.layers.Softmax(axis=-1, dtype="float32")(attention_score)
        attention_distribution = tf.cast(attention_distribution, dtype=query.dtype)
        attention_distribution = self.attention_dropout(attention_distribution)

        context_layer = tf.matmul(attention_distribution, value)
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, [batch_size, -1, self.num_attention_heads * self.attention_head_size])

        output = self.output_projection(context_layer)
        output = self.output_dropout(output)
        return output

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
        return tf.transpose(x, [0, 2, 1, 3])


def get_activation_fn(activation: str):
    if activation == "gelu":
        return tf.nn.gelu
    if activation == "relu":
        return tf.nn.relu

    return activation
