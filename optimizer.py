from typing import Optional, List

import tensorflow as tf
import tensorflow_addons as tfa


class AdamW(tfa.optimizers.AdamW):
    def __init__(
        self,
        *args,
        decay_var_list: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._decay_var_list_kept = decay_var_list

    def minimize(self, *args, **kwargs):
        return super().minimize(*args, **kwargs, decay_var_list=self._decay_var_list_kept)

    def apply_gradients(self, *args, **kwargs):
        return super().apply_gradients(*args, **kwargs, decay_var_list=self._decay_var_list_kept)


class LinearWarmupAndDecayScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, rate, warmup_steps, total_steps, name=None):
        super().__init__()

        self.rate = rate
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.name = name

    def __call__(self, step):
        with tf.name_scope("LinearWarmupAndDecayScheduler"):
            total_steps = tf.convert_to_tensor(self.total_steps, name="total_steps")
            warmup_steps = tf.convert_to_tensor(self.warmup_steps, name="warmup_steps")

            current_step = tf.cast(step + 1, warmup_steps.dtype)

            return self.rate * tf.cond(
                current_step < warmup_steps,
                lambda: self.warmup(current_step, warmup_steps),
                lambda: self.decay(current_step, total_steps, warmup_steps),
            )

    @tf.function
    def warmup(self, step, warmup_steps):
        return step / tf.math.maximum(tf.constant(1.0), warmup_steps)

    @tf.function
    def decay(self, step, total_steps, warmup_steps):
        return tf.math.maximum(tf.constant(0.0), (total_steps - step) / tf.math.maximum(tf.constant(1.0), total_steps - warmup_steps))

    def get_config(self):
        return {
            "rate": self.rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "name": self.name,
        }
