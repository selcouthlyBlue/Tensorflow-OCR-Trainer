import tensorflow as tf


class CustomHook(tf.train.SessionRunHook):
    def __init__(self, model_fn, params, input_fn, checkpoint_dir,
                 every_n_steps=1):
        self._estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            params=params,
            model_dir=checkpoint_dir
        )
        self._input_fn = input_fn
        self._every_n_steps = every_n_steps

    def after_run(self, run_context, run_values):
        self._estimator.evaluate(
            self._input_fn
        )
