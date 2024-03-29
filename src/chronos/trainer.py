# Implements main loop chronos

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import h5py
import gin
import tensorflow as tf
import os
import numpy as np
from chronos.data import MODE
from chronos.model import LSTnet
from chronos.plots import plot_prediction, plt


@gin.configurable
def serving_input_receiver_fn(window, num_time_series):
    """ For exporting models """
    example = tf.placeholder(
        dtype=tf.float32,
        shape=[None, window, num_time_series],
        name="input_tensor"
    )
    receiver_tensors = {"x": example}
    return tf.estimator.export.ServingInputReceiver(example, receiver_tensors)


@gin.configurable
class Chronos(object):

    """ End-to-End learning manager """

    def __init__(self, input_generator, model_dir, train_steps, steps_per_eval,
                 eval_test_set=False, export=False, plot=False, plot_series=None):
        """

        Args:
            input_generator(AbstractInputGenerator): Feed data
            model_dir(str): Save a file to this dir
            train_steps(int): Total amount of train steps. One batch is one step
            steps_per_eval(int): Periodicity of evaluating counted in train steps.
            eval_test_set(bool): Check Test Set Performance
            export(bool): Export to model
            plot(bool): Plot the prediction over training validation and test
            plot_series(list): List of time series vars to plot
        """
        self.input_generator = input_generator
        self.model_dir = model_dir
        self.train_steps = train_steps
        self.steps_per_eval = steps_per_eval
        self.eval_test_set = eval_test_set
        self.export = export
        self.plot = plot
        self.plot_series = plot_series
        if self.plot_series is None:
            self.plot_series = [0]

        self.step = []
        self.eval_score_rmse = []
        self.eval_score_corr = []

    def run(self):
        """ Execute the train eval loop """

        current_step = tf.train.get_checkpoint_state(self.model_dir)
        if current_step is None:
            current_step = 0
        else:
            current_step = int(os.path.basename(current_step.model_checkpoint_path).split('-')[1])

        steps_per_epoch = int(len(self.input_generator) // self.input_generator.batch_size)
        tf.logging.info(
            'Training for %d steps (%.2f epochs in total). Current step %d.',
            self.train_steps,
            self.train_steps / steps_per_epoch,
            current_step
        )

        train_input_fn = self.input_generator(mode=MODE.TRAIN)
        eval_input_fn = self.input_generator(mode=MODE.EVAL)
        test_input_fn = self.input_generator(mode=MODE.TEST)

        start_time_stamp = time.time()

        lst_net = LSTnet()  # params are set by gin
        estimator = tf.estimator.Estimator(
            model_fn=lst_net.model_fn,
            model_dir=self.model_dir
        )
        eval_results = None
        while current_step < self.train_steps:
            # Train for up to steps_per_eval number of steps.
            # At the end of training, a checkpoint will be written to --model_dir.
            next_checkpoint = min(current_step + self.steps_per_eval, self.train_steps)
            estimator.train(
                input_fn=train_input_fn, max_steps=next_checkpoint)
            current_step = next_checkpoint

            tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                            next_checkpoint, int(time.time() - start_time_stamp))

            # Evaluate the model on the most recent model in --model_dir.
            # Since evaluation happens in batches of --eval_batch_size, some images
            # may be excluded modulo the batch size. As long as the batch size is
            # consistent, the evaluated images are also consistent.
            tf.logging.info('Starting to evaluate.')
            eval_results = estimator.evaluate(input_fn=eval_input_fn, steps=1)
            self.step.append(current_step)
            self.eval_score_corr.append(eval_results["corr"])
            self.eval_score_rmse.append(eval_results["rmse"])
            tf.logging.info('Eval results at step %d: %s', next_checkpoint, eval_results)

            elapsed_time = int(time.time() - start_time_stamp)
            tf.logging.info('Finished training up to step %d. Elapsed seconds %d.', self.train_steps, elapsed_time)

        if self.eval_test_set:
            results = estimator.evaluate(input_fn=test_input_fn, steps=1)
            print(results)

        if self.export:
            estimator.export_savedmodel(
                export_dir_base=self.model_dir,
                serving_input_receiver_fn=serving_input_receiver_fn
            )

        if self.plot:
            start = time.time()
            tf.logging.info("Making Plots..")

            train_input_fn_predict = self.input_generator(mode=MODE.TRAIN, epochs=1)
            val_input_fn_predict = self.input_generator(mode=MODE.EVAL, epochs=1)
            test_input_fn_predict = self.input_generator(mode=MODE.TEST, epochs=1)

            train_predict = list(estimator.predict(input_fn=train_input_fn_predict))
            tf.logging.info("Predicted Train Set.")
            val_predict = list(estimator.predict(input_fn=val_input_fn_predict))
            tf.logging.info("Predicted Validation Set.")
            test_predict = list(estimator.predict(input_fn=test_input_fn_predict))
            tf.logging.info("Predicted Test Set.")

            train_predict = np.concatenate([np.expand_dims(x['prediction'], axis=0) for x in train_predict], axis=0)
            val_predict = np.concatenate([np.expand_dims(x['prediction'], axis=0) for x in val_predict], axis=0)
            test_predict = np.concatenate([np.expand_dims(x['prediction'], axis=0) for x in test_predict], axis=0)

            file = h5py.File("predictions.h5", "w")
            file["train"] = train_predict
            file["validation"] = val_predict
            file["test"] = test_predict
            file.close()

            plot_prediction(
                data=self.input_generator,
                train_predict=train_predict,
                validation_predict=val_predict,
                test_predict=test_predict,
                save_plot=self.model_dir,
                start_plot=0,
                end_plot=len(self.input_generator),
                series=self.plot_series
            )
            tf.logging.info("Prediction and Plotting took {}".format(time.time()-start))
        self.make_metric_plots()

        return eval_results

    def make_metric_plots(self):
        """ Plot the correlation and rmse metric over time """
        plt.subplots(1)
        plt.plot(self.step, self.eval_score_rmse)
        plt.xlabel("Train Steps")
        plt.ylabel("RMSE")
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "rmse.pdf"))

        plt.subplots(1)
        plt.plot(self.step, self.eval_score_corr)
        plt.xlabel("Train Steps")
        plt.ylabel("CORR")
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "corr.pdf"))


# def main(unused_argv):
#
#     tf.logging.set_verbosity(tf.logging.INFO)
#     gin.parse_config_file(FLAGS.config)
#
#     chronos = Chronos()  # params set by gin
#     chronos.run()
#
#
# if __name__ == '__main__':
#     app.run(main)

