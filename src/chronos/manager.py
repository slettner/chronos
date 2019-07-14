# Implements a manger which can create scenarios from configuration
# Each inputs type has its own manager since they need different parameters

import os
import json
import gin
import jinja2
import tensorflow as tf

from chronos.trainer import Chronos


PATH_TO_GIT_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
TEMPLATE_FOLDER = os.path.join(PATH_TO_GIT_REPO, "templates")


class ExperimentState:
    FINISHED = 0
    UNFINISHED = 1
    NOT_STARTED = 2


class Manager(object):

    """ Creates scenarios from configuration """

    def __init__(
            self,
            window,
            horizon,
            num_time_series,
            skip,
            highway,
            model_dir,
            train_steps,
            steps_per_eval,
            data_path,
            template_file="template-numpy-input.gin",
            train_percentage=0.1,
            validation_percentage=0.2,
            batch_size=128,
            cnn_filters=100,
            cnn_kernel_size=6,
            cnn_drop_out=0.2,
            cnn_batch_normalization=False,
            gru_units=100,
            skip_gru_units=5,
            learning_rate=0.001,
            optimizer="Adam",
            weight_regularization=0.0001,
            clip_gradients=10,
            plot=True,
            plot_series=None,
            eval_test_set=False,
            export=False,
            seed=42
    ):
        """

        Args:
            window:
            horizon:
            num_time_series:
            skip:
            highway:
            model_dir:
            train_steps:
            steps_per_eval:
            data_path:
            template_file:
            train_percentage:
            validation_percentage:
            batch_size:
            cnn_filters:
            cnn_kernel_size:
            cnn_drop_out:
            cnn_batch_normalization:
            gru_units:
            skip_gru_units:
            learning_rate:
            optimizer:
            weight_regularization:
            clip_gradients:
            plot:
            plot_series:
            eval_test_set:
            export:
            seed:
        """
        tf.set_random_seed(seed=seed)
        self.window = window
        self.horizon = horizon
        self.num_time_series = num_time_series
        self.skip = skip
        self.highway = highway
        self.model_dir = model_dir
        self.train_steps = train_steps
        self.steps_per_eval = steps_per_eval
        self.data_path = data_path
        self.template_file = template_file
        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage
        self.batch_size = batch_size
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_drop_out = cnn_drop_out
        self.cnn_batch_normalization = cnn_batch_normalization
        self.gru_units = gru_units
        self.skip_gru_units = skip_gru_units
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_regularization = weight_regularization
        self.clip_gradients = clip_gradients
        self.plot = plot
        self.plot_series = plot_series
        self.eval_test_set = eval_test_set
        self.export = export

    def prepare_experiment(self):
        """

        Returns:

        """
        with open(os.path.join(self.model_dir, "config.json"), "w") as fg:
            json.dump(self.__dict__, fg)

        # create .gin file
        file_loader = jinja2.FileSystemLoader(TEMPLATE_FOLDER)
        environment = jinja2.Environment(loader=file_loader)
        template = environment.get_template(self.template_file)
        config = self.__dict__
        config.pop("template_file")
        output = template.render(**config)
        with open(os.path.join(self.model_dir, "config.gin"), "w") as fh:
            fh.write(output)

    def run(self):
        """ Run the experiment """
        self.prepare_experiment()

        tf.logging.set_verbosity(tf.logging.INFO)
        gin.parse_config_file(os.path.join(self.model_dir, "config.gin"))

        chronos = Chronos()
        final_eval = chronos.run()
        final_eval = {
            "rmse": float(final_eval["rmse"]),
            "corr": float(final_eval["corr"])
        }

        with open(os.path.join(self.model_dir, "final_eval.json"), "w") as gh:
            json.dump(final_eval, gh)

        return final_eval

