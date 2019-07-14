# Implements client for the advisor hyper parameter search

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app, flags
import os
import json
from advisor_client.client import AdvisorClient
from chronos.manager import Manager

FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="config",
    default=None,
    help="Path to the chronos.json configuration file"
)

ALGORITHMS = [
    "RandomSearch",
    "BayesianOptimization",
    "TPE",
    "SimulateAnneal",
    "QuasiRandomSearch",
    "ChocolateRandomSearch",
    "ChocolateBayes",
    "CMAES",
    "MOCMAES"
]


def main(unused_argv):

    client = AdvisorClient()

    with open(FLAGS.config, "r") as cfg:
        config = json.load(cfg)

    assert "study_name" in config, "The config needs a 'study_name' parameter (str)"
    study_name = config.pop("study_name")
    study_config = config.pop("hyper-parameter-config")
    algorithm = config.pop("algorithm")
    fixed_parameter_config = config.pop("fixed-parameter-config")
    exp_dir = config.pop("study_dir")

    if not os.path.isdir(os.path.join(exp_dir, study_name)):
        os.mkdir(os.path.join(exp_dir, study_name))

    assert algorithm in ALGORITHMS, "algorithm {} is not supported. Select one of {}".format(
        algorithm,
        ", ".join([algo for algo in ALGORITHMS])
    )

    study = client.get_or_create_study(study_name=study_name, study_configuration=study_config)
    print(study)

    max_trials = study_config["maxTrials"]
    for i in range(max_trials):
        trial = client.get_suggestions(study.name, 1)[0]
        parameter_value_dict = json.loads(trial.parameter_values)

        parameter_value_dict["skip_gru_units"] = int(parameter_value_dict["skip_gru_units"])
        parameter_value_dict["gru_units"] = int(parameter_value_dict["gru_units"])
        parameter_value_dict["cnn_batch_normalization"] = int(parameter_value_dict["cnn_batch_normalization"])

        merged_config = {**fixed_parameter_config, **parameter_value_dict}

        if os.path.isdir(os.path.join(exp_dir, study_name, "trial_{}".format(i))):
            raise RuntimeError("Trial Exists")
        else:
            os.mkdir(os.path.join(exp_dir, study_name, "trial_{}").format(i))
        merged_config["model_dir"] = os.path.join(exp_dir, study_name, "trial_{}".format(i))
        manager = Manager(**merged_config)
        ret = manager.run()
        # corr_score = ret["corr"]
        rmse_score = ret["rmse"]
        metric = rmse_score
        trial = client.complete_trial_with_one_metric(trial, metric)
        print(trial)

    best_trial = client.get_best_trial(study.name)
    print("Best trial: {}".format(best_trial))


if __name__ == '__main__':
    app.run(main)
