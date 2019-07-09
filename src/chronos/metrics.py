# Implement custom metrics for the tensorflow estimator api

import tensorflow as tf
from tensorflow.python.framework import ops


__all__ = ['corr', 'rmse']


def corr(
        labels,
        predictions,
        weights=None,
        metrics_collections=None,
        updates_collections=None,
        name=None,
        percentage=0.05
):
    """
    Calculates the correlation between output and label

    Args:
        labels: The ground truth values, a `Tensor` whose shape matches
                `predictions`.
        predictions: The predicted values, a `Tensor` of any shape.
        weights: Optional `Tensor` whose rank is either 0, or the same rank as
                 `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
                 be either `1`, or the same as the corresponding `labels` dimension).
        metrics_collections: An optional list of collections that `accuracy` should
                             be added to.
        updates_collections: An optional list of collections that `update_op` should
                             be added to.
        name: An optional variable_scope name.
        percentage(float): The percentage of relative deviation to consider the sample as a correctly predicted
    Returns:

    """

    num1 = labels - tf.keras.backend.mean(labels, axis=0)
    num2 = predictions - tf.keras.backend.mean(predictions, axis=0)

    num = tf.reduce_mean(num1 * num2, axis=0)
    den = tf.keras.backend.std(labels, axis=0) * tf.keras.backend.std(predictions, axis=0)

    wrapped_correlation_metric, update_op = tf.metrics.mean(num/den)

    if metrics_collections:
        ops.add_to_collections(metrics_collections, wrapped_correlation_metric)

    if updates_collections:
        ops.add_to_collections(updates_collections, )

    return wrapped_correlation_metric, update_op


rmse = tf.metrics.root_mean_squared_error

