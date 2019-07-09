# implements the model according to the LSTnet algorithm

import gin
import tensorflow as tf
from chronos.metrics import rmse, corr


@gin.configurable
class LSTnet(object):
    def __init__(self,
                 skip,
                 highway,
                 window,
                 num_time_series,
                 cnn_filters,
                 cnn_kernel_size,
                 cnn_kernel_init=tf.compat.v1.variance_scaling_initializer,
                 cnn_dropout=0.2,
                 cnn_batch_normalization=False,
                 gru_units=100,
                 skip_gru_units=5,
                 learning_rate=1e-3,
                 optimizer="SDG",
                 weight_regularization=1e-4,
                 clip_gradients=10
                 ):
        """
        Construct LSTnet

        Args:
            skip(int): Number of time slots to skip for the skip connection RNN.
            window(int): Size of the time window of the input.
            num_time_series(int): The number of time-series in the input
            highway(int): Number of time slots to consider for the auto regressive layer. If zero no highway is used.
            cnn_filters(int): Number of filter for the CNN layer. If 0, no CNN is applied
            cnn_kernel_size(int): Size of the zero dimension of the cnn kernel.
                                  The other dimension is given by the number of time series variables.
                                  The resulting filter size is cnn_kernel_size x time_series
            cnn_kernel_init(func): Initializer for the cnn kernels
            cnn_dropout(float): In range [0, 1]
            cnn_batch_normalization(bool): Whether to apply batch norm after the cnn encoder
            gru_units(int): Number of units for the GRU cell
            skip_gru_units(int): Number of units for the skip gru rnn cell
            learning_rate(float): Learning rate of the optimizer
            optimizer(str): One of 'SDG', 'RMSProb' or 'Adam'
            weight_regularization(float): Factor to multiply the weight loss with before adding it to
                                          the prediction loss
            clip_gradients(float): Norm to clip gradient to.
        """
        self.window = window
        self.skip = skip
        self.num_time_series = num_time_series
        assert highway < window, "Highway larger than window is not possible"
        assert skip < window, "Skip larger than window is no possible"

        ##############
        # CNN PARAMS #
        ##############
        assert 0 <= cnn_dropout <= 1, "CNN dropout parameter is out of range"
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_kernel_init = cnn_kernel_init
        self.cnn_dropout = cnn_dropout
        self.cnn_batch_normalization = cnn_batch_normalization

        ##############
        # GRU PARAMS #
        ##############
        self.gru_units = gru_units
        self.skip_gru_units = skip_gru_units

        ######
        # AR #
        ######
        self.highway = highway

        ############
        # TRAINING #
        ############
        self.learning_rate = learning_rate
        assert optimizer in ['SDG', 'RMSProp', 'Adam'], "invalid optimizer name {} must be " \
                                                        "one of 'SDG', 'RMSProb', 'Adam'".format(optimizer)
        self.optimizer = optimizer
        self.weight_regularization = weight_regularization
        self.clip_gradient = clip_gradients

    def _convolutional_decoder(self, inputs, mode):
        """
        Apply convolational layers to the input
        We convolve over a matrix of shape window(time) x feature_dim
        This operation can learn short term temporal dependencies since we convolve over time.

        Args:
            inputs(Tensor): Shape is batch_size, window, time_series_vars
            mode(tf.estiamtor.MODE_KEY): The mode.

        Returns:
            outputs(Tensor): Shape is batch_size, window, cnn_filters
        """
        input_shape = inputs.get_shape().as_list()
        # make shape batch_size, window, time_series_vars, 1
        # 1 is the channel dim the conv layer requires
        outputs = tf.expand_dims(inputs, axis=3)

        # the convolution is applied over window and time_series_vars.
        # the kernel size equals self.cnn_kernel_size x time_series_vars
        # This means the filters don't 'stride' along the time_series_vars dimension
        # Consequently this dimension is always size 1 and will be removed before returning.
        # After this operation inputs has shape batch_size, window, 1, self.cnn_filters
        outputs = tf.keras.layers.Conv2D(
            filters=self.cnn_filters,
            kernel_size=(self.cnn_kernel_size, outputs.get_shape().as_list()[2]),
            data_format="channel_last",
            kernel_initializer=self.cnn_kernel_init,
            activation=tf.nn.relu  # from paper
        )(outputs)
        # remove the dimension with size 1
        outputs = tf.squeeze(outputs, axis=2)

        training = bool(mode == tf.estimator.ModeKeys.TRAIN)
        if self.cnn_batch_normalization:
            outputs = tf.keras.layers.BatchNormalization(axis=3)(outputs, training=training)
        outputs = tf.keras.layers.Dropout(rate=self.cnn_dropout)(outputs, training=training)

        # since we use padding = 'valid' we might lose some time slots.
        # we concatenate window - kernel + 1 zeros to the output
        # We put the additional zeros to the beginning of the time axis
        outputs = tf.concat(
            [
                tf.zeros(shape=[input_shape[0], self.window - self.cnn_kernel_size + 1, input_shape[1]]),
                outputs
            ],
            axis=1
        )

        return outputs

    def _pre_skip_layer(self, inputs):
        """
        Reshape the input to prepare for the skip RNN.

        First, we calculate the number of time steps we use when only considering every skip time slot
        from the original input.
        Lets say the original input contains hourly data and we use a window size of one week.
        Lets say we want to use a skip value of one day (assuming we have daily periodicities)
        This means for each hour of the day we get 7 values over course of the week.
        For each of the 7 values we have #time-series-variables entries.
        I.e. we want a tensor of shape batch_size, 24, 7, #time-series-variables
        The 7 is the time dimension from which we want to learn temporal dependencies.
        The Rnn Cell accepts inputs of shape batch_size, time, feature.
        We reshape our tensor to shape [batch_size * 24, 7, #time-series-variables] to get this shape.
        The RNN cell will then roll over the 7 days for each hour 'slot' in the day for each element in the batch

        Args:
            inputs(Tensor): Has shape batch_size, window, time_series_vars

        Returns:
            outputs(Tensor): Has shape [batch_size * skip, window/skip, time_series_vars]
        """
        input_shape = inputs.get_shape().as_list()
        # the number of time slots when only considering every skip entry of the input time window
        num_slots = int(self.window/self.skip)

        output = inputs[:, -num_slots*self.skip:, :]  # get the last num_slots * skip time slots

        # unstack the time slots into a new dimension
        output = tf.reshape(output, [input_shape[0], num_slots, self.skip, input_shape[2]])

        # permotue skip and slot axis
        output = tf.transpose(output, [0, 2, 1, 3])

        # merge the batch axis and the skip axis
        output = tf.reshape(output, [input_shape[0]*num_slots, self.skip, input_shape[2]])

        return output

    def _post_skip_layer(self, inputs, original_batch_size):
        """
        Reshape the input after the skip connection

        The output from the rnn skip layer has shape batch_size*skip, self.skip_gru_units.
        We want Tensor of shape batch_size, skip * skip_gru_units

        Args:
            inputs(Tensor): Has shape batch_size*skip, self.skip_gru_units
            original_batch_size(int): The original batch size

        Returns:
            outputs(Tensor): Has shape batch_size, skip * self.skip_gru_units
        """
        input_shape = inputs.get_shape().as_list()

        # has shape [batch_size, skip, self.skip_gru_units]
        outputs = tf.reshape(inputs, [original_batch_size, self.skip, input_shape[1]])

        # has shape [batch_size, skip * self.skip_gru_units]
        outputs = tf.reshape(outputs, [original_batch_size, self.skip * input_shape[1]])

        return outputs

    def _pre_ar_layer(self, inputs):
        """
        Reshapes inputs for auto regressive component (highway)

        Args:
            inputs(Tensor): Has shape [batch_size, window, #time-series-variables]

        Returns
            output(Tensor): Has shape []
        """
        input_shape = inputs.get_shape().as_list()

        # pick the last highway timeslots
        output = inputs[:, -self.highway:, :]

        # permute axis
        output = tf.transpose(output, [0, 2, 1])

        # merge the time-series into the batch size and isolate the highway axis
        output = tf.reshape(output, [input_shape[0] * input_shape[2], self.highway])

        return output

    def _post_ar_layer(self, inputs, original_batch_size):
        """
        Reshape inputs after auto regressive component (highway)

        Args:
            inputs(Tensor): Has shape batch_size * #time-series-variables, 1
            original_batch_size(int): training batch size

        Returns:
            output(Tensor): Has shape batch_size, # time-series-variables

        """
        outputs = tf.reshape(inputs, [original_batch_size, self.num_time_series])
        return outputs

    def model_fn(self, features, labels, mode, params):
        """ Implement the tensorflow graph """

        inputs = features
        if isinstance(features, dict):
            inputs = features["features"]

        initial_input_shape = inputs.get_shape().as_list()

        # inputs is of shape batch_size x window x time_series_vars
        if self.cnn_filters > 0 and self.cnn_kernel_size > 0:
            # inputs is of shape batch_size x window x self.cnn_filters
            cnn_out = self._convolutional_decoder(inputs, mode)
        else:
            cnn_out = inputs
        # recurrent out has shape batch_size x self.gru_units
        _, recurrent_out = tf.keras.layers.GRU(
            units=self.gru_units,
            activation=tf.nn.relu,
            return_sequences=False,
            return_state=True
        )(cnn_out)

        if self.skip > 0:

            recurrent_out_skip = self._pre_skip_layer(cnn_out)
            _, recurrent_out_skip = tf.keras.layers.GRU(
                units=self.skip_gru_units,
                activation=tf.nn.relu,
                return_sequences=False,
                return_state=True
            )(recurrent_out_skip)
            recurrent_out_skip = self._post_skip_layer(recurrent_out_skip, initial_input_shape[0])
            recurrent_out = tf.concat([recurrent_out, recurrent_out_skip], axis=1)  # concat outputs

        flat_out = tf.keras.layers.Flatten()(recurrent_out)
        dense_out = tf.keras.layers.Dense(initial_input_shape[2])(flat_out)

        if self.highway > 0:
            ar_out = self._pre_ar_layer(inputs)
            ar_out = tf.keras.layers.Flatten()(ar_out)
            ar_out = tf.keras.layers.Dense(1)(ar_out)
            ar_out = self._post_ar_layer(ar_out, initial_input_shape[0])

            dense_out = tf.keras.layers.Add()([dense_out, ar_out])

        # has shape batch_size, #time-series-variable
        logits = dense_out

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "prediction": logits
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        loss = tf.keras.losses.mse(y_true=labels, y_pred=logits)

        if self.weight_regularization > 0:
            loss += self.weight_regularization * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables()
                 if 'batch_normalization' not in v.name]
            )

        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            if self.optimizer == "SDG":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer == "RMSProb":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):

                grads_and_vars = optimizer.compute_gradients(loss)

                grads_and_vars = [(tf.clip_by_norm(grad, self.clip_gradients), var) for grad, var in grads_and_vars]

                train_op = optimizer.apply_gradients(grads_and_vars)

        eval_metrics = None
        if mode == tf.estimator.ModeKeys.EVAL:
            """ Eval metrics """
            eval_metrics = {
                "rmse": rmse(labels=labels, predictions=logits),
                "corr": corr(labels=labels, predictions=logits)
            }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metrics
        )
