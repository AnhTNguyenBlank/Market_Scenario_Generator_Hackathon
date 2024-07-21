import tensorflow as tf
import keras
from keras import layers
import tensorflow_probability as tfp

from abc import ABC, abstractmethod

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class C_VariationalAutoencoder(keras.Model, ABC):
    def __init__(self,
            seq_len,
            feat_dim,
            latent_dim,
            reconstruction_wt,
            **kwargs):
        super(C_VariationalAutoencoder, self).__init__(**kwargs)

        # number of points in one simulated result
        self.seq_len = seq_len

        # number of input features - may be just log daily return or in multivariate case, the number of assets, set to 1
        self.feat_dim = feat_dim

        # number of dimensions in the latent space, set to 1
        self.latent_dim = latent_dim

        self.reconstruction_wt = reconstruction_wt

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        # mse distance between original and generated sample
        self.reconstruction_loss_tracker_1 = keras.metrics.Mean(name="reconstruction_loss_1")

        # probability distance between original and generated sample
        self.reconstruction_loss_tracker_2 = keras.metrics.Mean(name="reconstruction_loss_2")

        # probability distance between the latent space variable Z and the Gaussian distribution
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()


    def call(self, X):
        X_cur, X_cond = X
        z_mean, _, _ = self.encoder(X_cur)
        x_decoded = self.decoder([z_mean, X_cond])
        if len(x_decoded.shape) == 1:
            x_decoded = x_decoded.reshape((1, -1))
        return x_decoded


    def _get_encoder(self):
        self.encoder_cur_inputs = keras.layers.Input(shape=(self.seq_len, self.feat_dim), name='encoder_curr_input')

        # masked_layer = keras.layers.Masking(mask_value=-1)(self.encoder_inputs)

        x = keras.layers.Flatten()(self.encoder_cur_inputs)
        x = keras.layers.Dense(100, activation = None, name = f'enc_dense_1')(x)

        z_mean = keras.layers.Dense(self.latent_dim, activation = None, name="z_mean")(x)
        z_log_var = keras.layers.Dense(self.latent_dim, activation = None, name="z_log_var")(x)

        encoder_output = Sampling()([z_mean, z_log_var])
        self.encoder_output = encoder_output

        encoder = keras.Model(self.encoder_cur_inputs, [z_mean, z_log_var, encoder_output], name="encoder")
        # encoder.summary()
        return encoder


    def _get_decoder(self):
        decoder_rand_inputs = keras.layers.Input(shape=(self.latent_dim), name='decoder_random')
        decoder_cond_inputs = keras.layers.Input(shape=(1,), name='decoder_cond_input')

        x1 = tf.keras.layers.RepeatVector(100)(decoder_cond_inputs)
        x1 = keras.layers.Flatten()(x1)

        x = keras.layers.concatenate([decoder_rand_inputs, x1])

        x = keras.layers.Dense(100,
                               activation = None,
                               name='decoder_dense',
                               kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                               bias_initializer=keras.initializers.Zeros())(x)

        # Asset 1
        ret_1 = keras.layers.Dense(5,
                                   activation = 'tanh',
                                   name='ret_1',
                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                   bias_initializer=keras.initializers.Zeros())(x)
        ret_1 = keras.layers.Reshape(target_shape=(self.seq_len, 1))(ret_1)

        vol_1 = keras.layers.Dense(5,
                                   activation = 'sigmoid',
                                   name='vol_1',
                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                   bias_initializer=keras.initializers.Zeros())(x)
        vol_1 = keras.layers.Reshape(target_shape=(self.seq_len, 1))(vol_1)

        # Asset 2
        ret_2 = keras.layers.Dense(5,
                                   activation = 'tanh',
                                   name='ret_2',
                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                   bias_initializer=keras.initializers.Zeros())(x)
        ret_2 = keras.layers.Reshape(target_shape=(self.seq_len, 1))(ret_2)

        vol_2 = keras.layers.Dense(5,
                                   activation = 'sigmoid',
                                   name='vol_2',
                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                   bias_initializer=keras.initializers.Zeros())(x)
        vol_2 = keras.layers.Reshape(target_shape=(self.seq_len, 1))(vol_2)

        # Asset 3
        ret_3 = keras.layers.Dense(5,
                                   activation = 'tanh',
                                   name='ret_3',
                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                   bias_initializer=keras.initializers.Zeros())(x)
        ret_3 = keras.layers.Reshape(target_shape=(self.seq_len, 1))(ret_3)

        vol_3 = keras.layers.Dense(5,
                                   activation = 'sigmoid',
                                   name='vol_3',
                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                   bias_initializer=keras.initializers.Zeros())(x)
        vol_3 = keras.layers.Reshape(target_shape=(self.seq_len, 1))(vol_3)

        # Asset 4
        ret_4 = keras.layers.Dense(5,
                                   activation = 'tanh',
                                   name='ret_4',
                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                   bias_initializer=keras.initializers.Zeros())(x)
        ret_4 = keras.layers.Reshape(target_shape=(self.seq_len, 1))(ret_4)

        vol_4 = keras.layers.Dense(5,
                                   activation = 'sigmoid',
                                   name='vol_4',
                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                   bias_initializer=keras.initializers.Zeros())(x)
        vol_4 = keras.layers.Reshape(target_shape=(self.seq_len, 1))(vol_4)

        # Asset 5
        ret_5 = keras.layers.Dense(5,
                                   activation = 'tanh',
                                   name='ret_5',
                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                   bias_initializer=keras.initializers.Zeros())(x)
        ret_5 = keras.layers.Reshape(target_shape=(self.seq_len, 1))(ret_5)

        vol_5 = keras.layers.Dense(5,
                                   activation = 'sigmoid',
                                   name='vol_5',
                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                   bias_initializer=keras.initializers.Zeros())(x)
        vol_5 = keras.layers.Reshape(target_shape=(self.seq_len, 1))(vol_5)

        # Concat
        self.decoder_outputs = keras.layers.concatenate([ret_1, vol_1, ret_2, vol_2, ret_3, vol_3, ret_4, vol_4, ret_5, vol_5,])

        decoder = keras.Model([decoder_rand_inputs, decoder_cond_inputs], self.decoder_outputs, name="decoder")
        # decoder.summary()
        return decoder


    def _get_reconstruction_loss(self, X, X_recons):

        def get_reconst_loss_by_axis(X, X_c, axis):
            x_r = tf.reduce_mean(X, axis = axis)
            x_c_r = tf.reduce_mean(X_recons, axis = axis)
            err = tf.math.squared_difference(x_r, x_c_r)
            loss = tf.reduce_sum(err)
            return loss

        # overall
        err = tf.math.squared_difference(X, X_recons)
        reconst_loss = tf.reduce_sum(err)

        reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=1)
        reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=2)

        return reconst_loss


    def summary(self):
        self.encoder.summary()
        self.decoder.summary()


    def train_step(self, X):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X[0])
            reconstruction = self.decoder([z, X[1]])

            # adjusted MSE
            reconstruction_loss_1 = self._get_reconstruction_loss(tf.cast(X[0], tf.float64), tf.cast(reconstruction, tf.float64))

            reconstruction_loss_2 = 0

            pred_ret_1 = tf.cast(reconstruction[:, :, 0], tf.float64)
            pred_vol_1 = tf.cast(reconstruction[:, :, 1], tf.float64)

            pred_ret_2 = tf.cast(reconstruction[:, :, 2], tf.float64)
            pred_vol_2 = tf.cast(reconstruction[:, :, 3], tf.float64)

            pred_ret_3 = tf.cast(reconstruction[:, :, 4], tf.float64)
            pred_vol_3 = tf.cast(reconstruction[:, :, 5], tf.float64)

            pred_ret_4 = tf.cast(reconstruction[:, :, 6], tf.float64)
            pred_vol_4 = tf.cast(reconstruction[:, :, 7], tf.float64)

            pred_ret_5 = tf.cast(reconstruction[:, :, 8], tf.float64)
            pred_vol_5 = tf.cast(reconstruction[:, :, 9], tf.float64)

            # True
            true_ret_1 = tf.cast(X[0][:, :, 0], tf.float64)
            true_vol_1 = tf.cast(X[0][:, :, 1], tf.float64)

            true_ret_2 = tf.cast(X[0][:, :, 2], tf.float64)
            true_vol_2 = tf.cast(X[0][:, :, 3], tf.float64)

            true_ret_3 = tf.cast(X[0][:, :, 4], tf.float64)
            true_vol_3 = tf.cast(X[0][:, :, 5], tf.float64)

            true_ret_4 = tf.cast(X[0][:, :, 6], tf.float64)
            true_vol_4 = tf.cast(X[0][:, :, 7], tf.float64)

            true_ret_5 = tf.cast(X[0][:, :, 8], tf.float64)
            true_vol_5 = tf.cast(X[0][:, :, 9], tf.float64)

            # PSI
            # reconstruction_loss = tf.reduce_sum(
            #         keras.losses.kl_divergence(X, reconstruction)
            #         + keras.losses.kl_divergence(reconstruction, X)
            #         + keras.losses.mean_absolute_error(X, reconstruction)
            #         + keras.losses.mse(X, reconstruction)
            #        )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(kl_loss, axis=1)

            reconstruction_loss_2 += tf.keras.losses.KLDivergence()(true_ret_1, pred_ret_1)
            reconstruction_loss_2 += tf.keras.losses.KLDivergence()(true_ret_2, pred_ret_2)
            reconstruction_loss_2 += tf.keras.losses.KLDivergence()(true_ret_3, pred_ret_3)
            reconstruction_loss_2 += tf.keras.losses.KLDivergence()(true_ret_4, pred_ret_4)
            reconstruction_loss_2 += tf.keras.losses.KLDivergence()(true_ret_5, pred_ret_5)

            reconstruction_loss_2 += tf.keras.losses.KLDivergence()(true_vol_1, pred_vol_1)
            reconstruction_loss_2 += tf.keras.losses.KLDivergence()(true_vol_2, pred_vol_2)
            reconstruction_loss_2 += tf.keras.losses.KLDivergence()(true_vol_3, pred_vol_3)
            reconstruction_loss_2 += tf.keras.losses.KLDivergence()(true_vol_4, pred_vol_4)
            reconstruction_loss_2 += tf.keras.losses.KLDivergence()(true_vol_5, pred_vol_5)


            true_corr = tfp.stats.correlation(X[0], sample_axis = 0, event_axis = -1)
            pred_corr = tfp.stats.correlation(reconstruction, sample_axis = 0, event_axis = -1)

            reconstruction_loss_2 += tf.reduce_mean(tf.math.squared_difference(tf.cast(true_corr, tf.float64),
                                                                               tf.cast(pred_corr, tf.float64)))

            total_loss = tf.cast(self.reconstruction_wt * reconstruction_loss_1, tf.float64) \
                            + tf.cast(reconstruction_loss_2, tf.float64) \
                            + tf.cast(kl_loss, tf.float64)


        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker_1.update_state(reconstruction_loss_1)
        self.reconstruction_loss_tracker_2.update_state(reconstruction_loss_2)

        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss_1": self.reconstruction_loss_tracker_1.result(),
            "reconstruction_loss_2": self.reconstruction_loss_tracker_2.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }