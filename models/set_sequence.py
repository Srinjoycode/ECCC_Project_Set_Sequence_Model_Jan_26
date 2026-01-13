import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SetModule(layers.Layer):
    """
    Implements the permutation-invariant Set Module:
    F_t = rho( mean( phi(x_{t,i}) ) )
    """

    def __init__(self, latent_dim, hidden_dim=64, activation='relu', dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)

        # Phi (Encoder): Processes each unit (grid point) independently
        # Input: (Batch, Time, Grid_Points, Features) -> Applied to last dim
        self.phi = keras.Sequential([
            layers.Dense(hidden_dim, activation=activation),
            layers.Dropout(dropout_rate),
            layers.Dense(hidden_dim, activation=activation),
        ], name='phi_encoder')

        # Rho (Projector): Processes the aggregated summary
        self.rho = keras.Sequential([
            layers.Dense(hidden_dim, activation=activation),
            layers.Dropout(dropout_rate),
            layers.Dense(latent_dim, activation=activation),  # Output matches your desired LATENT_DIM
        ], name='rho_projector')

    def call(self, inputs, mask=None):
        # inputs shape: (Batch, Time, Grid_Points, Features)

        # 1. Encode each grid point independently
        # We use TimeDistributed to apply Phi to every timestep and grid point
        # Reshape to (Batch*Time*Grid, Features) happens internally or we can force it
        x_encoded = self.phi(inputs)  # Shape: (Batch, Time, Grid_Points, Hidden)

        # 2. Aggregation (Deep Sets Pooling)
        # We average over the Grid_Points dimension (axis=2)
        if mask is not None:
            # If using padding/masking for variable grid sizes in a single batch
            # mask shape: (Batch, Time, Grid_Points)
            mask_expanded = tf.cast(mask[:, :, :, tf.newaxis], x_encoded.dtype)
            x_encoded = x_encoded * mask_expanded
            sum_x = tf.reduce_sum(x_encoded, axis=2)
            count_x = tf.reduce_sum(mask_expanded, axis=2) + 1e-8
            set_summary = sum_x / count_x
        else:
            set_summary = tf.reduce_mean(x_encoded, axis=2)

        # 3. Projection
        # Shape: (Batch, Time, Latent_Dim)
        output = self.rho(set_summary)
        return output


def build_set_sequence_model(
        seq_length,
        n_features,  # Number of variables per grid point (e.g., 5: prcp, temp, etc.)
        latent_dim,  # Size of the basin summary vector
        lstm_units=128,
        lstm_layers=2,
        dropout_rate=0.3,
        output_horizon=1
):
    """
    Constructs the end-to-end Set-Sequence Model.
    """
    # Input Shape: (Sequence_Length, None, n_features)
    # The 'None' allows this model to accept any number of grid points/substations!
    inputs = keras.Input(shape=(seq_length, None, n_features), name='set_input')

    # 1. Set Layer (Spatial Aggregation)
    # Processes (Batch, Time, Grid, Feat) -> (Batch, Time, Latent)
    set_module = SetModule(latent_dim, dropout_rate=dropout_rate)
    latent_sequence = set_module(inputs)

    # 2. Sequence Module (Temporal Dynamics)
    # This matches your existing LSTM logic but takes the learned summary as input
    x = latent_sequence
    for i in range(lstm_layers):
        return_sequences = (i < lstm_layers - 1)
        x = layers.LSTM(lstm_units, return_sequences=return_sequences, dropout=dropout_rate)(x)

    # 3. Prediction Head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(output_horizon, activation='linear', name='flow_output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='set_sequence_model')
    return model