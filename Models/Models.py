from tensorflow.keras.layers import Input, Multiply, Dense, Concatenate, Dropout, Layer
from tensorflow.keras import Model
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from functools import reduce
from pandas import Series
class CFs():
    def __init__(self):
        self.backup_path = 'training/backup.ckpt'
        self.cp_callback = ModelCheckpoint(filepath=self.backup_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    def model_info(self):
        if self.model:
            self.model.summary()
            return plot_model(self.model, to_file='model.png')

    def fit(self, inputs, label, epochs=10, verbose=1):
        self.model.fit(inputs, label, epochs=epochs, verbose=verbose, callbacks=[self.cp_callback])
    
    def load(self):
        self.model.load_weights(self.backup_path)

    def test(self, inputs, label):
        self.model.evaluate(inputs, label, batch_size=1)

    def _create_inputs(self, user_size, item_size):
        u_input = Input(shape=[user_size])
        i_input = Input(shape=[item_size])
        return [u_input, i_input]

    def _create_mlp(self, input, layers_size=[], dropout=0, activation='relu'):
        layers = [Dense(size, activation=activation) for size in layers_size]
        return reduce(lambda last, current: Dropout(dropout)(current(last)) if dropout else current(last) , layers, input)


class DeepCF(CFs):
    def __init__(self, user_size=100, item_size=100, representation_layers=[], embedding_size=16, matching_layers = [32], activation='relu'):
        self.backup_path = f'./training/deepcf_{str(representation_layers)}_{str([embedding_size]+matching_layers)}/backup.ckpt'
        self.cp_callback = ModelCheckpoint(filepath=self.backup_path, save_weights_only=True, verbose=0)
        inputs = self._create_inputs(user_size, item_size)
        representation_model = self._create_representation_model(inputs, representation_layers, activation)
        matchingfunction_model = self._create_matchingfunction_model(inputs, embedding_size,  matching_layers, activation)
        fusion_layer = Concatenate()([representation_model, matchingfunction_model])
        output = Dense(1, activation=activation)(fusion_layer)
        self.model = Model(inputs, output, name='DeepCF')
        self.model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])
        
    def _create_representation_model(self, inputs, representation_layers, activation='relu'):
        user_latent_factor = self._create_mlp(inputs[0], representation_layers, dropout=0.1)
        item_latent_factor = self._create_mlp(inputs[1], representation_layers, dropout=0.1)
        return Multiply()([user_latent_factor, item_latent_factor])

    def _create_matchingfunction_model(self, inputs, embedding_size=16, matching_layers = [32], activation='relu'):
        user_latent_factor = Dense(embedding_size, activation=activation)(inputs[0])
        item_latent_factor = Dense(embedding_size, activation=activation)(inputs[1])
        concat = Concatenate()([user_latent_factor, item_latent_factor])
        return self._create_mlp(concat, matching_layers, dropout=0.1)


