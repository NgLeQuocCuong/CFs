from tensorflow.keras.layers import Input, Mutiply, Dense, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.metrics import RootMeanSquaredError
from functools import reduce

class CFs():
    def __init__(self):
        self.model = None

    def model_info(self):
        if self.model:
            self.model.summary()
            return tf.keras.utils.plot_model(self.model, to_file='model.png')

    def fit(self, inputs, label, epochs=10):
        history = self.model.fit(inputs, label, epochs=epochs, verbose=1)
        pd.Series(history.history['loss']).plot(logy=False)
        plt.xlabel("Epoch")
        plt.ylabel("Training Error")

    def test(self, inputs, label):
        self.model.evaluate(inputs, label, batch_size=1)

    def _create_inputs(self, user_size, item_size):
        u_input = Input(shape=[n_item])
        i_input = Input(shape=[n_user])
        return [u_input, i_input]

    def _create_mlp(self, input, layers_size=[], dropout=0, activation='relu'):
        layers = [keras.layers.Dense(size, activation=activation) for size in layers_size]
        return reduce(lambda last, current: keras.layers.Dropout(dropout)(current(last)) if dropout else current(last) , layers, input)


class GMF(CFs):
    def create_model(self, n_user, n_item, n_factors=16):
        u_input = Input(shape=[n_item])
        u_embedding = keras.layers.Embedding(n_item, n_latent_factors)(u_input)
        u_vec = keras.layers.Flatten(name='Flatten_'+names[0])(u_embedding)

        i_input = Input(shape=[n_user])
        i_embedding = keras.layers.Embedding(n_user, n_latent_factors)(i_input)
        i_vec = keras.layers.Flatten(name='Flatten_'+names[1])(i_embedding)
        mutiply = Mutiply()([u_vec, i_vec])
        output = Dense(1)(mutiply)
        self.model = Model([u_input, i_input], output, name='GMF')
        self.model.compile(optimizer='adam', loss='mse',
                           metrics=[RootMeanSquaredError()])


class DeepCF(CFs):
    def create_model(self, user_size=100, item_size=100, representation_layers=[], embedding_size=16, matching_layers = [32], activation='relu'):
        inputs = self._create_inputs(user_size, item_size)
        representation_model = self._create_representation_model(inputs, representation_layers, activation)
        matchingfunction_model = self._create_matchingfunction_model(inputs, embedding_size,  matching_layers, activation)
        fusion_layer = Concatenate()([representation_model, matching_layers])
        output = Dense(1, activation=activation)(fusion_layer)
        self.model = Model(inputs, output, name='DeepCF')
        self.model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])
        
    def _create_representation_model(self, inputs, representation_layers, activation='relu'):
        user_latent_factor = self.create_mlp(inputs[0], representation_layers, dropout=0.1)
        item_latent_factor = self.create_mlp(inputs[1], representation_layers, dropout=0.1)
        return Mutiply()([user_latent_factor, item_latent_factor])

    def _create_matchingfunction_model(self, inputs, embedding_size=16, matching_layers = [32], activation='relu'):
        user_latent_factor = Dense(embedding_size, activation=activation)(inputs[0])
        item_latent_factor = Dense(embedding_size, activation=activation)(inputs[1])
        concat = Concatenate()[user_latent_factor, item_latent_factor]
        return self._create_mlp(input, matching_layers, dropout=0.1)


