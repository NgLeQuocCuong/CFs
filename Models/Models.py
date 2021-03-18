from tensorflow.keras.layers import Input, Mutiply, Dense
from tensorflow.keras import Model
from tensorflow.keras.metrics import RootMeanSquaredError
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
        self.model.evaluate(inputs, label, self._test.rating, batch_size=1)

class GMF(CFs):
    def create_model(self, n_user, n_item, n_factors = 16):
        u_input = Input(shape=[n_item])
        u_embedding = keras.layers.Embedding(n_item, n_latent_factors)(u_input)
        u_vec = keras.layers.Flatten(name='Flatten_'+names[0])(u_embedding)
 
        i_input = Input(shape=[n_user])
        i_embedding = keras.layers.Embedding(n_user, n_latent_factors)(i_input)
        i_vec = keras.layers.Flatten(name='Flatten_'+names[1])(i_embedding)
        mutiply = Mutiply()([u_vec, i_vec])
        output = Dense(1)(mutiply)
        self.model = Model([u_input, i_input], output, name='GMF')
        self.model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])



        

        
        