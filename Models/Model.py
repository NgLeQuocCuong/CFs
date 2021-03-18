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

    def test(self):
        self.model.evaluate((self._test.customer_id, self._test.product_id), self._test.rating, batch_size=1)