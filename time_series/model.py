import tensorflow as tf

class DnnModel(object):
    '''
    深度神经网络
    '''
    def __init__(self, config, input_size):
        self._output_path = config['output_path']
        self._model_path = config['model_path']
        self._hidden_size = config['hidden_size']
        self.config = config
        self.input_size = input_size

    def model_structure(self):
        '''
        模型结构
        '''
        model = tf.keras.Sequential()
        input_size = self.input_size
        for idx, hidden_size in enumerate(self._hidden_size):
            model.add(tf.keras.layers.Dense(hidden_size, activation='relu', input_dim=input_size))
            input_size = hidden_size
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.config['learning_rate']))
        model.summary()
        return model

    def lstm_structure(self):

        model = tf.keras.Sequential()
        input_size = self.input_size
        for idx, hidden_size in enumerate(self._hidden_size):
            model.add(tf.keras.layers.LSTM(hidden_size, activation='relu', input_shape=(1, input_size)))
            # input_size = hidden_size
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.config['learning_rate']))
        model.summary()
        return model

    def train(self, model, train_df, val_df):
        '''
        模型训练
        '''
        model_history = model.fit(train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values,
                  validation_data=(val_df.iloc[:, :-1].values, val_df.iloc[:, -1].values),
                  epochs=self.config['epochs'],
                  batch_size=self.config['batch_size'])
        return model_history, model

    def predict(self, model, test_df):
        '''
        模型预测
        '''
        self.prediction = model.predict(test_df.iloc[:, :-1].values)

    def train_lstm(self, model, train_df, val_df):
        train_X, train_y = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
        val_X, val_y = val_df.iloc[:, :-1].values, val_df.iloc[:, -1].values

        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))

        model_history = model.fit(train_X, train_y,
                                  validation_data=(val_X, val_y),
                                  epochs=self.config['epochs'],
                                  batch_size=self.config['batch_size'])
        return model_history, model

    def predict_lstm(self, model, test_df):
        '''
        模型预测
        '''
        test_X, test_y = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


        self.prediction = model.predict(test_X)


