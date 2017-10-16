import keras
import pickle
import pickling

pickling.make_keras_picklable()

m = keras.models.Sequential()
m.add(keras.layers.Dense(10, input_shape=(10,)))
m.compile(optimizer='sgd', loss='mse')

pickle.dumps(m)

