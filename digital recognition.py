import keras
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.astype('float32')
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size = (3, 3),
                              activation = 'relu',
                              input_shape = input_shape))
model.add(keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = keras.optimizers.Adadelta(),
              loss = keras.losses.categorical_crossentropy,
              metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 128, epochs = 20, validation_data = (x_test, y_test))

print(model.evaluate(x_test, y_test))
