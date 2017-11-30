# Gerard Naughton G00209309 train.py EmergingTechProject
# Code adapted from https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
# Comments were derived/learned and quoted from keras website: https://keras.io/
# Imports required for program
import keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Number of samples per gradient update
batch_size = 128
# Number of binary class matrices needed. Our outputs in binary. We have 10 different outputs 0-9
num_classes = 10
# Number of iterations the model does over the x and y data
epochs = 12

# input image dimensions (28pixels each)
img_rows, img_cols = 28, 28

# We then shuffle and split the data into x_train y_train(training set) and x_test y_test(test set). x = inputs and y = outputs
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# This if statements checks the way image data is layed out. Is it channel, row, col(channel_first) or row, col, channel(channel_last).
# It tests the first image and sets imput_shape to either channel_first or channel_last 
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Sets x_train to a float32 type as it is most efficient for keras on gpu and cpu
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Scales the inputs(x_train,y_test) to be in a range of 0-1
x_train /= 255
x_test /= 255

# Print out the x_train shape which should be (60000, 28, 28, 1)
print('x_train shape:', x_train.shape)

# Print the amount of train and test samples their are
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices. Allows us to match our binary matrices to our train and test answers(0 to 9)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create a sequential Model. A sequential model is a linear stack of layers. We can then add layers to our model using the add() method.
model = Sequential()

# We add Conv2D layer. Allows spatial convolution over images. This is a 2d convolution layer.
# It is the first layer in the model and it create a convolution kernal.
# All parameters are added to the layer
model.add(Conv2D(32, # 32 represents the amount of filters for your conv2d layer.
                kernel_size=(3, 3),# Kernal_size receives a list/tuple of 2 integers specifying width and height of a 2D convolution window. 
                activation='relu',# relu is a activations function. Activation functions allows us to get better accuracy. Depending on which activation function you use, it can provide lower or higher gradients. Relu allows the network the max amount of the error in back propagation.
                input_shape=input_shape))# Input shape represents the elements an array has in each dimension(ie. image_rows,image_cols,1)

# Add another convolution layer with parameters (filters,kernal_size and activation function)
model.add(Conv2D(64, (3, 3), activation='relu'))

# The MaxPooling2D layer is used for downsampling the image size. So in a (2,2) pool it splits a pixel image into 4 chucks and takes the 4 highest values from each chunk to represent or (2,2) pool
model.add(MaxPooling2D(pool_size=(2, 2)))

# A Dropout method consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
model.add(Dropout(0.25))

# Flatten method flattens the input provided
model.add(Flatten())

# Apply a dense layer with a output of 128 (nodes) and activation function relu yet again
model.add(Dense(128, activation='relu'))

# A Dropout method consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
model.add(Dropout(0.5))

# Apply a final dense layer with a output of num_class(10) which is our number of outcomes and use activation function softmax. We apply the softmax activation function to the final layer as it can be used to represent catagorical data. Outputing results ranging from 0 upto 1
model.add(Dense(num_classes, activation='softmax'))

# Compile method configures our model before training. Its takes arguments: Optimizer, loss and metrics
model.compile(loss=keras.losses.categorical_crossentropy,# Loss(objective) Function. Categorical_crossentrophy: catagorises our loss with our catagories
              optimizer=keras.optimizers.Adadelta(),# Adadelta Optimizer used to leave perametersat their default values. Better algorithm then the classic gradient descent
              metrics=['accuracy'])# Used to judge the performance of your model

# To train our model we use the fit function
model.fit(x_train, y_train,# inputs and outputs
          batch_size=batch_size,# Batch_size will give us samples per gradient update
          epochs=epochs, # Epochs is the number of iterations it will do over the data
          verbose=1, # Verbose provides us with a progress bar of the training when set to 1
          validation_data=(x_test, y_test)) # Evaluate the loss and any model metrics at the end of each epoch

# Calculate our loss and accuracy of our test data.
score = model.evaluate(x_test, y_test, verbose=0)# Input and output and verbose set to 0 means silent mode

# Print out Loss and accuracy of our test set.
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save Model in order to be reused again.
model.save("mnist_model.h5")