# save the final model to file
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
from matplotlib import pyplot as plt
import time
import tensorflow as tf
import numpy as np

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
 
# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
 
# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
 
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	#model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
# run the test harness for evaluating a model
def run_test_harness():
	#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()
	epochs=16

	time_callback = TimeHistory()
	# fit model
	history=model.fit(trainX, trainY, validation_split=0.25,epochs=epochs, batch_size=32, verbose=0,callbacks=[time_callback])
	
	#print(time_callback.times[0])	
	#print(len(time_callback.times))

	time= time_callback.times
	print(time)

	for i in range(1,len(time_callback.times)):
		time[i]=time_callback.times[i]+time_callback.times[i-1]
	#time=np.array(time)/60
	print(time)
	# save model
	#model.save('final_model.h5')

	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))

	# list all data in history
	print(history.history.keys())
	# summarize history for accuracy
	train_accuracy=np.array(history.history['accuracy'])*100
	val_accuracy=np.array(history.history['val_accuracy'])*100

	#print(accuracy)

	plt.plot(train_accuracy)
	plt.plot(val_accuracy)
	plt.title('model accuracy')
	plt.ylabel('accuracy (%)')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	#plt.show()
	plt.savefig('acc.png')
	plt.clf()

	plt.plot(train_accuracy)
	plt.title('model accuracy')
	plt.ylabel('accuracy (%)')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	#plt.show()
	plt.savefig('Tacc.png')
	plt.clf()
	
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	#plt.show()
	plt.savefig('loss.png')
	plt.clf()

	
	plt.plot(time,train_accuracy)
	plt.title('Train accuracy Vs Time')
	plt.ylabel('accuracy(%)')
	plt.xlabel('Training time (m)')
	plt.legend(['Train_single node'], loc='upper left')
	#plt.show()
	plt.savefig('time.png')

 
# entry point, run the test harness
run_test_harness()
