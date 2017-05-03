import os
import csv
import cv2
import numpy as np
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
STEER_COMP = 0.25
DELETE_ZERO = 0.95

lines = []
# read UDACITY training data from csv file
with open('data/data/driving_log.csv') as csvfile:
	next(csvfile)
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


train_set, validation_set = train_test_split(lines, test_size=0.05)


def LambdaResize(x):
	from keras.backend import tf as ktf
	return ktf.image.resize_images(x, (64,64), align_corners=True)



def generator(samples, batch_size=BATCH_SIZE, training=True):
	# remove 95% of the steering angle = 0 data during training phase
	if training == True:
		samples = np.array(samples)
		y = samples[:,3]
		zero_idx = np.where(y == ' 0')
		zero_idx = zero_idx[0]
		del_size = int(len(zero_idx) * DELETE_ZERO)
		sel_zero_idx = np.random.choice(zero_idx, size=del_size, replace=False)
		samples = np.delete(samples, sel_zero_idx, axis=0)
	else:
		samples = np.array(samples)

	num_samples = len(samples)
	batches_per_epoch = num_samples // batch_size

	i = 0

	while True:
		shuffle(samples)

		X_batch = np.zeros((batch_size, 160, 320, 3))
		y_batch = np.zeros((batch_size))
		start = i * batch_size
		end = start + batch_size - 1

		j = 0

		for entry in samples[start:end]:
			steering = 0
			column_idx = 0
			# randomly select from left, center, right camera. randomly decide to flip image
			camera = np.random.choice(['center', 'left', 'right'])
			flip = np.random.random()

			if camera == 'left':
				steering += STEER_COMP
				column_idx = 1
			elif camera == 'right':
				steering -= STEER_COMP
				column_idx = 2
			else:
				steering = 0
				column_idx = 0

			if training == False:
				steering = 0
				column_idx = 0

			source_path = entry[column_idx]
			source_path = source_path.strip()
			full_path = 'data/data/' + source_path
			# convert from BGR to RGB
			bgr_image = cv2.imread(full_path)
			rgb_image = bgr_image[...,::-1]
			measurement = float(entry[3]) + steering

			if flip > 0.5:
				rgb_image = cv2.flip(rgb_image, 1)
				measurement = -measurement

			X_batch[j] = np.array(rgb_image)
			y_batch[j] = np.array(measurement)

			j += 1

		i += 1

		if i == batches_per_epoch - 1:
			i = 0

		yield X_batch, y_batch


train_generator = generator(train_set, batch_size=BATCH_SIZE, training=True)
validation_generator = generator(validation_set, batch_size=BATCH_SIZE, training=False)

model = Sequential()
# pre-process data in model
# normalize
# crop out sky and hood of car
# resize to 64 x 64 image
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,26), (0,0))))
model.add(Lambda(LambdaResize))

model.add(Conv2D(8,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(8,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(16,(5,5),activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=200, validation_data=validation_generator, validation_steps=50, epochs=5)


model.save('model.h5')