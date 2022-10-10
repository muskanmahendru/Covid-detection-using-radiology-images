from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
import cv2
import os
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16


arg = argparse.ArgumentParser()
arg.add_argument("-d", "--coviddataset", required=True,
	help="provide the required path of dataset")
arg.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to graph")
arg.add_argument("-m", "--model", type=str, default="covid19.model",
	help="path for model")
args = vars(arg.parse_args())


image_path = list(paths.list_images(args["coviddataset"]))
data = []
new_labels = []


for i in image_path:
	# extract the class label 
	label = i.split(os.path.sep)[-2]

	image = cv2.imread(i)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# update the data and labels lists, respectively
	data.append(image)
	new_labels.append(label)

# convert the data and labels to NumPy arrays 
data = np.array(data)/255.0
new_labels = np.array(new_labels)

# perform one-hot encoding on the labels
labelb = LabelBinarizer()
new_labels = labelb.fit_transform(new_labels)
new_labels = to_categorical(new_labels)

# partition the data into training and testing splits 
(trainX, testX, trainY, testY) = train_test_split(data, new_labels,
	test_size=0.20, stratify=new_labels, random_state=42)

# initialize the training data augmentation object
train_data_aug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")

# load the VGG16 network, 
lower_model = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the top of the model that will be placed above the base model
top_model = lower_model.output
top_model = AveragePooling2D(pool_size=(4, 4))(top_model)
top_model = Flatten(name="flatten")(top_model)
top_model = Dense(64, activation="relu")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

# place the top_model above the base model 
model = Model(inputs=lower_model.input, outputs=top_model)


for layer in lower_model.layers:
	layer.trainable = False


LEARNING_RATE = 1e-3
EPOCHS = 50
BS = 25
# compile our model
opt = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
H = model.fit_generator(
	train_data_aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

#predictions
predIdxs = model.predict(testX, batch_size=BS)


predIdxs = np.argmax(predIdxs, axis=1)


print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=labelb.classes_))

# compute the confusion matrix 
conf_mat = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(conf_mat))
accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / total
sensitivity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
specificity = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])


print(conf_mat)
print("accuracy: {:.4f}".format(accuracy))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="training_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="value_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="training_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="value_acc")
plt.title("Plot data on covid-19")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the model to disk
model.save(args["model"], save_format="h5")