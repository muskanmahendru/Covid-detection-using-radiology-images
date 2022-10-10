
import cv2
import tensorflow as tf
import tensorflow.keras.models
import numpy as np

#tf.keras.Model()
tf.keras.models.Model() 

def prepare(imagePath):
    data = []
    
    test_image = cv2.imread(imagePath)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (224, 224))
    data.append(test_image)
    data = np.array(data) / 255.0
    

    return data

my_tester= prepare('tester2.jpg')
    
my_model= tf.keras.models.load_model("covid19.model")

prediction=my_model.predict(my_tester)

print(prediction)
