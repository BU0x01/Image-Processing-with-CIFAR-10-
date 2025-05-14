import tensorflow as tf 
 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, 
BatchNormalization, Activation 
 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
 
import numpy as np 
import random 
 
from PIL import Image 
 
import os 

# Load data from TF Keras 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() 
 
# CIFAR10 class names 
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 
'Horse', 'Ship', 'Truck'] 
num_classes = len(class_names) 

path_images = "./Data/images/" 
 
# Create directory 
if not os.path.exists(path_images): 
    os.mkdir(path_images) 
 
# Save one image per class  
ext=".jpg" 
for image_index in range(0,100): 
    im = Image.fromarray(x_test[image_index]) 
    im.save("./images/"+str(class_names[int(y_test[image_index])])+ext)

# Show saved images 
files = os.listdir(path_images)   
for img in files: 
    if os.path.splitext(img)[1] == ext and os.path.splitext(img)[0] in class_names: 
        #print(os.path.splitext(img)[0]) 
        plt.subplot(2,5,class_names.index(os.path.splitext(img)[0])+1) 
        plt.xticks([]) 
        plt.yticks([]) 
        plt.grid(False) 
        plt.imshow(mpimg.imread(path_images+img),) 
        plt.xlabel(os.path.splitext(img)[0]) 
plt.show() 

# Normalize pixel values to be between 0 and 1 
x_train = x_train.astype(np.float32)/255 
x_test = x_test.astype(np.float32)/255 
 
# Convert class vectors to binary class matrices. 
y_train = tf.keras.utils.to_categorical(y_train, num_classes) 
y_test = tf.keras.utils.to_categorical(y_test, num_classes) 
 
# Print arrays shape 
print('x_train shape:', x_train.shape) 
print('y_train shape:', y_train.shape) 
print('x_test shape:', x_test.shape) 
print('y_test shape:', y_test.shape) 

# Hyperparameters 
batch_size = 32 
num_classes = len(class_names) 
epochs = 1 
img_rows, img_cols = x_train.shape[1], x_train.shape[2] 
input_shape = (x_train.shape[1], x_train.shape[2], 1) 

# Creating a Sequential Model and adding the layers 
model = Sequential() 
model.add(Conv2D(16, (3, 3), padding='same', input_shape=(32,32,3))) 
model.add(BatchNormalization()) 
model.add(Activation('relu')) 
model.add(Conv2D(16, (3, 3),padding='same')) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2))) 
model.add(Dropout(0.2)) 
 
model.add(Conv2D(32, (3, 3), padding='same')) 
model.add(BatchNormalization()) 
model.add(Activation('relu')) 
model.add(Conv2D(32, (3, 3),padding='same')) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2))) 
model.add(Dropout(0.3)) 
 
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization()) 
model.add(Activation('relu')) 
model.add(Conv2D(64, (3, 3),padding='same')) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2))) 
model.add(Dropout(0.4)) 
 
model.add(Flatten()) 
model.add(Dense(32)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(10)) #The number of classes we have 
model.add(Activation('softmax')) 

# Check model structure and the number of parameters 
model.summary() 
 
# Let's train the model using Adam optimizer 
model.compile(loss='categorical_crossentropy', optimizer='adam', 
metrics=['accuracy']) 

# Train model 
history = model.fit(x=x_train, 
          y=y_train, 
          batch_size=batch_size, 
          epochs=epochs,  
          validation_data=(x_test, y_test)) 

# Save keras model 
path_models = "./Data/models/" 
path_keras_model = path_models + "own_cifar10_model.h5" 
 
# Create directory 
if not os.path.exists(path_models): 
    os.mkdir(path_models) 
 
model.save(path_keras_model) 
 
# Score trained model. 
scores = model.evaluate(x_test, y_test, verbose=1) 
print('Test loss:', scores[0]) 
print('Test accuracy:', scores[1]) 

path_csv = "./Data/" 
path_csv_file = path_csv+"own_cifar10_validation_20image.csv" 
 
# Create directory 
if not os.path.exists(path_csv): 
    os.mkdir(path_csv) 
 
# Remove old csv file 
if os.path.exists(path_csv_file): 
    os.remove(path_csv_file) 
 
# Load data from TF Keras 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() 
 
 
# CIFAR10 class names 
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 
'Horse', 'Ship', 'Truck'] 
 
# Normalize pixel values to be between 0 and 1 
x_train = x_train.astype(np.float32)/255 
x_test = x_test.astype(np.float32)/255 
 
# Print arrays shape 
print('x_train shape:', x_train.shape) 
print('y_train shape:', y_train.shape) 
print('x_test shape:', x_test.shape) 
print('y_test shape:', y_test.shape) 
     
# Save csv file that contain pixel's value 
num_sample = 50 
rx = random.sample(range(0,len(x_test)),num_sample) 
 
for i in range(0,num_sample): 
    data = x_test[rx[i]] 
    #print(data.shape) 
    data = data.flatten() 
    output = y_test[rx[i]] 
    data=np.append(data,output) 
    data = np.reshape(data, (1,data.shape[0])) 
    #print(data.shape) 
    with open(path_csv_file, 'ab') as f: 
        np.savetxt(f, data, delimiter=",")

path_labels = "./Data/labels/” 
path_labels_file = path_labels+"own_cifar10_labels.txt" 
 
# Create directory 
if not os.path.exists(path_labels): 
    os.mkdir(path_labels) 
     
# Remove old label file 
if os.path.exists(path_labels_file): 
    os.remove(path_labels_file) 
 
# Create label file 
for i in range(0,len(class_names)): 
    with open(path_labels_file, 'a') as f: 
        f.write(str(i)+","+class_names[i]+"\n") 

