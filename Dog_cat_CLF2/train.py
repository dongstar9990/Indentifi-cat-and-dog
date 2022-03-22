from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from warnings import filterwarnings
filterwarnings('ignore')

classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2)) #if stride not given it equal to pool filter size
classifier.add(Conv2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
adam = tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
classifier.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
from tensorflow.keras.models import load_model
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
classifier.summary()

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

#Training Set
trainingset = "dataset/train"
testingset = "dataset/test"
train_set = train_datagen.flow_from_directory(trainingset,
                                             target_size=(64,64),
                                             batch_size=32,
                                             class_mode='binary')
#Validation Set

test_set = test_datagen.flow_from_directory(testingset,
                                           target_size=(64,64),
                                           batch_size = 32,
                                           class_mode='binary',
                                           shuffle=False)
classifier.fit_generator(train_set,
                        steps_per_epoch=32,
                        epochs = 50,
                        validation_data = test_set,
                        validation_steps = 20,
                        #callbacks=[tensorboard]
                        );# test_set1 = test_datagen.flow_from_directory('test1',
classifier.save('model1.h5')

classifier = load_model('model1.h5')

#Prediction of image
# %matplotlib inline
# import tensorflow
# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt
# import numpy as np
# img1 = image.load_img('dataset/test/cat/cat.1000.jpg', target_size=(64, 64))
# img = image.img_to_array(img1)
# img = img/255
# # create a batch of size 1 [N,H,W,C]
# img = np.expand_dims(img, axis=0)
# prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
# if(prediction[:,:]>0.5):
#     value ='Dog :%1.2f'%(prediction[0,0])
#     plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
# else:
#     value ='Cat :%1.2f'%(1.0-prediction[0,0])
#     plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
#
# plt.imshow(img1)
# plt.show()
