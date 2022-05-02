import numpy as np
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix

def load_data():
    path = './DigitDataset/'
    files = os.listdir(path)
    images = []
    labels = []
    for x in files:
        path = './DigitDataset/'
        path  = path + x
        files = os.listdir(path)
        for f in files:
            img_path = path 
            print(img_path)
            print(img_path+'/'+str(f))
            img = image.load_img(img_path+'/'+str(f), grayscale=True, target_size=(28, 28))
            img_array = image.img_to_array(img)
            images.append(img_array)
    
            lb = x
            labels.append(lb);
    data = np.array(images)
    labels = np.array(labels)

    return data, labels


print("Loading data...")
images, lables = load_data()
images /= 255
(x_train, x_test, y_train, y_test) = train_test_split(images, lables, test_size=0.2)


y_train = np_utils.to_categorical(y_train)
y_test2 = y_test
y_test = np_utils.to_categorical(y_test)

print(x_test.shape)
print(y_test.shape)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
print(x_test.shape)
print(y_test.shape)
train_history = model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(x_test, y_test))

def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()  
    
show_train_history(train_history, 'loss', 'val_loss')  
show_train_history(train_history, 'accuracy', 'val_accuracy')  

scores = model.evaluate(x_test, y_test)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0)) 

 
prediction = model.predict_classes(x_test) 

C2= confusion_matrix(y_test2.astype(str), prediction.astype(str))
