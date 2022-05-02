from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.datasets import cifar10
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = cifar10.load_data()#從網路讀取cifar10資料

x_train = x_train/255
x_test = x_test/255

y_train = np_utils.to_categorical(y_train)
y_test2 = y_test                                       #將原始標籤儲存 可以進行混淆矩陣檢測
y_test = np_utils.to_categorical(y_test)

#建立VGG16模型
model = Sequential([
    Conv2D(64, (3, 3), input_shape=(32,32,3), padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()
opt = Adam(lr=0.0001)

model.compile(loss='binary_crossentropy',optimizer=opt, metrics=['accuracy'])
#訓練
train_history = model.fit(x_train, y_train, batch_size=32, epochs=360, verbose=1, validation_data=(x_test, y_test))

#預測
prediction = model.predict(x_test) 
#畫圖
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
#預測分數
scores = model.evaluate(x_test, y_test)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0)) 
#做混淆矩陣
prediction = np.argmax(prediction,axis=1)
C2= confusion_matrix(y_test2.astype(str), prediction.astype(str))
print(C2)
