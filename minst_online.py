from keras.datasets import mnist
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding
from matplotlib import pyplot as plt
from keras.models import Sequential 
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.metrics import confusion_matrix

(x_train, y_train), (x_test, y_test) = mnist.load_data() #從網路讀取minst資料

x_train = x_train.reshape(60000,28,28,1)/255
x_test = x_test.reshape(10000,28,28,1)/255

y_train = np_utils.to_categorical(y_train) #將 label 標籤轉為 one-hot-encoding
y_test2 = y_test			      #將原始標籤儲存 可以進行混淆矩陣檢測
y_test = np_utils.to_categorical(y_test) 

#建立模型
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

#進行訓練
train_history = model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(x_test, y_test))

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


scores = model.evaluate(x_test, y_test)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0)) 

#預測
prediction = model.predict_classes(x_test) 

#混淆矩陣檢測
C2= confusion_matrix(y_test2.astype(str), prediction.astype(str))
