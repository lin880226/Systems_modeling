from keras.layers import Dense, Flatten,Conv2D,MaxPooling2D,Dropout
from keras.datasets import cifar10,mnist
from keras.utils import np_utils # 用來後續將 label 標籤轉為 one-hot-encoding
from sklearn.metrics import confusion_matrix
from keras import optimizers,applications
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = [cv2.resize(i,(48,48))for i in x_train]
  

x_train = np.concatenate([arr[np.newaxis]for arr in x_train]).astype('float32')
x_train /=255

x_test = [cv2.resize(i,(48,48))for i in x_test]
x_test = np.concatenate([arr[np.newaxis]for arr in x_test]).astype('float32')
x_test /=255

y_test2 = y_test
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# 數據集
#img_height, img_width = 48, 48  # 圖片高寬
#batch_size = 2  # 批量大小
#epochs = 50  # 迭代次數

# 定義模型
base_model = applications.VGG16(weights="imagenet", include_top=False,
                                input_shape=(48, 48, 3))  # 預訓練的VGG16網絡，替換掉頂部網絡
print(base_model.summary())

for layer in base_model.layers[:15]: layer.trainable = False  # 凍結預訓練網絡前15層

top_model = Sequential()  # 自定義頂層網絡
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))  # 將預訓練網絡展平
top_model.add(Dense(256, activation='relu'))  # 全連接層，輸入像素256
top_model.add(Dropout(0.5))  # Dropout概率0.5
top_model.add(Dense(10, activation='sigmoid'))  # 輸出層，

# top_model.load_weights("") # 單獨訓練的自定義網絡

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))  # 新網絡=預訓練網絡+自定義網絡

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
"""
train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)  # 訓練數據預處理器，隨機水平翻轉
test_datagen = ImageDataGenerator(rescale=1. / 255)  # 測試數據預處理器
train_generator = train_datagen.flow_from_directory(x_train, target_size=(48, 48),
                                                    batch_size=2, class_mode='binary')  # 訓練數據生成器
validation_generator = test_datagen.flow_from_directory(y_train, target_size=(img_height, img_width),
                                                        batch_size=batch_size, class_mode='binary',
                                                        shuffle=False)  # 驗證數據生成器
checkpointer = ModelCheckpoint(filepath='dogcatmodel.h5', verbose=1, save_best_only=True)  # 保存最優模型
"""
# 訓練&評估

train_history = model.fit(x_train, y_train, batch_size=64, epochs=2, verbose=1, validation_data=(x_test, y_test))

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

prediction = model.predict(x_test)
C2= confusion_matrix(y_test2.astype(str), prediction.astype(str))
