from keras.layers import Dense, Flatten,Dropout
from keras.utils import np_utils # 用來後續將 label 標籤轉為 one-hot-encoding
from sklearn.metrics import confusion_matrix
from keras import applications
from keras.models import Sequential, Model
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import os
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
def load_data():
    path = './Data/'
    files = os.listdir(path)
    images = []
    labels = []
    for x in files:
        path = './Data/'
        path  = path + x
        files = os.listdir(path)
        for f in files:
            img_path = path 
            #print(img_path)
            #print(img_path+'/'+str(f))
            img = image.load_img(img_path+'/'+str(f), grayscale=False, target_size=(224, 224))
            img_array = image.img_to_array(img)
            images.append(img_array)
    
            lb = x
            labels.append(lb);
    data = np.array(images)
    labels = np.array(labels)

    return data, labels
images, label = load_data()
images /= 255

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
encoder_labels = labelencoder.fit_transform(label) #進行Labelencoding編碼
#print(encoder_labels)

labels = np_utils.to_categorical(encoder_labels) # 用來將 label 標籤轉為 one-hot-encoding

(x_train, x_test, y_train, y_test) = train_test_split(images, labels, test_size=0.2)

y_test2 = np.argmax(y_test, axis = 1) #將資料轉為原始資料

# 定義模型
base_model = applications.VGG16(weights="imagenet", include_top=False,
                                input_shape=(224, 224, 3))  # 預訓練的VGG16網絡，替換掉頂部網絡
print(base_model.summary())
for layer in base_model.layers[:15]: layer.trainable = False  # 凍結預訓練網絡前15層
top_model = Sequential()  # 自定義頂層網絡
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))  # 將預訓練網絡展平
top_model.add(Dense(256, activation='relu'))  # 全連接層，輸入像素256
top_model.add(Dropout(0.5))  # Dropout概率0.5
top_model.add(Dense(3, activation='sigmoid'))  # 輸出層，
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))  # 新網絡=預訓練網絡+自定義網絡
opt = Adam(lr=0.0001)
model.compile(optimizer=opt,loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# 訓練&評估

train_history = model.fit(x_train, y_train, batch_size=10, epochs=20, verbose=1, validation_data=(x_test, y_test))

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
prediction = np.argmax(prediction,axis=1)

C2= confusion_matrix(y_test2.astype(str), prediction.astype(str))
