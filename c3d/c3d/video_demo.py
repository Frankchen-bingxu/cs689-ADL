# coding=utf8
#from c3d.models import c3d_model
from keras.optimizers import SGD
import numpy as np
import cv2
import json
import argparse
from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation
from keras.regularizers import l2
from keras.models import Model



input_shape = (112,112,16,3)
weight_decay = 0.005
nb_classes = 101

inputs = Input(input_shape)
x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
           activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)

x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
           activation='relu',kernel_regularizer=l2(weight_decay))(x)
x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
           activation='relu',kernel_regularizer=l2(weight_decay))(x)
x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
           activation='relu',kernel_regularizer=l2(weight_decay))(x)
x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
           activation='relu',kernel_regularizer=l2(weight_decay))(x)
x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

x = Flatten()(x)
x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
x = Dropout(0.5)(x)
x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
x = Dropout(0.5)(x)
x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
x = Activation('softmax')(x)

model = Model(inputs, x)

with open('/Users/chenbingxu/PycharmProjects/c3d/ucfTrain/classInd.txt', 'r') as f:
    class_names = f.readlines()
    f.close()

# init model
#model = c3d_model()
lr = 0.005
sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()
model.load_weights('/Users/chenbingxu/PycharmProjects/c3d/c3d/results/weights_c3d_101.h5', by_name=True)
# read video
video = '/Users/chenbingxu/PycharmProjects/c3d/video_test/8_actions.mp4'
cap = cv2.VideoCapture(video)#read the frame of the video

clip = []
submit = './extra_test_689.json'
result = []
while True:
    ret, frame = cap.read()#ret = Ture/Faslse frame = frame
    if ret:
        tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        clip.append(cv2.resize(tmp, (171, 128)))
        if len(clip) == 16:


            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            #print(time)
            temp_dict = {}


            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs[..., 0] -= 99.9
            inputs[..., 1] -= 92.1
            inputs[..., 2] -= 82.6
            inputs[..., 0] /= 65.8
            inputs[..., 1] /= 62.3
            inputs[..., 2] /= 60.3
            inputs = inputs[:,:,8:120,30:142,:]
            inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
            pred = model.predict(inputs)

            label = np.argmax(pred[0])

            temp_dict["label"] = class_names[label].split(' ')[-1].strip()
            temp_dict["time"] = time
            temp_dict["pro"] = pred[0][label]
            result.append(temp_dict)

            #print(class_names[label].split(' ')[-1].strip())
            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % pred[0][label], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            clip.pop(0)




        cv2.imshow('result', frame)
        cv2.waitKey(10)

    else:
        break

# with open(submit, 'a') as f:
#     json.dump(str(result), f)
cap.release()
cv2.destroyAllWindows()


