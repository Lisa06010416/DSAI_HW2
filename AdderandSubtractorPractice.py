from keras.models import Model
from keras.layers import *
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import RMSprop
import random
# onehot 編碼
onehot_label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", " "]
encoder = LabelEncoder()
encoder.fit(onehot_label)


# 把list轉成onehot
def get_onehot(dataset):
    dataset_onehot = []
    for data in dataset:
        data_onehot = encoder.transform(list(data))
        data_onehot = np_utils.to_categorical(data_onehot, num_classes=len(onehot_label))
        dataset_onehot.append(data_onehot)
    return dataset_onehot


# 把onehot轉回原本的label
def decoder(list):
    label_index = np.argmax(list, axis=1)
    label = encoder.inverse_transform(label_index)
    return label


def generate_trainingData():
    max = 1000
    train = []
    target = []
    for A in range(max):
        for B in range(max):
            str_add = str(A) + "+" + str(B)
            str_add = str_add.ljust(7)
            train.append(str_add)
            target.append(str(A + B).ljust(4))

            str_sub = str(A) + "-" + str(B)
            str_sub = str_sub.ljust(7)
            train.append(str_sub)
            target.append(str(A - B).ljust(4))
    np.save("dataset/train.npy", train)
    np.save("dataset/target.npy", target)
    with open("dataset/train", "w") as f:
        for data in train:
            f.write(data + "\n")
    with open("dataset/target", "w") as f:
        for ans in target:
            f.write(ans + "\n")


def pre():
    generate_trainingData()
    print(get_onehot(onehot_label))
    train = np.load("dataset/train.npy")
    target = np.load("dataset/target.npy")

    trainData = get_onehot(train)
    trainTarget = get_onehot(target)

    np.save("dataset/trainData.npy", trainData)
    np.save("dataset/trainTarget.npy", trainTarget)

def generalmin():
    trainData = np.load("dataset/trainData.npy")
    trainTarget = np.load("dataset/trainTarget.npy")

    # z = list(zip(trainData, trainTarget))
    # random.shuffle(z)
    # trainData[:], trainTarget[:] = zip(*z)
    # print(len(trainData))
    # trainData = trainData[0:250000]
    # trainTarget = trainTarget[0:250000]
    # print(len(trainData))
    # testData = trainData[150000:250000]
    # testTarget = trainTarget[150000:250000]
    # print(len(trainData[150000:250000]))
    np.save("dataset/trainData_min.npy", trainData[0:250000])
    np.save("dataset/trainTarget_min.npy", trainTarget[0:250000])
    #
    # np.save("dataset/testData_.npy", testData)
    # np.save("dataset/testTarget_.npy", testTarget)

# 資料前處理
# generate_trainingData()
# pre()
generalmin()
# 讀檔
trainData = np.load("dataset/trainData_min.npy")
trainTarget = np.load("dataset/trainTarget_min.npy")
testData = trainData[150000:]
testTarget = trainTarget[150000:]
trainData = trainData[0:150000]
trainTarget = trainTarget[0:150000]



# 拿80000train data
trainData = trainData[0:80000]
trainTarget = trainTarget[0:80000]

# 拿60000test data
testData = trainData[0:60000]
testTarget = testTarget[0:60000]


# model
rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)

input = Input(shape=(7, 13))
x = Flatten()(input)

x = Dense(64, activation='relu')(x)

x = Dense(52, activation='relu')(x)  # 4x13 = 52
x = Reshape((4, 13))(x)
x = Activation('softmax')(x)
model = Model(inputs=input, outputs=x)

# model.compile(optimizer='sgd', loss='mse')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(trainData, trainTarget,
          epochs=100,
          batch_size=800000,
          shuffle=True,
          validation_data=(testData, testTarget),
          verbose = 1,
          )

model.save('my_model.h5')

# # decoder(data_onehot)
