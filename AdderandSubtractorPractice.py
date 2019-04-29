from keras.models import Model
from keras.layers import *
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# onehot 編碼
onehot_label = ["0","1","2","3","4","5","6","7","8","9","+","-"," "]
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
            str_add = str(A)+"+"+str(B)
            str_add = str_add.ljust(7)
            train.append(str_add)
            target.append(str(A+B).ljust(4))

            str_sub = str(A) + "-" + str(B)
            str_sub = str_sub.ljust(7)
            train.append(str_sub)
            target.append(str(A - B).ljust(4))
    np.save("dataset/train.npy",train)
    np.save("dataset/target.npy",target)
    with open("dataset/train","w") as f:
        for data in train:
            f.write(data+"\n")
    with open("dataset/target","w") as f:
        for ans in target:
            f.write(ans+"\n")

def pre():
    generate_trainingData()
    print(get_onehot(onehot_label))
    train = np.load("train.npy")
    target = np.load("target.npy")


    trainData = get_onehot(train)
    trainTarget = get_onehot(target)

    np.save("dataset/trainData.npy",trainData)
    np.save("dataset/trainTarget.npy",trainTarget)

print(list(get_onehot(onehot_label)))


trainData = np.load("trainData.npy")
trainTarget = np.load("trainTarget.npy")

input = Input(shape=(7,13))

x=Dense(256, activation='relu')(input)
x=Dense(64, activation='relu')(x)
x=Dense(32, activation='relu')(x)
x=Dense(16, activation='relu')(x)
model=Dense(52, activation='relu')(x)  # 4x13 = 52

model = Model(inputs=input, outputs=model)
model.compile(optimizer='adam', loss='mse')

print(model.summary())
model.fit(trainData, trainTarget,
                epochs=1000,
                batch_size=20,
                shuffle=True,
                )


model.save('my_model.h5')



# decoder(data_onehot)