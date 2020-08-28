from tensorflow import keras
import config
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras import regularizers

class EmotionModel():
    def __init__(self):
        self.num_classes = config.NUM_CLASSES

    def conv_layer(self, model, num_filters, kernel_size, pad='same'):
        model.add(Conv2D(num_filters, kernel_size, padding=pad))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        return model 

    def fc_layer(self, model, num_filters):
        model.add(Dense(num_filters, bias_regularizer=regularizers.l2(1e-4)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))  
        return model

    def fin_model(self):
        model = Sequential()
        model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model = self.conv_layer(model, 128, (3,3))
        model = self.conv_layer(model, 256, (5,5))
        model = self.conv_layer(model, 256, (3,3))
        # model = conv_layer(model, 512, (3,3))
        # model = conv_layer(model, 512, (3,3))

        model.add(Flatten())

        model = self.fc_layer(model, 1024)
        # model = fc_layer(model, 512)
        model = self.fc_layer(model, 256)

        model.add(Dense(self.num_classes, activation= 'softmax'))

        model.compile(optimizer= Adam() , loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())
        return model

if __name__ == "__main__":
    mod = EmotionModel()
    mod.fin_model()