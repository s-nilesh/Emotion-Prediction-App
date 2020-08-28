from tensorflow import keras
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
import config

def data_generator():
    datagen_train = ImageDataGenerator(rescale=1./255,
                                  rotation_range=30,
                                  shear_range=0.3,
                                  horizontal_flip=True)

    datagen_valid = ImageDataGenerator(rescale=1./255)  
    
    train_generator = datagen_train.flow_from_directory(config.TRAIN_PATH,
                                                   target_size = (48,48),
                                                   batch_size = config.BATCH_SIZE,
                                                   shuffle = True,
                                                   color_mode = 'grayscale',
                                                   class_mode = 'categorical')

    valid_generator = datagen_valid.flow_from_directory(config.VALID_PATH,
                                                   target_size = (48,48),
                                                   batch_size = config.BATCH_SIZE,
                                                   shuffle = False,
                                                   color_mode = 'grayscale',
                                                   class_mode = 'categorical')

    print('***************************',train_generator.classes)
    return train_generator, valid_generator

if __name__=="__main__":
    data_generator()
