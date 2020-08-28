from tensorflow import keras
import config
import model
from datagen import data_generator
from keras.callbacks import ModelCheckpoint, EarlyStopping


# es = EarlyStopping(monitor='val_loss')
data_gen = data_generator()
train_generator = data_gen[0]
valid_generator = data_gen[1]
mod = model.EmotionModel()
model_final = mod.fin_model()

def train(model=model_final):
    checkpoint = ModelCheckpoint(config.MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    history = model.fit(train_generator,
                        steps_per_epoch=train_generator.n//train_generator.batch_size,
                        validation_steps=valid_generator.n//valid_generator.batch_size,
                        epochs=config.EPOCHS,
                        verbose=1,
                        validation_data = valid_generator,
                        callbacks=[checkpoint])

    return history

if __name__ == '__main__':
    train()