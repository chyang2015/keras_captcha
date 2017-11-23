from keras.utils.visualize_util import plot
from captcha_model import captcha_model
from captcha_generate_image import gen,decode
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

model = captcha_model()
plot(model,to_file='model.png',show_shapes=True)
checkpointer =ModelCheckpoint(filepath="net-weight\\net-epoch.hdf5", verbose=1, save_best_only=True)
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit_generator(gen(),samples_per_epoch=51200,nb_epoch=5,validation_data=gen(),nb_val_samples=1280,callbacks=[checkpointer])

