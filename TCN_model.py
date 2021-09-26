from keras.models import Input, Model
from tcn import TCN
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
import keras.backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '0' #2
np.random.seed(1337)  # for reproducibility

# X = np.load('atrain_stft.npy', allow_pickle=True)
# Y = np.load('atrain_label.npy', allow_pickle=True)

X = np.load('../data/springer/springer_stft41.npy', allow_pickle=True)
Y = np.load('../data/springer/springer_label.npy', allow_pickle=True)

num = len(Y)
lens_Y = [0]*num
for i in range(num):
    lens_Y[i] = len(Y[i])

X = np.array(X)
maxlen_0 = 1826

X = keras.preprocessing.sequence.pad_sequences(X, maxlen=maxlen_0, value=0,
                                               padding='pre', truncating='pre')
Y = keras.preprocessing.sequence.pad_sequences(Y, maxlen=maxlen_0, value=4,
                                               padding='pre', truncating='pre')
Y = Y - 1
Y = keras.utils.to_categorical(Y, num_classes=4)

row_rand = np.arange(X.shape[0])
np.random.shuffle(row_rand)

XTrain = X[row_rand[0:350]]
YTrain = Y[row_rand[0:350]]
XValid = X[row_rand[350:500]]
YValid = Y[row_rand[350:500]]
XTest = X[row_rand[500:792]]
YTest = Y[row_rand[500:792]]

lens_Y = np.array(lens_Y)
lens_YTest = lens_Y[row_rand[500:792]]

batch_size, timesteps, input_dim = None, maxlen_0, 41
i = Input(batch_shape=(batch_size, timesteps, input_dim))
dilations_num = [1, 2, 4, 8, 16, 32]
o = TCN(nb_filters=60, nb_stacks=1, kernel_size=2, dilations=dilations_num,
        return_sequences=True, padding='causal',
        activation='relu', dropout_rate=0.1)(i)

o = keras.layers.TimeDistributed(keras.layers.Dense(10, activation='relu'))(o)
o = keras.layers.TimeDistributed(keras.layers.Dense(4, activation='softmax'))(o)
m = Model(inputs=[i], outputs=[o])
m.summary()

m.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

filepath = 'model_TCN_springer_stft41_{epoch:02d}-{val_loss:.2f}-{val_accuracy:.4f}.hdf5'
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_tcn_springer.h5'
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='auto')


class best_model(keras.callbacks.Callback):

    def __init__(self):
        super(best_model, self).__init__()
        self.model_best = m
        self.acc = 0
        self.times = []

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_accuracy')
        if current > self.acc:
            print('\nNew score!\n')
            self.acc = current
            self.model_best = m

    def on_train_end(self, logs=None):
        preds = self.model_best.predict(XTest)
        right_num = np.zeros([1, 292])

        for i in range(292):
            temp1 = YTest[i]
            temp2 = preds[i]
            Ind_temp1 = np.argmax(temp1, axis=1)
            Ind_temp2 = np.argmax(temp2, axis=1)
            A = np.reshape(Ind_temp1, [1, maxlen_0])
            Tre = A[0, (maxlen_0 - int(lens_YTest[i])):maxlen_0]
            B = np.reshape(Ind_temp2, [1, maxlen_0])
            Prd = B[0, (maxlen_0 - int(lens_YTest[i])):maxlen_0]
            result = [Tre == Prd]
            right_num[0, i] = np.array(result).sum()

        all_acc = right_num.sum() / lens_YTest.sum()
        print([self.acc, all_acc])


def get_flops(model):
    run_meta = tf.RunMetadata()
    opt1 = tf.profiler.ProfileOptionBuilder.float_operation()
    opt2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opt1)
    params_ = tf.profiler.profile(graph=K.get_session().graph,
                                  run_meta=run_meta, cmd='op', options=opt2)

    return flops.total_float_ops, params_.total_parameters


callbacks_list = [checkpoint, best_model()]

flops, params = get_flops(m)
print(flops)
print(params)

m.fit(XTrain, YTrain, epochs=200, batch_size=50,
      validation_data=[XValid, YValid], callbacks=callbacks_list)


