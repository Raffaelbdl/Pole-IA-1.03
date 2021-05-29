import utils.io as io
import utils.merging as merging
import utils.tools as tools
import utils.scoring as scoring
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
kl = tf.keras.layers


class CNN_dense():

    def __init__(self, input_shape, name, model: tf.keras.models.Model = None, dense_neurons=128, dropout=False, **kwargs):
        self.input_shape = input_shape
        self.name = name
        self.model = model
        self.dense_neurons = dense_neurons
        self.dropout = dropout

    def build_model(self):

        input_layer = kl.Input(self.input_shape)

        x = kl.Conv1D(filters=16, kernel_size=3, strides=1,
                      padding='valid', activation='relu')(input_layer)
        x = kl.MaxPool1D(pool_size=2, strides=2, padding='same')(x)

        y = kl.Conv1D(filters=32, kernel_size=3, strides=1,
                      padding='valid', activation='relu')(x)
        y = kl.MaxPool1D(pool_size=2, strides=2, padding='same')(y)

        z = kl.Flatten()(y)
        z = kl.Dense(dense_neurons, activation='sigmoid')(z)
        if self.dropout:
            z = kl.Dropout(0.4)(z)

        a = kl.Dense(6, activation='softmax')(z)

        model = tf.keras.models.Model(
            inputs=input_layer, outputs=a, name="preprocess")
        model.summary()
        self.model = model

    def save_model(self):
        tf.keras.models.save_model(
            self.model, './saved_models/' + self.name, save_format='h5')

    def load_model(self):
        self.model = tf.keras.models.load_model(
            './saved_models/' + self.name)
        self.model.summary()

    def build_first_conv(self):

        input_layer = kl.Input((None, 9))
        x = kl.Conv1D(filters=16, kernel_size=3, strides=1,
                      padding='valid', activation='relu')(input_layer)
        x = kl.MaxPool1D(pool_size=2, strides=2, padding='same')(x)
        model = tf.keras.models.Model(inputs=input_layer, outputs=x)
        model.set_weights(self.model.get_weights()[:2])
        return model

    def build_second_conv(self):

        input_layer = kl.Input((None, 9))
        x = kl.Conv1D(filters=16, kernel_size=3, strides=1,
                      padding='valid', activation='relu')(input_layer)
        x = kl.MaxPool1D(pool_size=2, strides=2, padding='same')(x)
        y = kl.Conv1D(filters=32, kernel_size=3, strides=1,
                      padding='valid', activation='relu')(x)
        y = kl.MaxPool1D(pool_size=2, strides=2, padding='same')(y)
        model = tf.keras.models.Model(inputs=input_layer, outputs=y)
        model.set_weights(self.model.get_weights()[:4])
        return model

    def build_dense(self):

        input_layer = kl.Input((128, 9))
        x = kl.Conv1D(filters=16, kernel_size=3, strides=1,
                      padding='valid', activation='relu')(input_layer)
        x = kl.MaxPool1D(pool_size=2, strides=2, padding='same')(x)
        y = kl.Conv1D(filters=32, kernel_size=3, strides=1,
                      padding='valid', activation='relu')(x)
        y = kl.MaxPool1D(pool_size=2, strides=2, padding='same')(y)
        z = kl.Flatten()(y)
        z = kl.Dense(dense_neurons, activation='sigmoid')(z)

        model = tf.keras.models.Model(inputs=input_layer, outputs=z)
        model.set_weights(self.model.get_weights()[:6])
        return model


if __name__ == "__main__":

    LABELS = {
        1: 'WALKING',
        2: 'WALKING_UPSTAIRS',
        3: 'WALKING_DOWNSTAIRS',
        4: 'SITTING',
        5: 'STANDING',
        6: 'LAYING'
    }

    ### TRAINING CNN ###

    # Loading Dataset #
    trainX, trainy, testX, testy = io.load_dataset()
    train_labels = io.readFile('./original_data/train/y_train.txt')
    test_labels = io.readFile('./original_data/test/y_test.txt')
    labels = np.concatenate((train_labels, test_labels))
    labels = np.reshape(labels, (labels.shape[0], ))

    # Building or loading model #
    model_name = "model_dense_dropout_128"
    dense_neurons = 128
    epochs = 10
    dropout = True
    Model = CNN_dense(
        input_shape=trainX.shape[1:], name=model_name, dense_neurons=dense_neurons, dropout=dropout)
    try:
        Model.load_model()
    except:
        Model.build_model()

    # Training Model #
    training = True
    if training:
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        Model.model.compile(loss='categorical_crossentropy',
                            optimizer=opt, metrics='accuracy')
        Model.model.fit(trainX, trainy,
                        batch_size=16, epochs=epochs, verbose=1)
        (loss, accuracy) = Model.model.evaluate(testX, testy)
        Model.save_model()

    # Creating submodels #
    model_first_conv = Model.build_first_conv()
    model_first_conv.summary()

    model_second_conv = Model.build_second_conv()
    model_second_conv.summary()

    model_dense = Model.build_dense()
    model_dense.summary()

    ### COCLUSTERING ###

    # Loading Data #
    train_subjects = io.readFile(
        './original_data/train/subject_train.txt')
    test_subjects = io.readFile(
        './original_data/test/subject_test.txt')
    subjects = np.concatenate((train_subjects, test_subjects))
    X = np.concatenate((trainX, testX))
    # X = testX
    labels = np.concatenate((train_labels, test_labels))
    labels = np.reshape(labels, (labels.shape[0], ))

    # Vizualizing coclustering for different layers #
    first_output = model_first_conv(X).numpy()
    first_output = first_output.reshape(
        first_output.shape[0], first_output.shape[1] * first_output.shape[2]) + 1
    second_output = model_second_conv(X).numpy()
    second_output = second_output.reshape(
        second_output.shape[0], second_output.shape[1] * second_output.shape[2]) + 1
    third_output = model_dense(X).numpy() + 1

    # rows, cols = 26, 3
    # rows, cols = 6, 2
    # first_cluster, first_array = tools.biclustering_impact_viewer_bis(
    #     first_output, rows, cols, "First output")
    # scores1 = scoring.evaluate_clustering(labels, first_cluster)
    # scores1 = scoring.better_score(scores1, LABELS)
    # i = 1
    # for score in scores1:
    #     print(f"Classe en ligne {i}: ", score)
    #     i += 1

    # second_cluster, second_array = tools.biclustering_impact_viewer_bis(
    #     second_output, rows, cols, "Second output")
    # scores2 = scoring.evaluate_clustering(labels, second_cluster)
    # scores2 = scoring.better_score(scores2, LABELS)
    # i = 1
    # for score in scores2:
    #     print(f"Classe en ligne {i}: ", score)
    #     i += 1

    # third_cluster, third_array = tools.biclustering_impact_viewer_bis(
    #     third_output, rows, cols, "Third output")
    # scores3 = scoring.evaluate_clustering(labels, third_cluster)
    # scores3 = scoring.better_score(scores3, LABELS)
    # i = 1
    # for score in scores3:
    #     print(f"Classe ligne {i}: ", score)
    #     i += 1

    # # Mean signal for every row of third cluster #
    # line_indices = list()
    # for i in range(rows*cols):
    #     row, col = third_cluster.get_shape(i)
    #     if row not in line_indices:
    #         line_indices.append(row)
    # assert len(line_indices) == rows
    # line_blocks = [third_array[:line_indices[0]]]
    # indice = 0
    # for i in range(0, rows-1):
    #     line_blocks.append(
    #         third_array[indice + line_indices[i]:indice + line_indices[i] + line_indices[i+1]])
    #     indice += line_indices[i]
    # assert len(line_blocks) == rows
    # mean_blocks = list()
    # for i in range(rows):
    #     line_block = line_blocks[i]
    #     mean = np.sum(line_block, axis=0) / line_block.shape[0]
    #     mean_blocks.append(mean)
    # assert mean_blocks[0].shape[-1] == line_blocks[0].shape[-1]
    # for i in range(rows):
    #     plt.plot(mean_blocks[i])
    #     # plt.show()

    ### COMPUTING BEST BICLUSTERING ###

    # scoring.get_best_score(
    #     first_output, subjects, 30, scoring.DB_score, 'min')
    # plt.show()
    # scoring.get_best_score(
    #     second_output, subjects, 30, scoring.DB_score, 'min')
    # plt.show()
    # scoring.get_best_score(
    #     third_output, subjects, 30, scoring.DB_score, 'min')
    # plt.show()
