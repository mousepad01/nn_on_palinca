from enum import Enum
from math import floor
import random
import os.path

from model import *

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *

MODEL_PATH_PREFFIX = "./model_data_multi/"

class DataState(Enum):
    
    UNINITIALIZED = "uninitialized"
    RAW = "raw"
    READY = "ready"

class ModelState(Enum):

    UNINITIALIZED = "uninitialized"
    UNTRAINED = "untrained"
    TRAINED = "trained"

class KeystrokeFingerprintClassificator:
    """Its task is to differentiate
        between different humans' keystroke timestamps
        and also random keystroke timestamps

        NOTE: if you want to also include random data,
                set versus_random to True
                (if versus_random is false, other random params will be ignored)"""

    def __init__(self, 
                        contrastive_learning = True,

                        data_paths = ["timestamps.bin"],
                        new_timeslice_thresh = 4,
                        micros_to_ms = 1000,
                        timeslice_len = 15,

                        versus_random = True,
                        random_to_human_ratio = 1.5,
                        random_min_decal = 80,
                        random_max_decal = 300,
                        validation_ratio = 0.2,
                        ):

        if type(data_paths) is str:
            data_paths = [[data_paths]]

        elif type(data_paths) is list:

            if type(data_paths[0]) is str:
                data_paths = [data_paths]

        assert(len(data_paths) >= 1)
        assert(not(len(data_paths) == 1 and len(data_paths[0]) == 1 and versus_random is False))
        assert(1 <= micros_to_ms <= 1000000)
        assert(new_timeslice_thresh > 0)
        assert(timeslice_len > 0)
        assert(random_to_human_ratio > 0) 
        assert(random_min_decal > 0)
        assert(random_max_decal < new_timeslice_thresh * (1000000 // micros_to_ms))
        assert(0 < validation_ratio < 1)

        self.human_cnt = len(data_paths)
        """number of classes (-1 if versus_random is true)"""

        self.versus_random = versus_random
        """whether to include random as a class, or classify only
            humans vs humans"""

        self.timeslice_len = timeslice_len
        """timeslice length to be fed to the ML model"""

        self.micros_to_ms = micros_to_ms
        """conversion between microseconds (raw_data[idx][1]) to 'miliseconds'"""

        self.s_to_ms = 1000000 // micros_to_ms
        """conversion between seconds (raw_data[idx][0]) to 'miliseconds'"""

        self.data_paths = data_paths
        self.new_timeslice_thresh = new_timeslice_thresh
        """threshold (in seconds) that forces a new timeslice"""

        self.data_state = DataState.UNINITIALIZED
        self.data = {} # {class: [data...]}
        """human train data"""

        self.random_data_state = DataState.UNINITIALIZED
        self.random_data = [] # [data...]
        """random train data"""

        self.v_data_state = DataState.UNINITIALIZED
        self.v_data = {} # {class: [data...]}
        """human validation data"""

        self.v_random_data_state = DataState.UNINITIALIZED
        self.v_random_data = [] # [data...]
        """random validation data"""

        self.random_min_decal = random_min_decal
        """minimum milisec distance between random keystrokes"""
        self.random_max_decal = random_max_decal
        """maximum milisec distance between random keystrokes"""

        self.random_to_human_ratio = random_to_human_ratio
        """ratio of random vs human data"""

        self.validation_ratio = validation_ratio
        """how much data to be used as validation data"""

        self.contrastive_learning = contrastive_learning
        """whether the model is trained with contrastive learning
            or in a 'classic' manner"""

        self.model_state = ModelState.UNINITIALIZED
        self.model: Model = None
        """ (final) model"""

        if self.contrastive_learning:
            
            self.encoder: Model = None
            """convolutional part of the model"""

            self.feature_extractor: Model = None
            """used for creating the embedding vector"""

            self.classifier: Model = None
            """attach to encoder to obtain the final classificator model"""

        """DECLARED BELOW, ONLY FOR INTERNAL USE"""

        self._dataname_to_class = {tuple(data_paths[idx]): idx for idx in range(len(data_paths))}
        self._class_to_dataname = {idx: data_paths[idx] for idx in range(len(data_paths))}
        # currently redundant, but to make sure no bugs appear due to list operations
        # NOTE: if changing this mapping, REWORK RANDOM DATA CLASSIFICATION LABELS !!!

        if self.contrastive_learning:

            self._contrastive_loss = None
            self._contrastive_optimizer = None

            self._classifier_loss = None
            self._classifier_optimizer = None

            self._metrics = None

            """auxiliary variables"""

    # data

    def load_data(self):

        data_bufs = {}

        # sort files by timestamps 
        # so the loading function remains consistent

        def _tstamp_cmp(path):

            with open(path, "rb") as data:
                return (int.from_bytes(data.read(8), 'little'),
                        int.from_bytes(data.read(8), 'little'))

        for class_, data_paths in self._class_to_dataname.items():
            self._class_to_dataname[class_] = sorted(data_paths, key = _tstamp_cmp)

        for class_, data_paths in self._class_to_dataname.items():

            last_s, last_micros = 0, 0

            data_bufs.update({class_: []})
            for data_path in data_paths:
                
                with open(data_path, "rb") as data:
                    alldata = data.read()

                idx = 0
                while idx < len(alldata):

                    s = int.from_bytes(alldata[idx: idx + 8], 'little')
                    micros = int.from_bytes(alldata[idx + 8: idx + 16], 'little')

                    assert((s, micros) >= (last_s, last_micros))
                    last_s, last_micros = s, micros

                    data_bufs[class_].append((s, micros))

                    # print(f"{idx // 16}: {s}, {micros}")

                    idx += 16
                
        min_data_len = 2 ** 100

        for class_ in data_bufs.keys():
            min_data_len = min(min_data_len, len(data_bufs[class_]))

        for class_ in data_bufs.keys():
            data_bufs[class_] = data_bufs[class_][:min_data_len]

        self.data = data_bufs
        self.data_state = DataState.RAW

    def format_data(self):
        """* make timestamps relative to one another
            * split in different timeslices"""

        for class_, raw_data in self.data.items():

            data_timeslices = []

            last_t = 0

            current_timeslice_base = 0
            current_timeslice_len = 0
            
            idx = 0
            timeslice_idx = -1

            while idx < len(raw_data):

                if current_timeslice_len == self.timeslice_len + 1 or \
                    raw_data[idx][0] - last_t > self.new_timeslice_thresh:
                    
                    timeslice_idx += 1
                    data_timeslices.append([])

                    current_timeslice_base = raw_data[idx][0] * self.s_to_ms + raw_data[idx][1] // self.micros_to_ms
                    current_timeslice_len = 0

                data_timeslices[-1].append([(raw_data[idx][0] * self.s_to_ms + raw_data[idx][1] // self.micros_to_ms) - \
                                                current_timeslice_base])

                assert(data_timeslices[-1][-1][0] >= 0)

                current_timeslice_len += 1
                last_t = raw_data[idx][0]
                idx += 1

            self.data[class_] = []
            for tslice in data_timeslices:

                assert(len(tslice) <= self.timeslice_len + 1)

                if len(tslice) == self.timeslice_len + 1:
                    self.data[class_].append(tslice[1:])    # first entry always 0, also add a 'dummy' dimension

            v_data_len = int(self.validation_ratio * len(self.data[class_]))
            splitpoint = random.randint(0, len(self.data[class_]) - v_data_len)

            self.v_data.update({class_: self.data[class_][splitpoint: splitpoint + v_data_len]})
            self.data[class_] = self.data[class_][:splitpoint] + self.data[class_][splitpoint + v_data_len:]

            print(f"[i] Data (train, human {class_}) timeslice count: {len(self.data[class_])}")
            print(f"[i] Data (validate, human {class_}) timeslice count: {len(self.v_data[class_])}")

            self.data[class_] = np.array(self.data[class_], dtype = np.float32)
            self.v_data[class_] = np.array(self.v_data[class_], dtype = np.float32)

        self.data_state = DataState.READY
        self.v_data_state = DataState.READY

    def random_data_gen(self):
        """generate random keystroke timestamps"""

        assert(self.data_state is DataState.READY)
        
        total_human_data = 0
        for _, d in self.data.items():
            total_human_data += d.shape[0]

        random_data_len = int(self.random_to_human_ratio * total_human_data)
        self.random_data = []

        for _ in range(random_data_len):
            
            self.random_data.append([])
            self.t_offset = 0

            for _ in range(self.timeslice_len):
                
                self.t_offset += random.randint(self.random_min_decal, self.random_max_decal)
                self.random_data[-1].append([self.t_offset])

        print(f"[i] Data (train, random) timeslice count: {len(self.random_data)}")

        self.random_data = np.array(self.random_data, dtype = np.float32)
        self.random_data_state = DataState.READY

        #TODO calibrate ratio between 
        #       random validation data and other human data ???

        self.v_random_data = []
        for _ in range(int(random_data_len * self.validation_ratio)):
            
            self.v_random_data.append([])
            self.t_offset = 0

            for _ in range(self.timeslice_len):
                
                self.t_offset += random.randint(self.random_min_decal, self.random_max_decal)
                self.v_random_data[-1].append([self.t_offset])

        print(f"[i] Data (validation, random) timeslice count: {len(self.v_random_data)}")

        self.v_random_data = np.array(self.v_random_data, dtype = np.float32)
        self.v_random_data_state = DataState.READY

    def prep_data(self):

        self.load_data()
        self.format_data()

        if self.versus_random:
            self.random_data_gen()

    # model

    def init_model(self, classifier_optimizer = SGD(1e-4, 0.9),
                            classifier_loss = CategoricalCrossentropy(),
                            
                            contrastive_optimizer = SGD(1e-5, 0.9),
                            contrastive_loss = SupCon(0.1),
                            
                            metrics = ['accuracy']):

        """NOTE: ignore cntrastive optimizer and contrastive loss
                    if contrastive learning is not applied"""

        assert(self.model_state is ModelState.UNINITIALIZED)

        def _init_classic_model():

            assert(self.human_cnt if not self.versus_random else self.human_cnt + 1 <= 16)
            
            self.model = Sequential([
                InputLayer(input_shape = (self.timeslice_len, 1)),

                Inception1D(16),
                BatchNormalization(),
                ReLU(),

                Inception1D(64),
                BatchNormalization(),
                ReLU(),

                Inception1D(256),
                BatchNormalization(),
                ReLU(),

                Flatten(),

                Dense(64),
                ReLU(),

                Dropout(0.4),

                Dense(16),
                ReLU(),

                Dense(self.human_cnt if not self.versus_random else self.human_cnt + 1),
                Softmax()
            ])

            self.model.compile(optimizer = classifier_optimizer, 
                                loss = classifier_loss, 
                                metrics = metrics)

            self.model_state = ModelState.UNTRAINED

        def _init_contrastive_model():
            
            self.encoder = \
                Sequential([

                    InputLayer(input_shape = (self.timeslice_len, 1)),

                    Inception1D(16),
                    BatchNormalization(),
                    ReLU(),

                    Inception1D(64),
                    BatchNormalization(),
                    ReLU(),

                    Inception1D(256),
                    BatchNormalization(),
                    ReLU(),

                    Flatten()
                ])

            self.feature_extractor = \
                Sequential([

                    self.encoder,

                    Dense(128),
                    ReLU()
                ])

            self.classifier = None # later, after training
            '''Sequential([

                    self.encoder,

                    Dropout(0.4),

                    Dense(64),
                    ReLU(),

                    Dropout(0.4),

                    Dense(16),
                    ReLU(),

                    Dense(self.human_cnt if not self.versus_random else self.human_cnt + 1),
                    Softmax()
                ])'''

            self.model = self.classifier    # this will hold even after training 
                                            # (currently, does nothing since classifier is None)

            self._contrastive_loss = contrastive_loss
            self._contrastive_optimizer = contrastive_optimizer

            self._classifier_loss = classifier_loss
            self._classifier_optimizer = classifier_optimizer

            self._metrics = metrics

            self.model_state = ModelState.UNTRAINED

        if self.contrastive_learning is True:
            return _init_contrastive_model()

        else:
            return _init_classic_model()

    # TODO rework load/save model for contrastive learning

    def load_best_model(self):

        if self.contrastive_learning is True:
            raise RuntimeError("save/load not currently supported for contrastive learning")

        with open(f"{MODEL_PATH_PREFFIX}best_accuracy.txt", "rb") as bestscore_f:
            bestscore = int.from_bytes(bestscore_f.read(8), 'little')
            model_path = bestscore_f.read().decode()

        print(f"[i] loading model with validation accuracy {bestscore / 1000}")

        self.load_model_(model_path)

    def load_model_(self, model_path):

        if self.contrastive_learning is True:
            raise RuntimeError("save/load not currently supported for contrastive learning")

        assert(self.model_state is ModelState.UNINITIALIZED)

        self.model = load_model(model_path, custom_objects={"Res1D": Res1D, "Inception1D": Inception1D})
        self.model_state = ModelState.TRAINED

    def save_model_(self, model_path):

        if self.contrastive_learning is True:
            raise RuntimeError("save/load not currently supported for contrastive learning")

        self.model.save(model_path, overwrite=True)

    def train_model(self, save_model_name = None,
                            batch_size = 32,
                            epochs = 10,
                            display_history = True):

        assert(self.data_state is DataState.READY)

        if self.versus_random:
            assert(self.random_data_state is DataState.READY)

        assert(self.model_state in [ModelState.UNTRAINED, ModelState.TRAINED])

        train_data = []
        train_data_labels = []

        for class_, d in self.data.items():

            train_data.append(d)
            train_data_labels.append(np.full((d.shape[0],), fill_value = class_))

        if self.versus_random:
            
            train_data.append(self.random_data)
            train_data_labels.append(np.full((self.random_data.shape[0],), fill_value = self.human_cnt))

        train_data = tf.concat(train_data, axis = 0)
        train_data_labels = tf.concat(train_data_labels, axis = 0)

        print(f"\n[i] training started\n")

        def _train_classic():

            nonlocal train_data
            nonlocal train_data_labels
            
            # one-hot encoding
            train_data_labels = tf.keras.utils.to_categorical(train_data_labels, 
                                self.human_cnt if not self.versus_random else self.human_cnt + 1)

            history = self.model.fit(x = train_data, y = train_data_labels,
                                    batch_size = batch_size,
                                    epochs = epochs)

            return history.history

        def _train_contrastive():

            nonlocal train_data
            nonlocal train_data_labels

            print(f"\n[i] contrastive learning phase started\n")

            print(train_data.shape, train_data_labels.shape)

            self.feature_extractor.compile(optimizer = self._contrastive_optimizer, 
                                            loss = self._contrastive_loss)

            history_fst = self.feature_extractor.fit(x = train_data, y = train_data_labels,
                                                    batch_size = batch_size,
                                                    epochs = epochs)

            print(f"\n[i] contrastive learning phase ended\n")
            print(f"\n[i] classifier training started\n")

            # freeze encoder weights
            for layer in self.encoder.layers:
                layer.trainable = False

            self.classifier = \
                Sequential([

                    self.encoder,

                    Dense(128),
                    ReLU(),

                    Dropout(0.4),

                    Dense(64),
                    ReLU(),

                    Dense(16),
                    ReLU(),

                    Dense(self.human_cnt if not self.versus_random else self.human_cnt + 1),
                    Softmax()
                ])

            self.classifier.compile(optimizer = self._classifier_optimizer, 
                                    loss = self._classifier_loss, 
                                    metrics = self._metrics)

            train_data_labels = tf.keras.utils.to_categorical(train_data_labels, 
                                self.human_cnt if not self.versus_random else self.human_cnt + 1)

            history_snd = self.classifier.fit(x = train_data, y = train_data_labels,
                                                batch_size = batch_size,
                                                epochs = epochs)

            self.model = self.classifier

            history = {"contrastive": history_fst.history, "classifier": history_snd.history}
            return history

        if self.contrastive_learning:
            history = _train_contrastive()

        else:
            history = _train_classic()

        print(f"\n[i] training ended\n")

        if display_history:
            self.display_stats(history, self.contrastive_learning)

        self.model_state = ModelState.TRAINED

        if save_model_name is not None:
            self.save_model_(f"{MODEL_PATH_PREFFIX}{save_model_name}")

    def validate(self, save_model_name = None, save_if_best = True):
        
        assert(self.v_data_state is DataState.READY)

        if self.versus_random:
            assert(self.v_random_data_state is DataState.READY)

        assert(self.model_state is ModelState.TRAINED)

        validation_data = []
        validation_data_labels = []

        for class_, d in self.v_data.items():

            validation_data.append(d)
            validation_data_labels.append(np.full((d.shape[0],), fill_value = class_))

        if self.versus_random:
            
            validation_data.append(self.v_random_data)
            validation_data_labels.append(np.full((self.v_random_data.shape[0],), fill_value = self.human_cnt))

        validation_data = tf.concat(validation_data, axis = 0)
        validation_data_labels = tf.concat(validation_data_labels, axis = 0)
        validation_data_labels = tf.keras.utils.to_categorical(validation_data_labels, 
                                        self.human_cnt if not self.versus_random else self.human_cnt + 1)

        print(f"\n[i] validation started\n")

        validate_data_predictions = self.model.evaluate(validation_data, validation_data_labels,
                                                        batch_size = validation_data.shape[0],
                                                        return_dict = True)

        print(f"[*] validation ended: {validate_data_predictions}")

        if save_model_name is not None:

            if save_if_best:

                best_score_path = f"{MODEL_PATH_PREFFIX}best_accuracy.txt"
                
                if os.path.isfile(best_score_path):
                    
                    with open(best_score_path, "rb") as bestscore_f:
                        bestscore = int.from_bytes(bestscore_f.read(8), 'little')

                    current_score = validate_data_predictions['accuracy']
                    current_score = floor(current_score * 1000)

                    if current_score > bestscore:

                        with open(best_score_path, "wb+") as bestscore_f:
                            bestscore_f.write(int.to_bytes(current_score, 8, 'little'))
                            bestscore_f.write(save_model_name.encode())

                        self.save_model_(f"{MODEL_PATH_PREFFIX}{save_model_name}")

                else:

                    with open(best_score_path, "wb+") as bestscore_f:
                        bestscore_f.write(int.to_bytes(0, 8, 'little'))
                        bestscore_f.write(save_model_name.encode())
                        
                    self.save_model_(f"{MODEL_PATH_PREFFIX}{save_model_name}")

            else:
                self.save_model_(f"{MODEL_PATH_PREFFIX}{save_model_name}")

        return validate_data_predictions

    # stats, other varius methods

    def seed(self, seed = 0):
        """set seed for all rnd"""

        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

    def display_stats(self, stats: dict, contrastive_learning: bool = False):
        """display stats
            (currently only for accuracy and loss)"""

        def _plot_classic():

            assert('accuracy' in stats.keys() and 'loss' in stats.keys())

            epoch_axis = [i for i in range(1, len(stats['accuracy']) + 1)]

            fig, acc_loss_epoch = plt.subplots(1)
            
            acc_loss_epoch.plot(epoch_axis,
                                stats['accuracy'],
                                color = 'green',
                                label = 'accuracy')

            acc_loss_epoch.plot(epoch_axis,
                                stats['loss'],
                                color = 'blue',
                                label = 'loss')

            acc_loss_epoch.legend(loc = "upper left")
            plt.show()

        def _plot_contrastive_learning():
            
            assert('contrastive' in stats.keys() and 'classifier' in stats.keys())

            h_fst = stats["contrastive"]
            h_snd = stats["classifier"]

            assert('loss' in h_fst.keys())
            assert('accuracy' in h_snd.keys() and 'loss' in h_snd.keys())

            # NOTE: same number of epochs for both training steps
            epoch_axis = [i for i in range(1, len(h_fst['loss']) + 1)]

            fig, (loss_contrastive, acc_loss_classifier) = plt.subplots(2)

            loss_contrastive.set_title("Encoder trainig loss")
            acc_loss_classifier.set_title("Classifier training loss and accuracy")

            loss_contrastive.plot(epoch_axis,
                                    h_fst['loss'],
                                    color = 'blue',
                                    label = 'loss')
            
            acc_loss_classifier.plot(epoch_axis,
                                        h_snd['accuracy'],
                                        color = 'green',
                                        label = 'accuracy')

            acc_loss_classifier.plot(epoch_axis,
                                        h_snd['loss'],
                                        color = 'blue',
                                        label = 'loss')

            acc_loss_classifier.legend(loc = "upper left")
            plt.show()

        if contrastive_learning:
            _plot_contrastive_learning()

        else:
            _plot_classic()
