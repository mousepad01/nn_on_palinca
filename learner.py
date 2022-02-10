from enum import Enum
from math import floor
import random
import os.path

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *

MODEL_PATH_PREFFIX = "./model/"

class DataState(Enum):
    
    UNINITIALIZED = "uninitialized"
    RAW = "raw"
    READY = "ready"

class ModelState(Enum):

    UNINITIALIZED = "uninitialized"
    UNTRAINED = "untrained"
    TRAINED = "trained"

class HumanVsRnd:
    """Its task is to differentiate
        between human keystroke timestamps
        and random keystroke timestamps"""

    def __init__(self, data_path = "timestamps.bin",
                        new_timeslice_thresh = 5,
                        micros_to_ms = 1000,
                        timeslice_len = 10,
                        random_to_human_ratio = 1.5,
                        random_min_decal = 200,
                        random_max_decal = 2000,
                        validation_ratio = 0.2,
                        ):

        assert(1 <= micros_to_ms <= 1000000)
        assert(new_timeslice_thresh > 0)
        assert(timeslice_len > 0)
        assert(random_to_human_ratio > 0) 
        assert(random_min_decal > 0)
        assert(random_max_decal < new_timeslice_thresh * micros_to_ms)
        assert(0 < validation_ratio < 1)

        self.timeslice_len = timeslice_len
        """timeslice length to be fed to the ML model"""

        self.micros_to_ms = micros_to_ms
        """conversion between microseconds (raw_data[idx][1]) to 'miliseconds'"""

        self.data_path = data_path
        self.new_timeslice_thresh = new_timeslice_thresh
        """threshold (in seconds) that forces a new timeslice"""

        self.data_state = DataState.UNINITIALIZED
        self.data = None
        """human train data"""

        self.random_data_state = DataState.UNINITIALIZED
        self.random_data = None
        """random train data"""

        self.v_data_state = DataState.UNINITIALIZED
        self.v_data = None
        """human validation data"""

        self.v_random_data_state = DataState.UNINITIALIZED
        self.v_random_data = None
        """random validation data"""

        self.random_min_decal = random_min_decal
        """minimum milisec distance between random keystrokes"""
        self.random_max_decal = random_max_decal
        """maximum milisec distance between random keystrokes"""

        self.random_to_human_ratio = random_to_human_ratio
        """ratio of random vs human data"""

        self.validation_ratio = validation_ratio
        """how much data to be used as validation data"""

        self.model_state = ModelState.UNINITIALIZED
        self.model = None
        """model"""

    def seed(self, seed = 0):
        """set seed for all rnd"""

        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # data

    def load_data(self):

        data_buf = []

        with open(self.data_path, "rb") as data:
            alldata = data.read()

        last_s, last_ms = 0, 0

        idx = 0
        while idx < len(alldata):

            s = int.from_bytes(alldata[idx: idx + 8], 'little')
            ms = int.from_bytes(alldata[idx + 8: idx + 16], 'little')

            assert((s, ms) > (last_s, last_ms))

            data_buf.append((s, ms))

            # print(f"{idx // 16}: {s}, {ms}")

            idx += 16

        self.data = data_buf
        self.data_state = DataState.RAW

    def format_data(self):
        """* make timestamps relative to one another
            * split in different timeslices"""

        raw_data = self.data

        data_timeslices = []

        last_t = 0

        current_timeslice_base_s = 0
        current_timeslice_base_ms = 0
        current_timeslice_len = 0
        
        idx = 0
        timeslice_idx = -1

        while idx < len(raw_data):

            if current_timeslice_len == self.timeslice_len + 1 or \
                raw_data[idx][0] - last_t > self.new_timeslice_thresh:
                
                timeslice_idx += 1
                data_timeslices.append([])

                current_timeslice_base_s = raw_data[idx][0]
                current_timeslice_base_ms = raw_data[idx][1] // self.micros_to_ms
                current_timeslice_len = 0

            data_timeslices[-1].append([(raw_data[idx][0] - current_timeslice_base_s) * self.micros_to_ms + 
                                    raw_data[idx][1] // self.micros_to_ms - current_timeslice_base_ms])

            assert(data_timeslices[-1][-1][0] >= 0)

            current_timeslice_len += 1
            last_t = raw_data[idx][0]
            idx += 1

        self.data = []
        for tslice in data_timeslices:

            assert(len(tslice) <= self.timeslice_len + 1)

            if len(tslice) == self.timeslice_len + 1:
                self.data.append(tslice[1:])    # first entry always 0, also add a 'dummy' dimension

        v_data_len = int(self.validation_ratio * len(self.data))
        splitpoint = random.randint(0, len(self.data) - v_data_len)

        self.v_data = self.data[splitpoint: splitpoint + v_data_len]
        self.data = self.data[:splitpoint] + self.data[splitpoint + v_data_len:]

        print(f"[i] Data (train, human) timeslice count: {len(self.data)}")
        print(f"[i] Data (validate, human) timeslice count: {len(self.v_data)}")

        self.data = np.array(self.data, dtype = np.float32)
        self.data_state = DataState.READY

        self.v_data = np.array(self.v_data, dtype = np.float32)
        self.v_data_state = DataState.READY

    def random_data_gen(self):
        """generate 'random' keystroke timestamps"""

        assert(self.data_state is DataState.READY)
        
        random_data_len = int(self.random_to_human_ratio * self.data.shape[0])
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

        self.v_random_data = []
        for _ in range(self.v_data.shape[0]):
            
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
        self.random_data_gen()

    # model

    def init_model(self, optimizer = SGD(1e-4, 0.9),
                            loss = CategoricalCrossentropy(), 
                            metrics = ['accuracy']):

        assert(self.model_state is ModelState.UNINITIALIZED)

        self.model = Sequential([
            InputLayer(input_shape = (self.timeslice_len, 1)),

            Conv1D(16, kernel_size = 4),
            BatchNormalization(),
            ReLU(),
            #MaxPool1D(2),

            Conv1D(32, kernel_size = 3),
            BatchNormalization(),
            ReLU(),
            #MaxPool1D(2),

            Conv1D(64, kernel_size = 3),
            BatchNormalization(),
            ReLU(),
            #MaxPool1D(2),

            Flatten(),

            Dense(64),
            ReLU(),

            Dense(32),
            ReLU(),

            Dense(16),
            ReLU(),

            Dense(2),
            Softmax()
        ])

        self.model.compile(optimizer = optimizer, 
                            loss = loss, 
                            metrics = metrics)

        self.model_state = ModelState.UNTRAINED

    def load_best_model(self):

        with open(f"{MODEL_PATH_PREFFIX}best_accuracy.txt", "rb") as bestscore_f:
            bestscore = int.from_bytes(bestscore_f.read(8), 'little')
            model_path = bestscore_f.read().decode()

        print(f"[i] loading model with validation accuracy {bestscore / 1000}")

        self.load_model_(model_path)

    def load_model_(self, model_path):

        assert(self.model_state is ModelState.UNINITIALIZED)

        self.model = load_model(model_path)
        self.model_state = ModelState.TRAINED

    def save_model_(self, model_path):
        self.model.save(model_path)

    def train_model(self, save_model_name = None,
                            batch_size = 32,
                            epochs = 10):

        assert(self.data_state is DataState.READY)
        assert(self.random_data_state is DataState.READY)

        assert(self.model_state in [ModelState.UNTRAINED, ModelState.TRAINED])

        train_data = tf.concat([self.data, self.random_data], axis = 0)
        train_data_labels = tf.concat([np.ones((self.data.shape[0],)), np.zeros((self.random_data.shape[0],))], axis = 0)
        train_data_labels = tf.keras.utils.to_categorical(train_data_labels, 2)

        print(f"\n[i] training started\n")

        self.model.fit(x = train_data, y = train_data_labels,
                        batch_size = batch_size,
                        epochs = epochs)

        print(f"\n[i] training ended\n")

        self.model_state = ModelState.TRAINED

        if save_model_name is not None:
            self.save_model_(f"{MODEL_PATH_PREFFIX}{save_model_name}")

    def validate(self, save_model_name = None, save_if_best = True):
        
        assert(self.v_data_state is DataState.READY)
        assert(self.v_random_data_state is DataState.READY)

        assert(self.model_state is ModelState.TRAINED)

        validation_data = tf.concat([self.v_data, self.v_random_data], axis = 0)
        validation_data_labels = tf.concat([np.ones((self.v_data.shape[0],)), np.zeros((self.v_random_data.shape[0],))], axis = 0)
        validation_data_labels = tf.keras.utils.to_categorical(validation_data_labels, 2)

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

                        model.save_model_(f"{MODEL_PATH_PREFFIX}{save_model_name}")

                else:

                    with open(best_score_path, "wb+") as bestscore_f:
                        bestscore_f.write(int.to_bytes(0, 8, 'little'))
                        bestscore_f.write(save_model_name.encode())
                        
                    self.save_model_(f"{MODEL_PATH_PREFFIX}{save_model_name}")

            else:
                self.save_model_(f"{MODEL_PATH_PREFFIX}{save_model_name}")

if __name__ == "__main__":

    model = HumanVsRnd()
    model.seed(123)

    model.prep_data()

    model.init_model()
    model.train_model()
    model.validate(save_model_name = "bestmodel")
