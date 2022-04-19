from learner import KeystrokeFingerprintClassificator
from model import *

from sys import argv

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *

def launch():

    assert(len(argv) > 1)

    if argv[1] == "0":
        '''base 88%'''

        model = KeystrokeFingerprintClassificator(contrastive_learning = True,
                                                    rnn = True,
                                                    overlap_timeslices = True,
                                                    data_paths = [["timestamps_d6d4_1.bin",
                                                                    "timestamps_d6d4_2.bin",
                                                                    "timestamps_d6d4_3.bin",
                                                                    "timestamps_d6d4_4.bin",
                                                                    "timestamps_d6d4_5.bin",
                                                                    "timestamps_d6d4_6.bin"],
                                                                ["timestamps_bb47.bin"]],
                                                    versus_random = False,
                                                    micros_to_ms = 100,
                                                    timeslice_len = 30,
                                                    validation_ratio = 0.1)

        model.seed(123)

        model.prep_data()

        model.init_model()
        model.train_model(classifier_epochs=60, encoder_epochs=100, batch_size=16)
        model.validate(save_model_name = "bestmodel_multiclassif")

    elif argv[1] == "1":
        '''base 81%'''

        model = KeystrokeFingerprintClassificator(contrastive_learning = True,
                                                    rnn = True,
                                                    overlap_timeslices = True,
                                                    data_paths = [["timestamps_d6d4_1.bin",
                                                                    "timestamps_d6d4_2.bin",
                                                                    "timestamps_d6d4_3.bin",
                                                                    "timestamps_d6d4_4.bin",
                                                                    "timestamps_d6d4_5.bin",
                                                                    "timestamps_d6d4_6.bin"],
                                                                ["timestamps_bb47.bin"]],
                                                    versus_random = False,
                                                    micros_to_ms = 100,
                                                    timeslice_len = 30,
                                                    validation_ratio = 0.1)

        model.seed(55556)

        model.prep_data()

        model.init_model()
        model.train_model(classifier_epochs=60, encoder_epochs=100, batch_size=16)
        model.validate(save_model_name = "bestmodel_multiclassif")

    elif argv[1] == "2":
        '''non-overlapping 74.5%'''

        model = KeystrokeFingerprintClassificator(contrastive_learning = True,
                                                    rnn = True,
                                                    overlap_timeslices = False,
                                                    data_paths = [["timestamps_d6d4_1.bin",
                                                                    "timestamps_d6d4_2.bin",
                                                                    "timestamps_d6d4_3.bin",
                                                                    "timestamps_d6d4_4.bin",
                                                                    "timestamps_d6d4_5.bin",
                                                                    "timestamps_d6d4_6.bin"],
                                                                ["timestamps_bb47.bin"]],
                                                    versus_random = False,
                                                    micros_to_ms = 100,
                                                    timeslice_len = 30,
                                                    validation_ratio = 0.1)

        model.seed(123)

        model.prep_data()

        model.init_model()
        model.train_model(classifier_epochs=60, encoder_epochs=100, batch_size=16)
        model.validate(save_model_name = "bestmodel_multiclassif")

    elif argv[1] == "3":
        '''tslice len 16 82%'''

        model = KeystrokeFingerprintClassificator(contrastive_learning = True,
                                                    rnn = True,
                                                    overlap_timeslices = True,
                                                    data_paths = [["timestamps_d6d4_1.bin",
                                                                    "timestamps_d6d4_2.bin",
                                                                    "timestamps_d6d4_3.bin",
                                                                    "timestamps_d6d4_4.bin",
                                                                    "timestamps_d6d4_5.bin",
                                                                    "timestamps_d6d4_6.bin"],
                                                                ["timestamps_bb47.bin"]],
                                                    versus_random = False,
                                                    micros_to_ms = 100,
                                                    timeslice_len = 16,
                                                    validation_ratio = 0.1)

        model.seed(123)

        model.prep_data()

        model.init_model()
        model.train_model(classifier_epochs=60, encoder_epochs=100, batch_size=16)
        model.validate(save_model_name = "bestmodel_multiclassif")

    elif argv[1] == "4":
        '''without contrastive learning'''

        model = KeystrokeFingerprintClassificator(contrastive_learning = False,
                                                    rnn = True,
                                                    overlap_timeslices = True,
                                                    data_paths = [["timestamps_d6d4_1.bin",
                                                                    "timestamps_d6d4_2.bin",
                                                                    "timestamps_d6d4_3.bin",
                                                                    "timestamps_d6d4_4.bin",
                                                                    "timestamps_d6d4_5.bin",
                                                                    "timestamps_d6d4_6.bin"],
                                                                ["timestamps_bb47.bin"]],
                                                    versus_random = False,
                                                    micros_to_ms = 100,
                                                    timeslice_len = 30,
                                                    validation_ratio = 0.1)

        model.seed(123)

        model.prep_data()

        model.init_model()
        model.train_model(classifier_epochs=150, encoder_epochs=100, batch_size=16)
        model.validate(save_model_name = "bestmodel_multiclassif")

    elif argv[1] == "5":
        '''without LSTM 100 encoder epochs'''

        model = KeystrokeFingerprintClassificator(contrastive_learning = True,
                                                    rnn = False,
                                                    overlap_timeslices = True,
                                                    data_paths = [["timestamps_d6d4_1.bin",
                                                                    "timestamps_d6d4_2.bin",
                                                                    "timestamps_d6d4_3.bin",
                                                                    "timestamps_d6d4_4.bin",
                                                                    "timestamps_d6d4_5.bin",
                                                                    "timestamps_d6d4_6.bin"],
                                                                ["timestamps_bb47.bin"]],
                                                    versus_random = False,
                                                    micros_to_ms = 100,
                                                    timeslice_len = 30,
                                                    validation_ratio = 0.1)

        model.seed(123)

        model.prep_data()

        model.init_model()
        model.train_model(classifier_epochs=60, encoder_epochs=100, batch_size=16)
        model.validate(save_model_name = "bestmodel_multiclassif")

    elif argv[1] == "6":
        '''without LSTM 40 encoder epochs'''

        model = KeystrokeFingerprintClassificator(contrastive_learning = True,
                                                    rnn = False,
                                                    overlap_timeslices = True,
                                                    data_paths = [["timestamps_d6d4_1.bin",
                                                                    "timestamps_d6d4_2.bin",
                                                                    "timestamps_d6d4_3.bin",
                                                                    "timestamps_d6d4_4.bin",
                                                                    "timestamps_d6d4_5.bin",
                                                                    "timestamps_d6d4_6.bin"],
                                                                ["timestamps_bb47.bin"]],
                                                    versus_random = False,
                                                    micros_to_ms = 100,
                                                    timeslice_len = 30,
                                                    validation_ratio = 0.1)

        model.seed(123)

        model.prep_data()

        model.init_model()
        model.train_model(classifier_epochs=60, encoder_epochs=40, batch_size=16)
        model.validate(save_model_name = "bestmodel_multiclassif")

    elif argv[1] == "7":
        '''replace inc1d with ressidual blocks
            NOTE to test this, change the model with the
                    one that is commented in model.init_model() method 
                    in Learner.py, line ~490'''

        model = KeystrokeFingerprintClassificator(contrastive_learning = True,
                                                    rnn = True,
                                                    overlap_timeslices = True,
                                                    data_paths = [["timestamps_d6d4_1.bin",
                                                                    "timestamps_d6d4_2.bin",
                                                                    "timestamps_d6d4_3.bin",
                                                                    "timestamps_d6d4_4.bin",
                                                                    "timestamps_d6d4_5.bin",
                                                                    "timestamps_d6d4_6.bin"],
                                                                ["timestamps_bb47.bin"]],
                                                    versus_random = False,
                                                    micros_to_ms = 100,
                                                    timeslice_len = 30,
                                                    validation_ratio = 0.1)

        model.seed(123)

        model.prep_data()

        model.init_model()
        model.train_model(classifier_epochs=60, encoder_epochs=100, batch_size=16)
        model.validate(save_model_name = "bestmodel_multiclassif")

if __name__ == "__main__":
    launch()