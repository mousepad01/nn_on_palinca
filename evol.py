import random
from enum import Enum

import tensorflow as tf
import numpy as np
from learner import HumanVsRnd, ModelState

from model import Res1D

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.keras.callbacks import History

class EvolState(Enum):

    UNINITIALIZED = "uninitialized"
    READY = "ready"

class Evolvable:
    """
    Evolution algorithm for hyperparameter search
    on a keras-based sequential model

    Structure of a chromosome and its interpretation:

    * NC - number of maximum Conv / Res layers
    * ND - number of maximum Dense layers
    * use_res - use res/conv layer
    * epochs_cnt - number of selection steps to perform
    * population_cnt - number of chromosomes

    * [gene for conv/res layer 1]
    * batch normalization
    * relu
    * [gene for conv/res layer 2]
    * batch normalization
    * relu
    ...
    * [gene for conv/res layer NC]
    
    * flatten layer 

    * [gene for dense layer 1]
    * relu
    * [gene for dense layer 2]
    * relu
    ...
    * [gene for dense layer ND]
    * relu

    * dense layer, 2 neurons
    * softmax

    gene for conv/res layer (66 bits):
    (all fields are considered unsigned integers)
    (LSB -> MSB)

    bit 0:      active / not active
    bit 1-9:    raw filter count (filter count = max(16, raw filter count))
    bit 9-11:   raw kernel size (kernel size = 2 + raw kernel size)

    gene for dense layer (66 bits):
    (all fields are considered unsigned integers)

    bit 0:      active / not active
    bit 1-9:    raw units count (units count = max(16, raw units count))
    bit 9-11:   raw dropout (dropout = raw dropout * 0.1)
    """

    BITS_PER_C_LAYER = 11

    C_LAYER_MASK = (2 ** BITS_PER_C_LAYER) - 1

    C_ACTIVE_MASK = 1
    C_FILTER_MASK = ((2 ** 9) - 1) & ~C_ACTIVE_MASK
    C_KER_SIZE_MASK = (((2 ** 11) - 1) & ~C_FILTER_MASK) & ~C_ACTIVE_MASK

    BITS_PER_D_LAYER = 11

    D_LAYER_MASK = (2 ** BITS_PER_D_LAYER) - 1

    D_ACTIVE_MASK = 1
    D_UNIT_MASK = ((2 ** 9) - 1) & ~D_ACTIVE_MASK
    D_DROPOUT_MASK = (((2 ** 11) - 1) & ~D_UNIT_MASK) & ~D_ACTIVE_MASK

    def __init__(self, nc=6, 
                        nd=6,
                        use_res=True,
                        epoch_cnt=20,
                        population_cnt=10,
                        elite_cnt=2,
                        selection_luck=0.1,
                        ):

        self.nc = nc
        """maximum number of Conv1D / Res1D layers"""

        self.nd = nd
        """maximum number of Dense layers"""
        
        self.use_res = use_res
        """if true, use Res1D, else use Conv1D"""

        self.epoch_cnt = epoch_cnt
        """number of selection steps to perform"""

        self.population_cnt = population_cnt
        """number of chromosomes"""

        self.population = []
        """list of (individual, score == fitness(individual))"""

        self.max_chrom_value = 2 ** (Evolvable.BITS_PER_C_LAYER * self.nc + Evolvable.BITS_PER_D_LAYER * self.nd)
        """chromosome with all bits set to 1"""

        self.state = EvolState.UNINITIALIZED
        self.learner: HumanVsRnd = None
        """associated model from learner.py"""

        self.elite_cnt = elite_cnt
        """first elite_cnt individuals have a copy that bypasses mutation and recombination"""

        self.selection_luck = selection_luck
        """randomly select a percentage 
            of the self.population_cnt to go to the next generation,
            without any regards to fitness score"""

    def init_learner(self, seed=None, **kwargs):
        
        self.learner = HumanVsRnd(**kwargs)

        if seed is not None:
            self.learner.seed(seed)

        self.learner.prep_data()

        self.state = EvolState.READY

    def init_population(self):

        self.population = []

        for c in self.random_chromosome(self.population_cnt):
            self.population.append([c, None])

    def random_chromosome(self, cnt):

        for _ in range(cnt):
            yield random.randint(0, self.max_chrom_value)

    def build_nn(self, chromosome):
        """chromosome -> model function"""

        model = Sequential()
        """obs: the layers will be considered in the following order:
                * the first (LSB) 11 bits will represent layer 1 conv/res
                ...
                * the last (MSB) 11 bits will represent layer 1 dense"""

        model.add(InputLayer(input_shape = (self.learner.timeslice_len, 1)))

        for _ in range(self.nc):
            
            gene = chromosome & Evolvable.LAYER_MASK
            chromosome >>= Evolvable.BITS_PER_C_LAYER

            if gene & Evolvable.C_ACTIVE_MASK:

                filter_cnt = max(16, gene & Evolvable.C_FILTER_MASK)
                ker_size = gene & Evolvable.C_KER_SIZE_MASK + 2
                
                if self.use_res:
                    model.add(Res1D(filters=filter_cnt, kernel_size=ker_size))
                else:
                    model.add(Conv1D(filters=filter_cnt, kernel_size=ker_size))
                
                model.add(BatchNormalization())
                model.add(ReLU())

        model.add(Flatten())

        for _ in range(self.nd):
            
            gene = chromosome & Evolvable.LAYER_MASK
            chromosome >>= Evolvable.BITS_PER_D_LAYER

            if gene & Evolvable.D_ACTIVE_MASK:

                unit_cnt = max(16, gene & Evolvable.D_UNIT_MASK)
                dropout = (gene & Evolvable.D_DROPOUT_MASK) * 0.1
                
                model.add(Dense(unit_cnt))
                model.add(ReLU())

                if dropout > 0:
                    model.add(Dropout(dropout))

        model.add(Dense(2))
        model.add(Softmax())

        return model

    def fitness(self):
        """perform fitness function evaluation
            on all self.population"""

        optimizer = SGD(1e-4, 0.9),
        loss = CategoricalCrossentropy(), 
        metrics = ['accuracy']
        
        for idx in range(self.population_cnt):

            nn = self.build_nn(self.population[idx][0])

            self.learner.model = nn
            self.learner.model.compile(optimizer = optimizer, 
                                        loss = loss, 
                                        metrics = metrics)
            self.learner.init_model()

            self.learner.model_state = ModelState.UNTRAINED
            self.learner.train_model(save_model_name=None)

            validation_results = self.learner.validate(save_model_name = "evolved_best_model")
            self.population[idx][1] = validation_results["accuracy"]

    def selection(self):
        """perform selection on self.population
        heuristic:
        * elitism applied on the first self.elite_cnt inidividuals
        * the first third of the population is kept
        * 1/10th of remaining individuals are randomly selected and kept"""
        
        self.fitness()
        self.population.sort(key = lambda x: x[1])

        elites = self.population[-2:]

        new_population = self.population[self.population_cnt // 3:]
        
        for _ in range(int(self.population_cnt * self.selection_luck)):
            new_population.append(random.choice(self.population))

        # rest of population cnt will be replinished at recombination
        self.population = new_population

        return elites

    def evolve(self):
        """Start the evolution process"""

        assert(self.state is EvolState.READY)

        self.init_population()

        for ep in range(self.epoch_cnt):

            print(f"[i] ================== Generation {ep} ==================")
            
            self.selection()
            self.mutate()
            self.recombinate()

        self.fitness()
        # TODO return / save the best config

    def mutate(self):
        """perform mutation on all self.population"""
        pass

    def recombination(self):
        """perform recombination on all self.population"""
        pass