import random
from enum import Enum
from copy import deepcopy
import time

import tensorflow as tf
import numpy as np
from learner import HumanVsRnd, KeystrokeFingerprintClassificator, ModelState

from model import Res1D

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *

# TODO make mutation is complement attribute also variable

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

    Chromosome interpretation:

    * [gene for training metadata (lr, momentum, batch_size)]

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

    (all fields are considered unsigned integers)
    (LSB -> MSB)

    gene for metadata (6 bits):

    bit 0-2: raw learning rate (learning rate = 10 ** -(raw learning rate + 2))
    bit 2-4: raw momentum (momentum = 0.6 + 0.1 * raw momentum)
    bit 4-6: raw batch size (batch size = 2 ** (raw batch size + 4))

    gene for conv/res layer (11 bits):

    bit 0:      active / not active
    bit 1-9:    raw filter count (filter count = max(16, raw filter count))
    bit 9-11:   raw kernel size (kernel size = 2 + raw kernel size)

    gene for dense layer (11 bits):

    bit 0:      active / not active
    bit 1-9:    raw units count (units count = max(16, raw units count))
    bit 9-11:   raw dropout (dropout = raw dropout * 0.1)
    """

    BITS_PER_METADATA_LAYER = 6
    METADATA_LAYER_MASK = (1 << BITS_PER_METADATA_LAYER) - 1

    LR_MASK = (1 << 2) - 1
    MOMENTUM_MASK = ((1 << 4) - 1) & ~LR_MASK
    BATCH_SIZE_MASK = (((1 << 6) - 1) & ~MOMENTUM_MASK) & ~LR_MASK

    BITS_PER_C_LAYER = 11
    C_LAYER_MASK = (1 << BITS_PER_C_LAYER) - 1

    C_ACTIVE_MASK = 1
    C_FILTER_MASK = ((1 << 9) - 1) & ~C_ACTIVE_MASK
    C_KER_SIZE_MASK = (((1 << 11) - 1) & ~C_FILTER_MASK) & ~C_ACTIVE_MASK

    BITS_PER_D_LAYER = 11
    D_LAYER_MASK = (1 << BITS_PER_D_LAYER) - 1

    D_ACTIVE_MASK = 1
    D_UNIT_MASK = ((1 << 9) - 1) & ~D_ACTIVE_MASK
    D_DROPOUT_MASK = (((1 << 11) - 1) & ~D_UNIT_MASK) & ~D_ACTIVE_MASK

    def __init__(self, nc=6, 
                        nd=6,
                        use_res=True,

                        epoch_cnt=30,
                        population_cnt=16,

                        elite_cnt=2,
                        selection_luck=0.1,

                        mutation_log_p=3,
                        mutation_p_is_complement=False,

                        crossover_p = 0.7,
                        crossover_point_cnt = 3,
                        ):
        """(Most of documentation is inside init)\n
            Some observations:
            * mutation_log_p can be an int or a dict {epoch_idx: value}
            * crossover_p can be an int or a dict {epoch_idx: value}
            * selection_luck can be an int or a dict {epoch_idx: value}"""

        def _todict(x):
            return {e: x for e in range(epoch_cnt)} if type(x) is int else x

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

        self.bits_pe_chrom = Evolvable.BITS_PER_C_LAYER * self.nc + \
                                Evolvable.BITS_PER_D_LAYER * self.nd + \
                                Evolvable.BITS_PER_METADATA_LAYER * 1

        self.max_chrom_value = 1 << self.bits_pe_chrom
        """chromosome with all bits set to 1"""

        self.state = EvolState.UNINITIALIZED
        self.learner: KeystrokeFingerprintClassificator = None
        """associated model from learner.py"""

        self.elite_cnt = elite_cnt
        """first elite_cnt individuals have a copy that bypasses mutation and crossover"""

        self.selection_luck = _todict(selection_luck)
        """randomly select a percentage 
            of the self.population_cnt to go to the next generation,
            without any regards to fitness score"""

        self.mutation_p_is_complement = mutation_p_is_complement
        self.mutation_log_p = _todict(mutation_log_p)
        """if mutation_p_is_complement: 
                mutation probability for an arbitrary bit = 1 - (1 / 1 << mutation_log_p)
            else:
                mutation probability for an arbitrary bit = 1 / 1 << mutation_log_p
                """

        self.crossover_p = _todict(crossover_p)
        """crossover probability for a chromosome"""

        self.crossover_point_cnt = crossover_point_cnt
        """number of crossover points"""

        """ONLY FOR INTERNAL USE"""

        self._elites = []
        """field to store (temporary) elite individuals"""

    def init_learner(self, seed=None, **kwargs):
        
        self.learner = KeystrokeFingerprintClassificator(**kwargs)

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

        def _decode_c_gene(gene):
            return max(16, (gene & Evolvable.C_FILTER_MASK) >> 1), \
                    ((gene & Evolvable.C_KER_SIZE_MASK) >> 9) + 2

        def _decode_d_gene(gene):
            return max(16, (gene & Evolvable.D_UNIT_MASK) >> 1), \
                ((gene & Evolvable.D_DROPOUT_MASK) >> 9) * 0.1

        model = Sequential()
        """obs: the layers will be considered in the following order:
                * the first (LSB) 11 bits will represent layer 1 conv/res
                ...
                * the last (MSB) 11 bits will represent layer 1 dense"""

        model.add(InputLayer(input_shape = (self.learner.timeslice_len, 1)))

        for _ in range(self.nc):
            
            gene = chromosome & Evolvable.C_LAYER_MASK
            chromosome >>= Evolvable.BITS_PER_C_LAYER

            if gene & Evolvable.C_ACTIVE_MASK:

                filter_cnt, ker_size = _decode_c_gene(gene)
                
                if self.use_res:
                    model.add(Res1D(filters=filter_cnt, kernel_size=ker_size))
                else:
                    model.add(Conv1D(filters=filter_cnt, kernel_size=ker_size))
                
                model.add(BatchNormalization())
                model.add(ReLU())

        model.add(Flatten())

        for _ in range(self.nd):
            
            gene = chromosome & Evolvable.D_LAYER_MASK
            chromosome >>= Evolvable.BITS_PER_D_LAYER

            if gene & Evolvable.D_ACTIVE_MASK:

                unit_cnt, dropout = _decode_d_gene(gene)
                
                model.add(Dense(unit_cnt))
                model.add(ReLU())

                if dropout > 0:
                    model.add(Dropout(dropout))

        model.add(Dense(2))
        model.add(Softmax())

        return model

    def fitness(self):
        """perform fitness function evaluation
            on all self.population which have associated score == 'None'"""

        def _decode_metadata_gene(gene):
            return 10 ** -((gene & Evolvable.LR_MASK) + 2), \
                    0.6 + 0.1 * ((gene & Evolvable.MOMENTUM_MASK) >> 2), \
                    1 << (4 + ((gene & Evolvable.BATCH_SIZE_MASK) >> 4))
        
        for idx in range(self.population_cnt):

            if self.population[idx][1] is not None:
                continue

            try:

                lr, momentum, batch_size = _decode_metadata_gene(self.population[idx][0])

                nn = self.build_nn(self.population[idx][0])

                self.learner.model = nn
                self.learner.model.compile(optimizer = SGD(lr, momentum), 
                                            loss = CategoricalCrossentropy(), 
                                            metrics = ['accuracy'])

                self.learner.model_state = ModelState.UNTRAINED
                self.learner.train_model(save_model_name=None, display_history=False,
                                            epochs=self.train_epoch_cnt,
                                            batch_size=batch_size)

                validation_results = self.learner.validate(save_model_name = "evolved_best_model")
                self.population[idx][1] = validation_results["accuracy"]

            except Exception as err:
                print(f"error while training or validating model: {err}")
                self.population[idx][1] = 0.0

    def selection(self, epoch):
        """perform selection on self.population
        heuristic:
        * elitism applied on the first self.elite_cnt inidividuals
        * the first third of the population is kept
        * a few remaining individuals are randomly selected and kept 
            (self.selection luck parameter)"""

        self.fitness()
        self.population.sort(key = lambda x: -1 if x[1] is None else x[1])

        print(f"[*] best from current population: {self.population[-1][1]}")

        self._elites = deepcopy(self.population[-2:])

        new_population = self.population[self.population_cnt * 2 // 3:]

        lucky_cnt = min(int(self.population_cnt * self.selection_luck[epoch]), 
                            self.population_cnt - len(self.population))
        
        for _ in range(lucky_cnt):
            new_population.append(random.choice(self.population))

        # rest of population cnt will be replinished at crossover
        self.population = new_population

    def evolve(self, train_epoch_cnt = 9):
        """Start the evolution process"""

        self.train_epoch_cnt = train_epoch_cnt

        assert(self.state is EvolState.READY)

        try:

            self.init_population()

            for ep in range(self.epoch_cnt):

                assert(len(self.population) == self.population_cnt)

                print(f"[i] ================== Generation {ep} ==================")
                
                self.selection(ep)
                self.mutate(ep)
                self.crossover(ep)
                self.refill(ep)

            self.fitness()

        except Exception as err:

            print(f"[!] exception when evolving: {err}")

            with open(f"BACKUP_CHROMOSOMES_{time.time()}.txt", "w+") as backup:
                print(self.population, file=backup, flush=True)
        
    def mutate(self, epoch):
        """perform mutation on all self.population\n
            in an attempt to make mutation faster,
            mutation is split in two phases:

            * mutation of some 0 bits to 1: 
                generate (mutation_log_p + 1) random chromosomes
                'and' them together, 'or' the chromosome with the result
                
            * mutation of some 1 bits to 0: 
                generate (mutation_log_p + 1) random chromosomes
                'and' them together, 'and not' the chromosome with the result"""

        for idx in range(len(self.population)):

            mask_0to1 = random.randint(0, self.max_chrom_value)
            mask_1to0 = random.randint(0, self.max_chrom_value)

            for _ in range(self.mutation_log_p[epoch] + 1):

                mask_0to1 &= random.randint(0, self.max_chrom_value)
                mask_1to0 &= random.randint(0, self.max_chrom_value)

            if self.mutation_p_is_complement:

                mask_0to1 = ~mask_0to1
                mask_1to0 = ~mask_1to0

            self.population[idx][0] |= mask_0to1
            self.population[idx][0] &= ~mask_1to0

            self.population[idx][1] = None

    def crossover(self, epoch):
        """perform crossover on all self.population"""

        for idx in range(len(self.population)):

            waiting_crossover = None

            do_crossover = True if random.random() < self.crossover_p[epoch] else False
            if do_crossover:

                if waiting_crossover is None:
                    waiting_crossover = idx

                else:

                    fst = self.population[idx][0]
                    snd = self.population[waiting_crossover][0]

                    for _ in range(self.crossover_point_cnt):

                        point = random.randint(1, self.bits_pe_chrom - 1)
                        mask = (1 << point) - 1

                        tmp = fst

                        fst = (mask & fst) | (~mask & snd)
                        snd = (~mask & tmp) | (mask & snd)
                    
                    self.population[idx][0] = fst
                    self.population[waiting_crossover][0] = snd

                    self.population[idx][1] = None
                    self.population[waiting_crossover][1] = None

                    waiting_crossover = None

    def refill(self, epoch):
        """method that re-integrates elites
            and refills self.population with random new chromosomes"""

        self.population.extend(self._elites)

        for c in self.random_chromosome(self.population_cnt - len(self.population)):
            self.population.append([c, None])
