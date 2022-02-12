from learner import HumanVsRnd
from evol import Evolvable

if __name__ == "__main__":

    evol_nn = Evolvable(epoch_cnt=10, population_cnt=16)
    evol_nn.init_learner(seed=1234)

    evol_nn.evolve()
