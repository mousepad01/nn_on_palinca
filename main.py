from learner import HumanVsRnd
from evol import Evolvable

if __name__ == "__main__":

    evol_nn = Evolvable(epoch_cnt=8)
    evol_nn.init_learner(seed=123)

    evol_nn.evolve()
