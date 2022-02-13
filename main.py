from learner import HumanVsRnd
from evol import Evolvable

if __name__ == "__main__":

    evol_nn = Evolvable(epoch_cnt=20, population_cnt=8,
                        nc=10, selection_luck=0)
    evol_nn.init_learner(seed=1234)

    evol_nn.evolve(15)
