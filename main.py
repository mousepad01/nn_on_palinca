from evol import Evolvable

if __name__ == "__main__":

    epoch_cnt = 18

    mutation_log_p = {e: 2 for e in range(epoch_cnt)}
    mutation_log_p.update({e: 3 for e in range(5, epoch_cnt)})
    mutation_log_p.update({e: 4 for e in range(12, epoch_cnt)})
    mutation_log_p.update({e: 2 for e in range(16, epoch_cnt)})

    crossover_p = {e: 0.9 for e in range(epoch_cnt)} 
    crossover_p.update({e: 0.8 for e in range(5, epoch_cnt)}) 
    crossover_p.update({e: 0.6 for e in range(12, epoch_cnt)}) 
    crossover_p.update({e: 0.3 for e in range(16, epoch_cnt)})

    evol_nn = Evolvable(epoch_cnt=epoch_cnt, 
                        population_cnt=8,
                        nc=12, nd=8, 
                        selection_luck=0,
                        mutation_log_p=mutation_log_p,
                        crossover_p=crossover_p)

    evol_nn.init_learner(seed=1234)

    evol_nn.evolve(15)
