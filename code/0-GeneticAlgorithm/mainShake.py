from src.organism import Organism
from src.evaluator import Evaluator
from src.geneticalgorithm import GeneticAlgorithm

from random import random, randrange, choice
import string
import math
import time
from typing import List

target = "to be or not to be"

class shakeOrg(Organism):
    def __init__(self, random = True):
        self.DNA: List[str] = []

        super().__init__(random)

    def rndInit(self):
        self.DNA = [choice(string.ascii_lowercase + " ") for _ in range(len(target))]

    def mutate(self, mutationrate: float):
        for i in range(len(self.DNA)):
            if random() < mutationrate:
                self.DNA[i] = choice(string.ascii_lowercase + " ")


    def crossover(self, spouse: 'shakeOrg') -> 'shakeOrg':
        child = shakeOrg(False)
        midpoint = randrange(0, len(self.DNA))
        child.DNA.extend(spouse.DNA[:midpoint])
        child.DNA.extend(self.DNA[midpoint:])
        child.setFitness(-1)
        return child

    def __repr__(self):
        return "".join(self.DNA)

class shakeEval(Evaluator):
    def __init__(self, ):
        super().__init__()

    def evalMulti(self, pop: List[shakeOrg]) -> float:
        avg_fit = 0

        for i in range(len(pop)):
            if pop[i].getFitness() == -1:
                fit = 0
                for cr in range(len(pop[i].DNA)):
                    if pop[i].DNA[cr] == target[cr]:
                        fit += 1
                fit = fit / len(target)
                pop[i].setFitness(fit)
            else:
                fit = pop[i].getFitness()
            avg_fit += fit

        avg_fit = avg_fit / len(pop)
        return avg_fit


def main():
    eval = shakeEval()
    GA = GeneticAlgorithm(shakeOrg, eval, populationSize=200, eliteSize=0, useTournament=False, tournamentSize=2, cutoffSize=100, mutationRate=0.01)
    GA.initGenerator()

    starttime = time.perf_counter()
    bfit = math.inf
    best_org = None

    while bfit != 1:
        gen, avg, best_org = GA.nextGeneration()

        bfit = best_org.getFitness()

        if gen % 1 == 0:
            elapsed = time.perf_counter() - starttime
            print("Generation {}: best_fit {}, avg_fit {}, elapsed {:0.4f}".format(gen, best_org.getFitness(), avg, elapsed))
            print("Best Organism: " + str(best_org))
            print("")
    print(best_org, GA.getGen())


if __name__ == '__main__':
    main()

