import time
from copy import copy
from typing import Tuple, List
from random import randrange
import random
from src.evaluator import Evaluator
from src.organism import Organism


class GeneticAlgorithm:
    def __init__(self, organismClass: type, evaluator: Evaluator,
                 populationSize: int = 128,
                 eliteSize: int = 1,
                 useTournament: bool = True,
                 cutoffSize: int = 25,
                 tournamentSize: int = 3,
                 mutationRate: float = 0.02):
        self.organism: type = organismClass
        self.evaluator: evaluator = evaluator

        self.popSize: int = populationSize
        self.elite: int = eliteSize
        self.cutoff: int = cutoffSize
        self.useTurn = useTournament
        self.tournament: int = tournamentSize
        self.mtnRate: float = mutationRate

        self.gen: int = 0

        self.population: List[Organism] = []

    def getConfig(self) -> str:
        return "'organismClass': {}, 'populationSize': {}, 'eliteSize': {}, 'useTournament': {}, 'cutoffSize': {}, 'tournamentSize': {}, 'mutationRate': {}".format(self.organism.__name__, self.popSize, self.elite, self.useTurn, self.cutoff, self.tournament, self.mtnRate)

    def initGenerator(self, newPopulation: bool = True):
        if newPopulation or not self.population:
            self.population.clear()
            for i in range(self.popSize):
                self.population.append(self.organism())
        self.gen = 0

    def getGen(self) -> int:
        return self.gen

    def _sortPop(self):
        self.population.sort(key=lambda x: x.getFitness(), reverse=True)

    def _tournamentSelect(self) -> int:
        trn: List[int] = []
        for i in range(self.tournament):
            # index of random organism, not including the 'cutoff' worst ones
            trn.append(randrange(0, self.popSize - self.cutoff))

        best_ind = trn[0]
        best_fit = self.population[best_ind].getFitness()
        for i in range(1, self.tournament):
            i_fit = self.population[trn[i]].getFitness()
            if i_fit > best_fit:
                best_ind = trn[i]
                best_fit = i_fit

        return best_ind

    def nextGeneration(self, getOrg:bool = True) -> Tuple[int, float, Organism]:
        # evaluate the current generation
        #starttime = time.perf_counter()

        avg_fit = self.evaluator.evalMulti(self.population)

        # if self.gen % 1 == 0:
        #     print("Fitness Calc took {}".format(time.perf_counter()-starttime))

        #starttime = time.perf_counter()

        self._sortPop()  # sort by fitness

        best_org = Organism(False)
        if getOrg:
            best_org: Organism = copy(self.population[0])  # save a copy of this gen's best org

        # if self.gen % 1 == 0:
        #     print("Sort took {}".format(time.perf_counter() - starttime))

        #starttime = time.perf_counter()
        # natural selection
        next_gen: List[Organism] = []

        if self.useTurn:
            for i in range(self.popSize):
                if i < self.elite:
                    next_gen.append(self.population[i])
                else:
                    parent0: Organism = self.population[self._tournamentSelect()]
                    parent1: Organism = self.population[self._tournamentSelect()]

                    child: Organism = parent0.crossover(parent1)
                    child.mutate(self.mtnRate)

                    next_gen.append(child)
        else:
            pool = range(self.popSize - self.cutoff)
            wts = [self.population[x].getFitness() for x in pool]

            parents = random.choices(pool, weights=wts, k=(self.popSize-self.elite)*2)
            for i in range(self.popSize):
                if i < self.elite:
                    next_gen.append(self.population[i])
                else:
                    ind = (i-self.elite)*2
                    parent0: Organism = self.population[parents[ind]]
                    parent1: Organism = self.population[parents[ind+1]]

                    child: Organism = parent0.crossover(parent1)
                    child.mutate(self.mtnRate)

                    next_gen.append(child)

        self.population = next_gen
        self.gen += 1
        # if self.gen % 1 == 0:
        #     print("Next gen took {}".format(time.perf_counter() - starttime))

        return self.gen - 1, avg_fit, best_org
