class Organism:
    def __init__(self, random: bool = True):
        self.fitness: float = -1
        if random:
            self.rndInit()

    def getFitness(self) -> float:
        return self.fitness

    def setFitness(self, newFit: float):
        self.fitness = newFit

    def rndInit(self):
        raise NotImplementedError()

    def mutate(self, mutationrate: float):
        raise NotImplementedError()

    def crossover(self, spouse: 'Organism') -> 'Organism':
        raise NotImplementedError()