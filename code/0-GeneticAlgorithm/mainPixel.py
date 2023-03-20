import math

from src.organism import Organism
from src.evaluator import Evaluator
from src.geneticalgorithm import GeneticAlgorithm

from random import random, randrange
import datetime
import os
import time
from PIL import Image
from typing import List

imageW = 64
imageH = 64
outputDir = "output"

class pixelOrg(Organism):
    def __init__(self, random = True):
        self.width = imageW
        self.height = imageH
        self.DNA = []

        super().__init__(random)

    def rndInit(self):
        for r in range(self.height * self.width):
            self.DNA.append(randrange(0, 256))

    def mutate(self, mutationrate: float):
        for i in range(len(self.DNA)):
            if random() < mutationrate:
                self.DNA[i] = randrange(0, 256)

    def crossover(self, spouse: 'pixelOrg') -> 'pixelOrg':
        child = pixelOrg(False)
        midpoint = randrange(0, len(self.DNA))
        child.DNA.extend(spouse.DNA[:midpoint])
        child.DNA.extend(self.DNA[midpoint:])
        child.setFitness(-1)
        return child

    def similarity(self, spouse: 'pixelOrg') -> float:
        fit = 0
        for index, val in enumerate(self.DNA):
            fit += 1 - abs(val - spouse.DNA[index]) / 255
        fit = fit / len(self.DNA)
        return fit

    def __repr__(self):
        return str(self.DNA)

    def save(self, gennum):
        a = Image.new("L", (imageH, imageW))
        a.putdata(self.DNA)
        a.save(os.path.join(outputDir, "{}.png".format(gennum)))

class pixelEval(Evaluator):
    def __init__(self, imagePath):
        img = list(Image.open(imagePath).getdata())
        self.img = []
        # convert image to grayscale
        for t in img:
            avg = math.floor((float(t[0]) + float(t[1]) + float(t[2]))/float(3))
            self.img.append(avg)
        super().__init__()

    def evalMulti(self, pop: List[pixelOrg]) -> float:
        avg_fit = 0

        for i in range(len(pop)):
            if pop[i].getFitness() == -1:
                fit = 0
                for index, val in enumerate(pop[i].DNA):
                    fit += 1 - abs(val - self.img[index])/255
                fit = fit / len(pop[i].DNA)
                pop[i].setFitness(fit)
            else:
                fit = pop[i].getFitness()
            avg_fit += fit

        avg_fit = avg_fit / len(pop)
        return avg_fit

def main():
    eval = pixelEval("images/mask64.png")
    GA = GeneticAlgorithm(pixelOrg, eval, populationSize=1280, cutoffSize=0, eliteSize=1, tournamentSize=30, mutationRate=0.02)

    GA.initGenerator()

    now = datetime.datetime.now().strftime("%d-%m-%H-%M_%S")
    global outputDir
    outputDir = os.path.join(os.getcwd(), outputDir, "out-" + now)
    os.mkdir(outputDir)

    starttime = time.perf_counter()
    bfit = 0

    while bfit < 0.99:
        if GA.getGen() % 50 == 0:
            gen, avg, best_org = GA.nextGeneration(True)
            elapsed = time.perf_counter() - starttime
            print("{}: best_fit {}, avg_fit {}, elapsed {:0.4f}".format(gen, best_org.getFitness(), avg, elapsed))
            if best_org.getFitness() > bfit:
                best_org.save(gen)
                bfit = best_org.getFitness()
        else:
            _, _, _ = GA.nextGeneration(False)
    print(GA.getGen())

if __name__ == '__main__':
    main()
