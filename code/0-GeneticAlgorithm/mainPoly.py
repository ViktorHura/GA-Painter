import copy

from src.organism import Organism
from src.evaluator import Evaluator
from src.geneticalgorithm import GeneticAlgorithm

from dataclasses import dataclass
from random import random, randrange
import datetime
import os
import time
from typing import List
import cv2
import math
import numpy as np

imageW = 250
imageH = 300
polygons = 50
vertices = 3
outputDir = "output"

@dataclass
class Gene:
    gray: float
    alpha: float
    points: List[List[int]]

    @classmethod
    def randomInit(cls) -> 'Gene':
        pts: List[List[int]] = []
        for i in range(vertices):
            pts.append([randrange(0, imageW), randrange(0, imageH)])
        return Gene(gray=random(), alpha=random(), points=pts)


class polyOrg(Organism):
    def __init__(self, random = True):
        self.width = imageW
        self.height = imageH
        self.DNA: List[Gene] = []

        super().__init__(random)

    def rndInit(self):
        for i in range(polygons):
            self.DNA.append(Gene.randomInit())

    def mutate(self, mutationrate: float):
        for i in range(len(self.DNA)):
            if random() < mutationrate:
                choice = randrange(0, 5)
                if choice == 0 or choice == 4:
                    self.DNA[i].gray = random()
                elif choice == 1:
                    j = randrange(0, len(self.DNA))
                    self.DNA[i], self.DNA[j] = self.DNA[j], self.DNA[i]
                elif choice == 2:
                    j = randrange(0, vertices)
                    pnt = [randrange(0, imageW), randrange(0, imageH)]
                    self.DNA[i].points[j] = pnt
                elif choice == 3:
                    self.DNA[i].alpha = random()

    def crossover(self, spouse: 'polyOrg') -> 'polyOrg':
        child = polyOrg(False)
        midpoint = randrange(0, len(self.DNA))
        child.DNA.extend(copy.deepcopy(spouse.DNA[:midpoint]))
        child.DNA.extend(copy.deepcopy(self.DNA[midpoint:]))
        child.setFitness(-1)
        return child

    def draw(self):
        image = np.zeros((self.height, self.width))
        ctrs = np.array([[0,0],[0, self.height-1], [self.width-1, self.height-1], [self.width-1, 0]])
        cv2.fillPoly(image, pts=[ctrs], color=(255, 255, 255))

        for gene in self.DNA:
            overlay = image.copy()
            col = gene.gray * 255
            contours = np.array(gene.points)
            cv2.fillPoly(overlay, pts=[contours], color=(col, col, col))
            # apply the overlay
            cv2.addWeighted(overlay, gene.alpha, image, 1 - gene.alpha, 0, image)

        return image

    def save(self, gennum):
        a = self.draw()
        cv2.imwrite(os.path.join(outputDir, "{}.png".format(gennum)), a)

class polyEval(Evaluator):
    def __init__(self, imagePath):
        a = cv2.imread(imagePath)
        self.img = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        super().__init__()

    def evalMulti(self, pop: List[polyOrg]) -> float:
        avg_fit = 0

        for i in range(len(pop)):
            if pop[i].getFitness() == -1:
                img = pop[i].draw()
                diff = self.img - img
                a = np.abs(diff)
                c = a / 255
                fit = np.sum(c) / (imageW*imageH)
                fit = 1-fit
                pop[i].setFitness(fit)
            else:
                fit = pop[i].getFitness()
            avg_fit += fit

        avg_fit = avg_fit / len(pop)
        return avg_fit


def main():

    eval = polyEval("images/mona.jpg")
    GA = GeneticAlgorithm(polyOrg, eval, populationSize=1280, cutoffSize=250, eliteSize=1, tournamentSize=30, mutationRate=0.02)
    GA.initGenerator()

    now = datetime.datetime.now().strftime("%d-%m-%H-%M_%S")
    global outputDir
    outputDir = os.path.join(os.getcwd(), outputDir, "out-" + now)
    os.mkdir(outputDir)

    starttime = time.perf_counter()
    bfit = 0

    while bfit < 0.99:
        if GA.getGen() % 1 == 0:
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

