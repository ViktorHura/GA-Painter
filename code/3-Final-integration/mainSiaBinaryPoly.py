import copy
import math

from src.organism import Organism
from src.evaluator import Evaluator
from src.geneticalgorithm import GeneticAlgorithm
from integration.siamese.main import SiameseNetwork
from PIL import Image

from dataclasses import dataclass
from random import random, randrange, choice, getstate, setstate
import datetime
import os
import time
from typing import List
import cv2
import torch
import torch.nn.functional as F
import numpy as np

imageW = 28
imageH = 28
polygons = 5
vertices = 3
outputDir = "output"

@dataclass
class Gene:
    gray: bool
    points: List[List[int]]

    @classmethod
    def randomInit(cls) -> 'Gene':
        pts: List[List[int]] = []
        for i in range(vertices):
            pts.append([randrange(0, imageW), randrange(0, imageH)])
        return Gene(gray=True, points=pts)


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

                #choice = 0
                choice = randrange(0, 2)
                if choice == 0:
                    j = randrange(0, vertices)
                    pnt = [randrange(0, imageW), randrange(0, imageH)]
                    self.DNA[i].points[j] = pnt
                elif choice == 1:
                    self.DNA[i].gray = not self.DNA[i].gray

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
        cv2.fillPoly(image, pts=[ctrs], color=(0, 0, 0))

        for gene in self.DNA:
            overlay = image.copy()
            col = 255 if gene.gray else 0
            contours = np.array(gene.points)
            cv2.fillPoly(overlay, pts=[contours], color=(col, col, col))
            # apply the overlay
            cv2.addWeighted(overlay, 1, image, 0, 0, image)

        return image

    def save(self, gennum):
        a = self.draw()
        cv2.imwrite(os.path.join(outputDir, "{}.png".format(gennum)), a)

class polyEval(Evaluator):
    def __init__(self):
        self.model = SiameseNetwork()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load("siamese/models/siameseBoot.pth"))
        self.model.eval()
        self.model = self.model.to(self.device)

        # img2 = Image.open("integration/siamese/0/136.png")
        # img2 = Image.open("integration/siamese/1/37.png")
        # img2 = Image.open("integration/siamese/2/225.png")
        # img2 = Image.open("integration/siamese/3/1683.png")
        # img2 = Image.open("integration/siamese/4/27.png")
        # img2 = Image.open("integration/siamese/5/751.png")
        # img2 = Image.open("integration/siamese/6/439.png")
        # img2 = Image.open("integration/siamese/7/550.png")
        # img2 = Image.open("integration/siamese/8/1319.png")
        # img2 = Image.open("integration/siamese/9/681.png")

        img2 = Image.open("siamese/boots/3745.png")
        img2 = img2.convert("L")
        img2 = img2.resize((100, 100))
        self.imog = np.array(img2, dtype=np.float32)
        img2 = torch.tensor(data=np.array(img2, dtype=np.float32))

        img2.unsqueeze_(0)
        img2 = img2.repeat(1, 1, 1)

        self.img2 = img2.to(self.device)

        super().__init__()

    def save(self):
        cv2.imwrite(os.path.join(outputDir, "{}.png".format("ref")), self.imog)

    def evalMulti(self, pop: List[polyOrg]) -> float:
        avg_fit = 0

        for i in range(len(pop)):
            if pop[i].getFitness() == -1:
                img = pop[i].draw()

                img = cv2.resize(img, (100, 100))

                img1 = torch.tensor(data=np.array(img, dtype=np.float32) )

                img1.unsqueeze_(0)
                img1 = img1.repeat(1, 1, 1)

                img1 = img1.to(self.device)

                with torch.no_grad():
                    output1, output2 = self.model(img1, self.img2)
                    euclidean_distance = F.pairwise_distance(output1, output2)
                    fit = float(euclidean_distance.item())
                    fit = -fit

                pop[i].setFitness(fit)
            else:
                fit = pop[i].getFitness()
            avg_fit += fit

        avg_fit = avg_fit / len(pop)
        return avg_fit


def main():
    eval = polyEval()
    GA = GeneticAlgorithm(polyOrg, eval, populationSize=1024, cutoffSize=200, eliteSize=1, tournamentSize=24, mutationRate=0.02)
    GA.initGenerator()

    now = datetime.datetime.now().strftime("%d-%m-%H-%M_%S")
    global outputDir
    outputDir = os.path.join(os.getcwd(), outputDir, "out-" + now)
    os.mkdir(outputDir)

    eval.save()

    starttime = time.perf_counter()
    bfit = -math.inf

    with open(os.path.join(outputDir, "{}.txt".format("results")), 'a') as f:

        f.write(GA.getConfig() + "\n")

        while bfit < 0.1:
            if GA.getGen() % 1 == 0:
                gen, avg, best_org = GA.nextGeneration(True)
                elapsed = time.perf_counter() - starttime
                log = "{}: best_fit {}, avg_fit {}, elapsed {:0.4f}".format(gen, best_org.getFitness(), avg, elapsed)
                f.write(log + "\n")
                print(log)
                if best_org.getFitness() > bfit:
                    best_org.save(gen)
                    bfit = best_org.getFitness()
            else:
                _, _, _ = GA.nextGeneration(False)

        gen, avg, best_org = GA.nextGeneration(True)
        log = "{}: best_fit {}, avg_fit {}, elapsed {:0.4f}".format(gen, best_org.getFitness(), avg, elapsed)
        f.write(log + "\n")
        print(log)

    f.close()

if __name__ == '__main__':
    main()

