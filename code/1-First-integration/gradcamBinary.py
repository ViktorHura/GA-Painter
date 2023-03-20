import copy

from src.organism import Organism
from src.evaluator import Evaluator
from src.geneticalgorithm import GeneticAlgorithm
from MNist.MNistNet import MNistNet

from dataclasses import dataclass
from random import random, randrange, choice
import datetime
import os
import time
from typing import List
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

imageW = 28
imageH = 28
polygons = 5
vertices = 3
outputDir = "output"
target = 3

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

        self.heatmap = None

        super().__init__(random)

    def rndInit(self):
        for i in range(polygons):
            self.DNA.append(Gene.randomInit())

    def mutate(self, mutationrate: float):
        for i in range(len(self.DNA)):
            if random() < mutationrate:
                #choice = randrange(0, 1)
                choice = 0
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

        gcam = self.heatmap.numpy()
        image = a.squeeze()
        stacked_img = np.stack((image,) * 3, axis=-1)
        cmap = cv2.resize(gcam, (28, 28), 0, 0, interpolation=cv2.INTER_LINEAR)

        plt.subplot(2,2,1)
        sh = sns.heatmap(cmap, cmap=matplotlib.pyplot.cm.jet_r, annot=False, linewidths=.5, alpha=0.6, zorder=2, xticklabels=False, yticklabels=False)
        sh = sh.imshow(stacked_img,
                    aspect=sh.get_aspect(),
                    extent=sh.get_xlim() + sh.get_ylim(),
                    zorder=1)  # put the map under the heatmap

        plt.subplot(2, 2, 2)
        sh2 = sns.heatmap(gcam, cmap=matplotlib.pyplot.cm.jet_r, annot=False, linewidths=.5, alpha=1, zorder=2, xticklabels=False, yticklabels=False)

        plt.savefig(os.path.join(outputDir, "{}-gcam.png".format(gennum)), bbox_inches = 'tight')
        #plt.show()


class polyEval(Evaluator):
    def __init__(self):
        self.model = MNistNet(grad_cam_layer='conv2')
        self.model.load_state_dict(torch.load("MNist/models/mnist_net.pth"))
        self.model.eval()


        super().__init__()

    def evalMulti(self, pop: List[polyOrg]) -> float:
        avg_fit = 0

        for i in range(len(pop)):
            if pop[i].getFitness() == -1:
                img2 = pop[i].draw()

                img2 = cv2.resize(img2, (28, 28))
                img = torch.tensor(data=img2)
                img = img.reshape(1, 1, 28, 28)
                img = img.float()

                output = self.model(img)
                pop[i].heatmap = self.model.get_heatmap(img, output, target)

                output = torch.exp(F.log_softmax(output, dim=1))
                # for j in range(10):
                #     f = float(output.data[0][j])
                #     print(str(j) + " : " + str(f))
                fit = float(output.data[0][target])
                pop[i].setFitness(fit)
            else:
                fit = pop[i].getFitness()
            avg_fit += fit

        avg_fit = avg_fit / len(pop)
        return avg_fit


def main():
    eval = polyEval()
    GA = GeneticAlgorithm(polyOrg, eval, populationSize=256, cutoffSize=50, eliteSize=1, tournamentSize=6, mutationRate=0.02)
    GA.initGenerator()

    now = datetime.datetime.now().strftime("%d-%m-%H-%M_%S")
    global outputDir
    outputDir = os.path.join(os.getcwd(), outputDir, "out-" + now)
    os.mkdir(outputDir)

    starttime = time.perf_counter()
    bfit = 0

    while bfit < 0.9999:
        if GA.getGen() % 25 == 0:
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

