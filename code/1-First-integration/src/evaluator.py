from src.organism import Organism
from typing import List


class Evaluator:
    def __init__(self):
        pass

    def evalMulti(self, pop: List[Organism]) -> float:
        raise NotImplementedError()