from random import random
import numpy as np


class Genotype:
    length = 54

    def __init__(self, genes=None):
        if genes is None:
            genes = self.init_genes()
        self.genes = genes
        self.fitness_indicator = 0

    def get_fitness_indicator(self):
        return self.fitness_indicator

    def init_genes(self):
        return np.random.choice(['0', '1'], size=self.length, p=[0.5, 0.5])

    def get_cross_index(self):
        while True:
            cross_index = int(random() * self.length)
            if (cross_index + 1) % 6 != 0:
                return cross_index

    def cross(self, other_genotype):
        cross_index = self.get_cross_index()
        genes_first_child = np.concatenate(
            (self.genes[0: cross_index], other_genotype.genes[cross_index:other_genotype.length]))
        genes_second_child = np.concatenate((other_genotype.genes[0:cross_index], self.genes[cross_index:self.length]))
        return [Genotype(genes_first_child), Genotype(genes_second_child)]

    def get_opposite_gene(self, gene):
        if gene == '0':
            return '1'
        else:
            return '0'

    def mutate(self, mutation_rate):
        for i in range(self.length):
            if random() < mutation_rate:
                # print("Mutating a gene")
                self.genes[i] = self.get_opposite_gene(self.genes[i])

    def fight(self, opponent):
        if self.get_fitness_indicator() <= opponent.get_fitness_indicator():
            return self
        return opponent
