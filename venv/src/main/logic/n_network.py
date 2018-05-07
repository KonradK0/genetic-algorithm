from main.logic import genotype
from main.logic import genotype_interperter
from math import exp
from random import randint
from operator import xor
import numpy as np


class NNetwork:

    def __init__(self, size, mutation_rate=0.9, beta=0.2):
        self.size = size
        self.mutation_rate = mutation_rate
        self.genotypes = np.array([genotype.Genotype() for i in range(self.size)])
        self.inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.current_learning_rate = 1
        self.current_best_genotype = self.genotypes[0]
        self.beta = beta

    def calc_fitness_function(self):
        for genotype_var in self.genotypes:
            for input in self.inputs:
                curr_fit_indicator = abs(float(xor(input[0], input[1])) - self.calc_out(input, 1, genotype_var))
                # print ("curr fit indicator " + str(curr_fit_indicator))
                if curr_fit_indicator > genotype_var.fitness_indicator:
                    genotype_var.fitness_indicator = curr_fit_indicator
            if genotype_var.fitness_indicator < self.current_learning_rate:
                print("Changing best fit from " + str(self.current_learning_rate) + " to " + str(
                    genotype_var.fitness_indicator))
                self.current_learning_rate = genotype_var.fitness_indicator
                self.current_best_genotype = genotype_var

    def calc_first_layer_out(self, input, bias, first_neuron_weights, second_neuron_weights):
        return np.array([sigmoid(np.dot(np.append(input, bias), first_neuron_weights), self.beta),
                         sigmoid(np.dot(np.append(input, bias), second_neuron_weights), self.beta)])

    def calc_out(self, input, bias, genotype_var):
        all_weights = genotype_interperter.get_all_weights(genotype_var)
        first_layer_out = self.calc_first_layer_out(input, bias, all_weights[0:3], all_weights[3:6])
        second_layer_out = sigmoid(np.dot(np.append(first_layer_out, bias), all_weights[6:9]), self.beta)
        return second_layer_out

    def selection(self):
        # for genotype_var in self.genotypes:
        #     print(genotype_var.fitness_indicator)
        # print()
        self.genotypes = self.new_generation()
        # for genotype_var in self.genotypes:
        #     print(genotype_var.fitness_indicator)
        # print()

    def new_generation(self):
        new_generation = []
        while np.size(new_generation) < self.size / 2:
            first_index = randint(0, self.genotypes.size - 1)
            second_index = randint(0, self.genotypes.size - 1)
            while first_index == second_index:
                second_index = randint(0, self.genotypes.size - 1)
            new_generation.append(self.genotypes[first_index].fight(self.genotypes[second_index]))
            self.genotypes = np.delete(self.genotypes, (first_index, second_index))
        return np.array(new_generation)

    def cross_population(self):
        # print ("Before crossing")
        # for genotype_var in self.genotypes:
        #     print (genotype_var.genes)
        #     print()
        new_generation = self.genotypes
        while new_generation.size < self.size:
            first_index = second_index = 0
            while first_index == second_index:
                first_index = randint(0, self.genotypes.size - 1)
                second_index = randint(0, self.genotypes.size - 1)
            new_generation = np.append(new_generation, self.genotypes[first_index].cross(self.genotypes[second_index]))
        self.genotypes = new_generation
        # print ("After crossing")
        # for genotype_var in self.genotypes:
        #     print (genotype_var.genes)
        #     print()

    def mutate_population(self):
        i = 0
        for genotype_var in self.genotypes:
            # print("Mutating genotype number: " + str(i))
            i += 1
            if genotype_var == self.current_best_genotype:
                # print("Mutating best genotype")
                pass
            genotype_var.mutate(self.mutation_rate)

    def cycle(self):
        self.calc_fitness_function()
        self.selection()
        self.cross_population()
        self.mutate_population()

    def find_fitting_genotype(self):
        learning_threshold = 0.5
        i = 0
        while self.current_learning_rate > learning_threshold:
            self.cycle()
            # print(genotype_interperter.get_all_weights(self.current_best_genotype))
            print("Best fit indicator in cycle number " + str(i) + ": " + str(
                self.current_best_genotype.get_fitness_indicator()))
            i += 1
        self.print_results()

    def print_results(self):
        print(self.current_best_genotype.get_fitness_indicator())
        print(genotype_interperter.get_all_weights(self.current_best_genotype))
        print("In: [0,0], ExpectedOut: 0, ActualOut: " + str(
            abs(self.calc_out([0, 0], 1, self.current_best_genotype))))
        print("In: [0,1], ExpectedOut: 1, ActualOut: " + str(
            abs(self.calc_out([0, 1], 1, self.current_best_genotype))))
        print("In: [1,0], ExpectedOut: 1, ActualOut: " + str(
            abs(self.calc_out([1, 0], 1, self.current_best_genotype))))
        print("In: [1,1], ExpectedOut: 0, ActualOut: " + str(
            abs(self.calc_out([1, 1], 1, self.current_best_genotype))))


def sigmoid(x, beta):
    return 1 / (1 + exp(beta * -x))


uut = NNetwork(1000, 0.6)
uut.find_fitting_genotype()
# size_before = uut.genotypes.size
# uut.cycle()
# size_after = uut.genotypes.size
# assert size_after == size_before
