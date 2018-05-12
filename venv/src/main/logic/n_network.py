from main.logic import genotype
from main.logic import genotype_interperter
from math import exp
from random import randint
from operator import xor
import numpy as np


class NNetwork:

    def __init__(self, size, mutation_rate=0.9, beta=1):
        self.size = size
        self.mutation_rate = mutation_rate
        self.genotypes = np.array([genotype.Genotype() for i in range(self.size)])
        self.inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.current_best_genotype = self.genotypes[0]
        self.beta = beta

    def calc_fitness_function(self):
        for genotype_var in self.genotypes:
            for input in self.inputs:
                curr_fit_indicator = abs(float(xor(input[0], input[1])) - self.calc_out(input, 1, genotype_var))
                # print ("curr fit indicator " + str(curr_fit_indicator))
                if curr_fit_indicator > genotype_var.get_fitness_indicator():
                    genotype_var.fitness_indicator = curr_fit_indicator
            if genotype_var.get_fitness_indicator() < self.current_best_genotype.get_fitness_indicator():
                print(
                    "Changing best fit from " + str(self.current_best_genotype.get_fitness_indicator()) + " to " + str(
                        genotype_var.fitness_indicator))
                self.current_best_genotype = genotype_var

    def calc_first_layer_out(self, input, bias, first_neuron_weights, second_neuron_weights):
        return np.array([sigmoid(np.dot((np.append(input, bias)), first_neuron_weights), self.beta),
                         sigmoid(np.dot((np.append(input, bias)), second_neuron_weights), self.beta)])

    def calc_out(self, input, bias, genotype_var):
        all_weights = genotype_interperter.get_all_weights(genotype_var)
        first_layer_out = self.calc_first_layer_out(input, bias, all_weights[0:3], all_weights[3:6])
        second_layer_out = sigmoid(np.dot(np.append(first_layer_out, bias), all_weights[6:9]), self.beta)
        return second_layer_out

    def selection(self):
        self.genotypes = self.new_generation()

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
        new_generation = self.genotypes
        while new_generation.size < self.size:
            first_index = second_index = 0
            while first_index == second_index:
                first_index = randint(0, self.genotypes.size - 1)
                second_index = randint(0, self.genotypes.size - 1)
            new_generation = np.append(new_generation, self.genotypes[first_index].cross(self.genotypes[second_index]))
        self.genotypes = new_generation

    def mutate_population(self):
        i = 0
        for genotype_var in self.genotypes:
            i += 1
            if genotype_var == self.current_best_genotype:
                pass
            genotype_var.mutate(self.mutation_rate)

    def cycle(self):
        self.calc_fitness_function()
        self.selection()
        self.cross_population()
        self.mutate_population()

    def find_fitting_genotype(self):
        learning_threshold = 0.2
        i = 0
        while True:
            self.cycle()
            # print(genotype_interperter.get_all_weights(self.current_best_genotype))
            print("Best fit indicator in cycle number " + str(i) + ": " + str(
                self.current_best_genotype.get_fitness_indicator()))
            i += 1
            if self.stop_condition():
                print("Best fit genotype on out :" + str(self.current_best_genotype.genes))
                break
        self.print_results()

    def stop_condition(self):
        cond0_0 = (self.calc_out([0, 0], 1, self.current_best_genotype)) < 0.2
        cond0_1 = (self.calc_out([0, 1], 1, self.current_best_genotype)) > 0.8
        cond1_0 = (self.calc_out([1, 0], 1, self.current_best_genotype)) > 0.8
        cond1_1 = (self.calc_out([1, 1], 1, self.current_best_genotype)) < 0.2
        return cond0_0 and cond0_1 and cond1_0 and cond1_1

    def print_results(self):
        print(self.current_best_genotype.get_fitness_indicator())
        print(genotype_interperter.get_all_weights(self.current_best_genotype))
        print("In: [0,0], ExpectedOut: 0, ActualOut: " + str(
            self.calc_out([0, 0], 1, self.current_best_genotype)))
        print("In: [0,1], ExpectedOut: 1, ActualOut: " + str(
            self.calc_out([0, 1], 1, self.current_best_genotype)))
        print("In: [1,0], ExpectedOut: 1, ActualOut: " + str(
            self.calc_out([1, 0], 1, self.current_best_genotype)))
        print("In: [1,1], ExpectedOut: 0, ActualOut: " + str(
            self.calc_out([1, 1], 1, self.current_best_genotype)))


def sigmoid(x, beta):
    return 1 / (1 + exp(beta * -x))


uut = NNetwork(1000, 0.1)
uut.find_fitting_genotype()
