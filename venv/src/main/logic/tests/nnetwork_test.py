import unittest
from main.logic import n_network
from main.logic import genotype
import numpy as np
from mockito import when, unstub


class TestNNetwork(unittest.TestCase):

    def test_calc_fitness_func(self):
        uut = n_network.NNetwork(5, 0.5, 0.2)
        uut.calc_fitness_function()

    def test_calc_first_layer_out(self):
        uut = n_network.NNetwork(1, 0.5, 0.2)
        when(genotype.Genotype).init_genes().thenReturn(np.ones(54, dtype=str))
        out = uut.calc_first_layer_out([0, 0], 1, [0, 0, 0], [0, 0, 0])
        expected = np.array([0.5, 0.5])
        unstub(genotype.Genotype)
        assert np.array_equal(expected, out)

    def test_calc_out(self):
        uut = n_network.NNetwork(1, 0.5, 0.2)
        when(genotype.Genotype).init_genes().thenReturn(np.zeros(54, dtype=str))
        genotype_var = genotype.Genotype()
        input = [0, 0]
        expected = 0.5
        assert uut.calc_out(input, 1, genotype_var) == expected
        unstub(genotype.Genotype)

    def test_cross_population_size(self):
        uut = n_network.NNetwork(4, 0.5, 0.2)
        uut.genotypes = uut.genotypes[0:2]
        uut.cross_population()
        assert uut.genotypes.size == 4

    def test_selection(self):
        uut = n_network.NNetwork(2, 1, 0.2)
        first_contender = uut.genotypes[0]
        second_contender = uut.genotypes[1]
        when(first_contender).get_fitness_indicator().thenReturn(0.5)
        when(second_contender).get_fitness_indicator().thenReturn(0.7)
        uut.selection()
        unstub(first_contender)
        unstub(second_contender)
        assert np.array_equal(first_contender.genes, uut.genotypes[0].genes)
        assert np.size(uut.genotypes) == 1

    # def cycle_test(self):
    #     uut = n_network.NNetwork(4, 0.5, 0.2)
    #     size_before = uut.genotypes.size
    #     uut.cycle()
    #     size_after = uut.genotypes.size
    #     assert size_after == size_before



if __name__ == '__main__':
    unittest.main()
