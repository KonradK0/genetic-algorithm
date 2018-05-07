import unittest
import numpy as np
from mockito import when
from mockito import unstub
from main.logic import genotype_interperter
from main.logic import genotype


class GenotypeInterpreterTest(unittest.TestCase):

    def test_split_genes(self):
        when(genotype.Genotype).init_genes().thenReturn(np.ones(54, dtype=str))
        genotype_var = genotype.Genotype()
        genes_split = genotype_interperter.split_genes(genotype_var)
        expected = np.split(np.ones(54, dtype=str), 9)
        assert np.array_equal(genes_split, expected)
        unstub(genotype.Genotype)

    def test_calculate_weight_positive(self):
        expected = 15.5
        assert genotype_interperter.calculate_weight(np.ones(6, dtype=str)) == expected

    def test_calculate_weight_negative(self):
        weight_binary = np.concatenate((np.zeros(1, str), np.ones(5, str)))
        expected = -15.5
        assert genotype_interperter.calculate_weight(weight_binary) == expected

    def test_get_all_ones_weights(self):
        when(genotype.Genotype).init_genes().thenReturn(np.ones(54,dtype=str))
        genotype_var = genotype.Genotype()
        all_weights = genotype_interperter.get_all_weights(genotype_var)
        expected = np.full(9, 15.5)
        assert np.array_equal(all_weights, expected)
        unstub(genotype.Genotype)

    def test_get_all_zeros_weights(self):
        when(genotype.Genotype).init_genes().thenReturn(np.zeros(54,dtype=str))
        genotype_var = genotype.Genotype()
        all_weights = genotype_interperter.get_all_weights(genotype_var)
        expected = np.full(9, 0)
        assert np.array_equal(all_weights, expected)
        unstub(genotype.Genotype)


if __name__ == '__main__':
    unittest.main()
