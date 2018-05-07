import unittest

from mockito import when, unstub
import numpy as np

from main.logic import genotype


class TestGenotype(unittest.TestCase):

    def test_init_no_args(self):
        print(genotype.Genotype().genes)

    def test_init_args(self):
        uut = genotype.Genotype(np.ones(54, dtype=str))
        assert np.array_equal(uut.genes, np.ones(uut.length, dtype=str))

    def test_get_cross_index(self):
        pass

    def test_cross(self):
        uut = genotype.Genotype(np.ones(54, dtype=str))
        other = genotype.Genotype(np.zeros(54, dtype=str))
        expected_cross_index = 54
        when(genotype.Genotype).get_cross_index().thenReturn(expected_cross_index)
        expected = genotype.Genotype(np.concatenate((np.ones(54, dtype=str), (np.zeros(0, dtype=str)))))
        crossed = uut.cross(other)
        unstub(genotype.Genotype)
        print(expected.genes)
        print(crossed.genes)
        assert np.array_equal(expected.genes, crossed.genes)

    def test_get_opposite_gene(self):
        uut = genotype.Genotype()
        assert uut.get_opposite_gene('0') == '1'
        assert uut.get_opposite_gene('1') == '0'

    def test_mutate_all_genes(self):
        when(genotype.Genotype).init_genes().thenReturn(np.ones(54, dtype=str))
        uut = genotype.Genotype()
        mutation_rate = 1
        uut.mutate(mutation_rate)
        unstub(genotype.Genotype)
        assert np.array_equal(uut.genes, np.full(54, 0, dtype=str))

    def test_mutate_no_genes(self):
        when(genotype.Genotype).init_genes().thenReturn(np.ones(54, dtype=str))
        uut = genotype.Genotype()
        mutation_rate = 0
        uut.mutate(mutation_rate)
        unstub(genotype.Genotype)
        assert np.array_equal(uut.genes, np.ones(54, dtype=str))


if __name__ == '__main__':
    unittest.main()
