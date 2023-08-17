# Mixed-Curvature Pathway Representation Learning
Mixed-curvature graph representation learning for biological pathways.
Most files are derived from and retain the commit history of [this repository](https://github.com/HazyResearch/hyperbolics) that provides hyperbolic embedding implementations of [Representation Tradeoffs for Hyperbolic Embeddings](https://arxiv.org/pdf/1804.03329.pdf) + product embedding implementations of [Learning Mixed-Curvature Representations in Product Spaces](https://openreview.net/pdf?id=HJxeWnCcF7)

The biological pathway analyses are presented in this workshop extended abstract:  
[Mixed-Curvature Representation Learning for Biological Pathway Graphs](https://icml-compbio.github.io/2023/papers/WCBICML2023_paper117.pdf)  
Daniel McNeela, Frederic Sala<sup>+</sup>, Anthony Gitter<sup>+</sup>.  
2023 ICML Workshop on Computational Biology.

<sup>+</sup> Equal contribution

### Pytorch optimizer
`python pytorch/pytorch_hyperbolic.py learn --help` to see options. Optimizer requires torch >=0.4.1. Example usage:

```
python pytorch/pytorch_hyperbolic.py learn data/edges/phylo_tree.edges --batch-size 64 --dim 10 -l 5.0 --epochs 100 --checkpoint-freq 10 --subsample 16
```

Products of hyperbolic spaces with Euclidean and spherical spaces are also supported. E.g. adding flags `-euc 1 -edim 20 -sph 2 -sdim 10` embeds into a product of Euclidean space of dimension 20 with two copies of spherical space of dimension 10.

## License
The code is available under the [Apache License 2.0](LICENSE).
Most of the source code is derived from the unlicensed [hyperbolics repository](https://github.com/HazyResearch/hyperbolics), and the [contributors](https://github.com/HazyResearch/hyperbolics/graphs/contributors) to that repository have been added to the license copyright.
