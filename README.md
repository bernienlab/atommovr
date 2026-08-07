![atommovr](./figs/logo.svg)

## A simulation framework for rearrangement in atomic arrays

by [Nikhil Kiran Harle*](https://github.com/khnikhil/khnikhil), [Bo-Yu Chen*](https://phys-mattchen.github.io/), [Bob Bao](https://www.bobbaothebuilder.com/#), and [Hannes Bernien](https://bernienlab.com).

*These authors contributed equally to this work

## Key features and `where to find them`
![overview](/figs/fig1.svg)

:books: Open-source library of rearrangement algorithms:   `atommovr.algorithms/`

:movie_camera: Visualization of rearrangement process: `atommovr.utils.animation.py`

:computer: Experimentally-relevant error modeling: `atommovr.utils.ErrorModel.py`

:chart_with_upwards_trend: Flexible benchmarking suite for running experiments: `atommovr.utils.benchmarking.py`

:toolbox: Core utils for moving atoms and simulating stochastic loading: `atommovr.utils.core.py` and `atommovr.utils.move_utils.py`


# Use

## Getting started
Want to learn how to use this code? Curious about what it means to be an atom mover?  Head on over to [`tutorial.ipynb`](tutorial.ipynb).

Want to reproduce the figures from our paper? Check out [`paper_figures.ipynb`](paper_figures.ipynb).

Want to add an algorithm to the library, or make your own? Check out our template in [`atommovr.algorithms.Algorithm.py`](/atommovr/algorithms/Algorithm.py).

Want to add some features and/or make this code nicer? Check out [`CONTRIBUTING.md`](CONTRIBUTING.md).


##  Installation

To make the virtual environment, navigate to the main folder and run the following command in Terminal/Powershell: 
```
conda env create -f environment.yml
```

# Acknowledgments

## Collaborators and discussions

We gratefully thank Shraddha Anand, Will Eckner, Noah Glachman, Andy Goldschmidt, Kevin Singh, Mariesa Teo, and Nayana Tiwari.

## External code
- [bottled](https://gitlab.inria.fr/bora-ucar/bottled): (copied into [`\PPSU2023`](/PPSU2023/README.md))
  - Authors: Ioannis Panagiotas, Grégoire Pichon, Somesh Singh, Bora Uçar. 
  - Reference: *Engineering fast algorithms for the bottleneck matching problem.* ESA 2023 - The 31st Annual European Symposium on Algorithms, Sep 2023, Amsterdam (Hollande), Netherlands. [⟨hal-04146298v2⟩](https://inria.hal.science/hal-04146298v2)

## Funding sources
* Office of Naval Research (Grant No. N00014-23-1-2540)
* Air Force Office of Scientific Research (Grant No. FA9550-21-1-0209)
* Army Research Office (Grant no. W911NF2410388) 
* Nikhil acknowledges support from the National Science Foundation Graduate Research Fellowship (Grant No. DGE-2040434) 
* Bo-Yu acknowledges support from the UChicago-Taiwan Student Exchange (UCTS) fellowship, National Taiwan University Fu Bell Scholarship, National Taiwan University College of Science Travel Grants and Chuan-Pu Lee Memorial Scholarship.

## Citation
```
@article{Harle2026atommovropensource,
  doi = {10.22331/q-2026-07-29-2177},
  url = {https://doi.org/10.22331/q-2026-07-29-2177},
  title = {atommovr: {A}n open-source simulation framework for rearrangement in atomic arrays},
  author = {Harle, Nikhil K and Chen, Bo-Yu and Bao, Bob and Bernien, Hannes},
  journal = {{Quantum}},
  issn = {2521-327X},
  publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
  volume = {10},
  pages = {2177},
  month = jul,
  year = {2026}
}
```

Correspondence and request for materials should be directed to [Hannes Bernien](hannes.bernien@uibk.ac.at).
