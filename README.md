Operator Inference and Data-Informed MOR for the Incompressible NSE
---

This is the code that accompanies the preprint

> Benner, Goyal, Heiland, Pontes (2020): [*Operator inference and Physics-based learning of low-dimensional model for incompressible flows*](https://arxiv.org/abs/2010.06701)

The sources are in `nse_opinf_poddmd`.

The scripts that produce the numerical results and the plots of the paper are in
`scripts`.

## Examples of Operator Inference / POD / DMD

```bash
cd scripts
source setpypath.sh
python3 main_opting_cwdc.py
```

## Reproducing the Data and Plots of the paper

```bash
cd scripts
source runme.sh
```

The plots are then found in `scripts/Figures/`.

## Rerun the Mesh Check

```bash
cd scripts
source setpypath
python3 convergence_tests.py
```

## Versions

 - tag `1.0` initial version (as in https://arxiv.org/abs/2010.06701)
 - tag `1.1` first revision (with splitting the data for testing and prediction)
