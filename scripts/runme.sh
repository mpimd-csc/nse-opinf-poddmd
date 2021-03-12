export PYTHONPATH="../"

# ## uncomment to use the reference data
# mkdir results
# cp reference-data/*.json results/

python3 pplot_dc_pod_sv_decay.py
# generates Fig. 7.4

python3 pplot_dc_lcurve_optinf.py
# generates Fig. 7.5

python3 pplot_dc_timedomain_error.py
# generates Fig. 7.7

python3 pplot_dc_pressure_approximation.py
# generates Fig. 7.6

# ## uncomment for computing (it's time consuming though)
#python3 pplot_dc_error_vs_rom.py
# generates Fig. 7.8

python3 pplot_cw_pod_sv_decay.py
# generates Fig. 7.9

python3 pplot_cw_timedomain_error.py
# generates Fig. 7.10.
# and generates the data for 7.13 (use paraview to examine)

python3 pplot_cw_pressure_approximation.py
# generates Fig. 7.11

# ## uncomment for computing (it's time consuming though)
# python3 pplot_cw_error_vs_rom.py
# generates Fig. 7.12

