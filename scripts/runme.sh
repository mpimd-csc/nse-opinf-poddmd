export PYTHONPATH="../"

# ## uncomment to use the reference dat
# mkdir results
# cp reference-data/*.json results/

python3 pplot_dc_pod_sv_decay.py
# generates Fig. 7.4

python3 pplot_dc_lcurve_optinf.py
# generates Fig. 7.5

python3 pplot_dc_timedomain_error.py
# generates Fig. 7.6

python3 pplot_dc_error_vs_rom.py
# generates Fig. 7.8

python3 pplot_cw_pod_sv_decay.py
# generates Fig. 7.9

python3 pplot_cw_error_vs_rom.py
