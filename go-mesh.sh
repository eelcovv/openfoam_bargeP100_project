python batch_process_barges.py \
  --angles=270 \
  --template templates/template_draft2100_35mps \
  --out-root cases \
  --case-prefix case_d2100_35mps_r \
  --np 16 \
  --mpirun "mpirun --bind-to none --map-by slot" \
  --start-solver

