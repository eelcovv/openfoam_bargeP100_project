python batch_process_barges.py \
  --angles=90 \
  --template bargeP100_template \
  --out-root barge \
  --case-prefix barge_ \
  --np 16 \
  --mpirun "mpirun --bind-to none --map-by slot" \
  --mesh-only

