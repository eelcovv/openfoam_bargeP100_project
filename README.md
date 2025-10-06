# barge windload study 

## running the cases

On a nixos system with openfoam in dockers, run

```bash
python3 batch_process_barges.py --angles=90 --template bargeP100_template --out-root barge --case-prefix barge_ --np 12 --mpirun "mpirun --bind-to none --map-by slot" --mesh-only
```

to make  all the meshes

Run:

```bash
python batch_process_barges.py --angles=90 --template bargeP100_template --out-root barge --case-prefix barge_ --np 12 --mpirun "mpirun --bind-to none --map-by slot" --start-solver
```

start the solvers.

On a normal openfoam system run:

```bash
python batch_process_barges.py --angles=90 --template bargeP100_template --out-root barge --case-prefix barge_ --start-solver
```
