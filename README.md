# AC4MPC-RTI Snow Hill Reproduction

This repository reproduces the illustrative snow hill experiment from:

**AC4MPC: Actor-Critic Reinforcement Learning for Nonlinear Model Predictive Control**  
Rudolf Reiter, Andrea Ghezzi, Katrin Baumgartner, Jasper Hoffmann,
Robert D. McAllister, Moritz Diehl.

The code implements the AC4MPC-RTI pipeline from the paper: a SAC actor provides
policy rollouts and warm starts, a SAC critic provides the terminal value
approximation, and active/parallel RTI MPC candidates are evaluated before
applying the first control input.

No pretrained weights, training logs, generated solver code, or acados binaries
are included in this repository. Training outputs are written under `runs/`,
which is ignored by git.

## Implemented Components

- Snow hill point-mass system with RK4 discretization.
- SAC training environment for the snow hill task.
- AC4MPC and AC4MPC-RTI controllers.
- `ac4eval` candidate ranking with actor correction, actor rollout, and critic
  terminal value.
- Optional acados SQP-RTI backend for the snow hill OCP.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For acados runs, install acados separately and set the usual runtime paths, for
example:

```bash
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
```

On Windows, make sure the acados `bin`/`lib` directories and MinGW `bin`
directory are available on `PATH`.

## Usage

Run the tests:

```bash
python -m pytest tests
```

Train a SAC policy:

```bash
python experiments/train_sac_snowhill.py \
  --timesteps 50000 \
  --out runs/sac_50k
```

Run AC4MPC-RTI with the local SciPy backend:

```bash
python experiments/run_snowhill.py \
  --controller rti \
  --steps 200 \
  --horizon 20 \
  --eval-rollout 20 \
  --period 5 \
  --alpha 1 \
  --sac-model runs/sac_50k/sac_50000.zip
```

Run AC4MPC-RTI with the acados backend:

```bash
python experiments/run_snowhill.py \
  --controller rti \
  --solver-backend acados \
  --steps 200 \
  --horizon 20 \
  --eval-rollout 20 \
  --period 5 \
  --alpha 1 \
  --solver-iterations 1 \
  --sac-model runs/sac_50k/sac_50000.zip \
  --acados-dir /path/to/acados \
  --mingw-bin /path/to/mingw/bin
```

Generate local plots after training and evaluation:

```bash
python experiments/visualize_snowhill_results.py \
  --sac-model runs/sac_50k/sac_50000.zip
```

## Result
<img width="1152" height="640" alt="training_curve" src="https://github.com/user-attachments/assets/54074748-38eb-4ba2-abae-8c15a3eec2e9" />

<img width="960" height="800" alt="phase_trajectory" src="https://github.com/user-attachments/assets/dd9eb44f-ec23-4760-80cc-f4164ebdc078" />

<img width="960" height="800" alt="phase_trajectory" src="https://github.com/user-attachments/assets/ec84eff1-1372-456f-8421-0f9637cd9cc7" />


## Notes

The snow hill paper section specifies the dynamics, input bound, RK4 step size,
simulation length, MPC horizon, actor/critic architecture, and AC4MPC-RTI
settings. The exact trained checkpoints used in the paper are not included, so
results depend on the locally trained SAC model.

## acados Citation

This project can use acados for SQP-RTI optimal control. If you use the acados
backend, please cite:

```bibtex
@Article{Verschueren2021,
  title   = {acados -- a modular open-source framework for fast embedded optimal control},
  author  = {Robin Verschueren and Gianluca Frison and Dimitris Kouzoupis and Jonathan Frey and Niels van Duijkeren and Andrea Zanelli and Branimir Novoselnik and Thivaharan Albin and Rien Quirynen and Moritz Diehl},
  journal = {Mathematical Programming Computation},
  year    = {2021},
  month   = {Oct},
  day     = {09},
  doi     = {10.1007/s12532-021-00208-8}
}
```
