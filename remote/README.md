# Remote compute policy

Heavy training is **not** run in the local assistant environment. These launchers are prepared for execution on Kaggle, Colab or Lightning AI, subject to account access and available quota.

## Google Colab
Open a new GPU notebook, then run:

```python
!git clone https://github.com/AdebanjiAdelowo/advanced-pinns-failure-analysis.git
%cd advanced-pinns-failure-analysis
!pip install -r requirements-pinn.txt
%env DDE_BACKEND=pytorch
!python -m remote.run_remote --iterations 10000 --seeds 42 0 1 2 3
```

## Kaggle
Create a new notebook, enable Internet for installation and GPU acceleration in notebook settings, and run the same commands. Save `results/remote/` as a Kaggle output artifact.

## Lightning AI
Create a GPU Studio, clone the repository, install `requirements-pinn.txt`, set `DDE_BACKEND=pytorch` and run `python -m remote.run_remote --iterations 10000 --seeds 42 0 1 2 3`.

## Important
This launcher is **prepared, not executed**. DeepXDE's PyTorch backend must be set before importing DeepXDE. The script records runtime, library versions, commit, device availability, seeds and held-out errors. Availability of CUDA does not by itself prove DeepXDE used the GPU; verify device placement and logs before reporting GPU results.
