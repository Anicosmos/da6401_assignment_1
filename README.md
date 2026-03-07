# Assignment 1: Multi-Layer Perceptron from Scratch

Built a neural network using only NumPy

## What i was able to  Implement

- [x] Loss Functions (Cross-Entropy, MSE) and Activations (ReLU, Sigmoid, Tanh)
- [x] Optimizers (SGD, Momentum, RMSProp, NAG)
- [x] Neural Layer (forward + backward pass) But More validation neede3d
- [x] Neural Network class (training, evaluation, save/load)
- [x] Training script with CLI args
- [x] W&B logging (sweeps, metrics, image tables)
- [x] Hyperparameter search (Bayes + Grid)
- [] Experiments notebook (dataset viz, optimizer showdown, loss comparison, weight init)

## How to Run

```
python src/train.py -d mnist -e 1000 -b 32 -lr 0.001 -o sgd -sz 6 --num_neurons 120 -a relu -l mse -w_i random "test" ./models/
```

## Experiments

All experiments are in `notebooks/experiments_updated.ipynb` and `notebooks/experiments_updated.ipynb`, split by question.

## Notes

- Bayes search was used for the 100-run hyperparameter sweep
- Best F1 score came from NAG, not RMSProp (RMSProp probably needs more LR tuning — only tried 0.001, 0.01, 0.1) 
- LR 0.1 caused dead neurons with ReLU

## Links

- **W&B Report:** [Link](https://wandb.ai/anicosmos-iitm/da6401_a1/reports/Assignment-1---VmlldzoxNjEzNDgyMw?accessToken=olix2q8aaxcg8l0mri3oa35lo25po9ib3lhyc45il3a0eupsuxk6vduxyknxcprb)
- **GitHub Repo:** [Link](https://github.com/Anicosmos/da6401_assignment_1)


## Contact

Anirudh Bharadwaj — EE25S046
