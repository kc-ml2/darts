# DARTS 🎯 playground 🧗‍

### This repo is made for ML2's playground projects
> **🌏 running environment info** <br>
> `python >= 3.6, pytorch == 1.0, and needs CUDA`
<br>

### 🚀 How to train?
> 🎲 You can run DARTS using `python run.py` 
>
> If you need customize some parameters, check `python run.py -h`
<br>

### 🔗 Process description. 🥚🐣🐥
#### 1. start setting
> 1. Get some arguments in shell
> 2. Set training environment such as using GPU
> 3. Define model(Network) and optimizers
> 4. Make Dataset(dataloader) -- cifar10
> 5. Set lr scheduler
> 6. and Define arch 

#### 2. under training
> 1. start epoch loop
> 2. -- set lr scheduler 
> 3. -- set genotype
> 4. -- start training
> 5. ---- start step loop
> 6. ------ dataset setting
> 7. ------ arch stepping (architecture weight)
> 8. -------- backward
> 9. -------- optimizer step
> 10. ----- model fitting()
> 11. ----- 