# DARTS 🎯 playground 🧗‍

### This repo is made for ML2's playground projects
> **🌏 running environment info** <br>
> `python >= 3.6, pytorch == 1.0, and needs CUDA`
> <br><br>
> **Requirements** <br>
>
> `torch`<br>
> `torchvision`<br>
> `graphviz`<br>
> `numpy`<br>
> `tensorboard`<br>
> `tensorboardx`<br>


<br>

### 🚀 How to search and train?
> 🎲 Simply, you can run DARTS search process with <br> &nbsp;&nbsp;&nbsp;&nbsp; `python run.py --name <your_pjt_name> --dataset <data_NAME> --data_path <your_PATH>` <br><br>
> --> ex) `python run.py --name DARTS_test1 --dataset cifar10 --data_path ../data`
> 
> If you need customize some parameters, check `python run.py -h`
>
> This process can visualize by using tensorboard <br>
> (After execute run.py)`tensorboard --logdir=./searchs/<your_pjt_name>/tb --port=6006`<br>
>
> You can visualize arch_graph with `python visualize.py <arch's Genotype>` 
> 

<br>

### 🔗 Process description. 🥚🐣🐥
#### 1. start setting
> 1. Get some arguments in shell
> 2. Set training environment such as using GPU
> 3. Define model(Network) and optimizers
> 4. Make Dataset(dataloader) -- cifar10
> 5. Set lr scheduler
> 6. and Define arch 


#### 2. under training (alpha searching)
>```
1. ○ epoch loop
2. ├─ set lr scheduler 
3. ├─ set genotype
4. ├─○ training loop (start step loop (batch streaming))
5. │ ├─ dataset setting
6. │ ├─○ arch stepping (architecture weight)
7. │ │ ├─ run virtual step & get gradients
8. │ │ ├─ compute hessian
9. │ │ └─ update alpha gradient
10.│ ├─ alpha optimizing
11.│ ├─ model training
12.│ └─ model fitting()
13.└─ validating loop
14. output best model's genotype
```

#### 3. under training (arch searching)



This project is referred from

- DARTS https://arxiv.org/abs/1806.09055

- git https://github.com/quark0/darts
