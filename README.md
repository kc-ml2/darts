# DARTS:Differentiable Architecture Search 🎯 - playground 🧗‍

### This repo is made for ML2's playground projects
 **🌏 running environment info** <br>
 `python >= 3.6, pytorch == 1.0, and needs CUDA`
 <br><br>
 **Requirements** <br>

> `torch`<br>
> `torchvision`<br>
> `graphviz`<br>
> `numpy`<br>
> `tensorboard`<br>
> `tensorboardx`<br>


<br>

### 🚀 How to search and train?
#### 🎲 Search process
 - Simply, you can run DARTS for architecture search process with <br> &nbsp;&nbsp;&nbsp;&nbsp; `python run.py --name <your_pjt_name> --dataset <data_NAME> --data_path <your_PATH>` <br><br>
 ex) `python run.py --name DARTS_test1 --dataset cifar10 --data_path ../data`


> ---

- This process can visualize by using tensorboard <br>
 (After execute run.py) `tensorboard --logdir=./searchs/<your_pjt_name>/tb --port=6006` <br>

- Check localhost:6006(or `<your ip>`:6006) by your browser.



<br>

#### 🎲 Train/Test process 
- After finished search or need proving some model architecture, then run <br> &nbsp;&nbsp;&nbsp;&nbsp; `python run_from.py --name <pjt_name> --dataset <data_NAME> --data_path <your_PATH> --genotype <Genotype>` <br><br>
ex) `python run_from.py --name DARTS_test1 --dataset cifar10 --data_path ../data --genotype Genotype(normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('skip_connect', 0), ('sep_conv_3x3', 1)], [('skip_connect', 0), ('sep_conv_3x3', 1)], [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 1), ('skip_connect', 2)], [('skip_connect', 2), ('max_pool_3x3', 1)], [('dil_conv_5x5', 4), ('dil_conv_5x5', 2)]], reduce_concat=range(2, 6))`


> ---

- This process also available visualizing <br>
 (After execute run_from.py)`tensorboard --logdir=./augments/<your_pjt_name>/tb --port=6007`<br>

- Check localhost:6007(or `<your ip>`:6007) by your browser.

This process makes you can check model(architecture)'s loss and accuracy.

<br>

#### 🕹 more
- You can visualize arch_graph with `python visualize.py <arch's Genotype>` 

- Finded `Genotype` is recorded in last line of `search/<your_pjt_name>/<your_pjt_name>.log`

- If you need customize some parameters, check `python run.py -h` or `python run_from.py -h`


<br>

### 🏁 Results (The average value of the results)

|mode|runtime(avg)|train acc|val acc|environment|GPU(single)|params|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Search       | 29hr | 99.9% | 91.3% | py3.6 // cuda10 // torch 1.0 | Titan V | epoch=100, dataset=cifar10, workers=12, batch_size=64 |
| Train/Test   | 8hr  | 98.6% | 96.7% | py3.6 // cuda10 // torch 1.0 | Titan V | epoch=300, dataset=cifar10, workers=16, batch_size=96 |
| Train/Test   | 24hr  | 99.0% | 97.2% | py3.6 // cuda10 // torch 1.0 | Titan V | **epoch=600**, dataset=cifar10, workers=16, batch_size=96 |

<br>

### 🔗 Process description. 🥚🐣🐥

#### 1. Start setting
  1. Get some arguments in shell
  2. Set training environment such as using GPU
  3. Define model(Network) and optimizers
  4. Make Dataset(dataloader) -- cifar10
  5. and Define arch (only search process)


#### 2. Alpha searching (arch searching)
```
1. ○ epoch loop
2. ├─ set lr scheduler 
3. ├─ set genotype
4. ├─○ training loop (batch streaming)
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


#### 3. optimizing searched model (run_from.py)
```
1. ○ epoch loop
2. ├─ set lr scheduler 
3. ├─ set dropout genotype
4. ├─○ training loop
5. │ ├─ dataset setting
6. │ ├─ model training
7. │ └─ model fitting()
8. └─ validating loop
9. output model's best score
```

<br>
<br>
<br>

Reference

- DARTS paper https://arxiv.org/abs/1806.09055


- official git https://github.com/quark0/darts


- codes  
    - https://github.com/MandyMo/DARTS
    - https://github.com/khanrc/pt.darts
    - https://github.com/galvinw/darts


- web
    - http://openresearch.ai/t/darts-differentiable-architecture-search/355
    
