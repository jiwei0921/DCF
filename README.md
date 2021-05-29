# DCF Code


### > Requirment
+ pytorch 1.0.0+
+ torchvision
+ PIL
+ numpy



### > Usage

#### 1. Inference Phase

> Our saliency maps.

【**1**】[Saliency Maps](https://pan.baidu.com/s/1cliPYQubTPb4W48kl4qliA), (fetch code is **j93d**), by our DCF trained on NJUD & NLPR (2185). 

【**2**】[Saliency Maps](https://pan.baidu.com/s/1plEYHtgmkToz8HO2XP03gA), (fetch code is **aeq0**), by our DCF trained on NJUD & NLPR & DUT (2985).

> Our pre-trained model for inferring your own dataset.

【**1**】Download the [pre-trained model](https://pan.baidu.com/s/1gWHgW1H9YiNc7hcL4jwdrQ), (fetch code is **ceqa**), which is trained on NJUD & NLPR & DUT. 

【**2**】Set the data path and ckpt_name in ```demo_test.py```, correctly.

【**3**】Run ```python demo_test.py``` to obtain the saliency maps.


#### 2. Training Phase

【**1**】Stage 1: Run ```python demo_train_pre.py```, which performs the **D**epth **C**alibration Strategy.

【**2**】Stage 2: Run ```python demo_train.py```, which performs the **F**usion Strategy.



### For Reference

All common RGB-D Saliency Datasets we collected are shared in ready-to-use manner.       
+ The web link is [here](https://github.com/jiwei0921/RGBD-SOD-datasets).

All results in our paper are measured by the ready-to-use evaluation toolbox.
+ The web link is [here](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).


### Acknowledgement

We thank all reviewers for their valuable suggestions. At the same time, thanks to the large number of researchers contributing to the development of open source in this field, particularly, [Deng-ping Fan](http://dpfan.net), [Runmin Cong](https://rmcong.github.io), [Tao Zhou](https://taozh2017.github.io), etc.

Our feature extraction network is based on [CPD backbone](https://github.com/wuzhe71/CPD).



#### Contact Us
If you have any questions, please contact us ( wji3@ualberta.ca ).
