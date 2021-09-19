## DCF (& DCBF)

Code repository for our paper entilted ["Calibrated RGB-D Salient Object Detection"](https://openaccess.thecvf.com/content/CVPR2021/papers/Ji_Calibrated_RGB-D_Salient_Object_Detection_CVPR_2021_paper.pdf) accepted at CVPR 2021.


# :notebook_with_decorative_cover: Source Code

### > Requirment
+ pytorch 1.0.0+; torchvision; PIL; numpy

### > Usage

#### 1. Inference Phase

> Our saliency maps.

【**1**】[Saliency Maps](https://pan.baidu.com/s/1cliPYQubTPb4W48kl4qliA), (fetch code is **j93d**), by our DCF trained on NJUD & NLPR (2185). 

【**2**】[Saliency Maps](https://pan.baidu.com/s/1plEYHtgmkToz8HO2XP03gA), (fetch code is **aeq0**), by our DCF trained on NJUD & NLPR & DUT (2985).

+ Notice that, the depth map is unified, which means that the closer region is closer to 1, and the farther region is closer to 0. The testset results of the new dataset **ReDWeb-S** can be downloaded in [here](https://pan.baidu.com/s/113jtSoEvdnRDem0-2ifREA) (fetch code is **likm**). 

> Our pre-trained model for inferring your own dataset.

【**1**】Download the [pre-trained model](https://pan.baidu.com/s/1JiMcwkezm6jldQfPCDSLUg), (fetch code is **2t7g**), which is trained on NJUD & NLPR & DUT. Or the [another model](https://pan.baidu.com/s/1X4mWuJ8SlOKRP210C1RwoA), (fetch code is **epp9**), which is trained on NJUD & NLPR.

【**2**】Set the data path and ckpt_name in ```demo_test.py```, correctly.

【**3**】Run ```python demo_test.py``` to obtain the saliency maps.


#### 2. Training Phase

【**1**】Stage 1: Run ```python demo_train_pre.py```, which performs the **D**epth **C**alibration Strategy.

【**2**】Stage 2: Run ```python demo_train.py```, which performs the **F**usion Strategy.


### > Evaluation/Training Setup

+ The related all test datasets in this paper can be found in [this link (fetch code is **b2p2**)](https://pan.baidu.com/s/1sx1En1ecNyDf12jNGFeYZQ).
+ [This evaluation tool](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox) is used to evaluate the above saliency maps in this paper.

+ The training set used in this paper can be accessed in [(NJUD+NLPR), code is **76gu**](https://pan.baidu.com/s/1sNxe3Szu7O_Qci1OGmKIKQ) and [(NJUD+NLPR+DUT), code is **201p**](https://pan.baidu.com/s/19aiosd_73VGMg7PB7HJzww).


### DCBF is an extension of our DCF. Its code can be approached in [BaiduYun](https://pan.baidu.com/s/1tZt132ggznxeDzJEfrEEUg) (code is 'p5b5').

### Acknowledgement

We thank all reviewers for their valuable suggestions. At the same time, thanks to the large number of researchers contributing to the development of open source in this field, particularly, [Deng-ping Fan](http://dpfan.net), [Runmin Cong](https://rmcong.github.io), [Tao Zhou](https://taozh2017.github.io), etc.

Our feature extraction network is based on [CPD backbone](https://github.com/wuzhe71/CPD).

### Bibtex
```
@InProceedings{Ji_2021_DCF,
    author    = {Ji, Wei and Li, Jingjing and Yu, Shuang and Zhang, Miao and Piao, Yongri and Yao, Shunyu and Bi, Qi and Ma, Kai and Zheng, Yefeng and Lu, Huchuan and Cheng, Li},
    title     = {Calibrated RGB-D Salient Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {9471-9481}
}
```

#### Contact Us
If you have any questions, please contact us ( wji3@ualberta.ca ).
