import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from evaluateSOD.evaluator import Eval_thread
from evaluateSOD.dataloader import EvalDataset
# from concurrent.futures import ThreadPoolExecutor


def evalateSOD(save_path,gt_root,dataset,ckpt):
    threads = []
    cuda = torch.cuda.is_available()
    output_dir = './'
    loader = EvalDataset(save_path, gt_root)
    thread = Eval_thread(loader, method=ckpt, dataset=dataset, output_dir=output_dir, cuda=cuda)
    threads.append(thread)
    for thread in threads:
        mae,msg = thread.run()
        print(msg)
        return mae

