import json
import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm as tqdm
from scipy.spatial.distance import euclidean, cosine
import numpy as np
import pickle,json
import time
from multiprocessing import Pool,Manager
from sklearn.neighbors import DistanceMetric

def write_result_file_image(a,file_name='output.txt'):
    outputs_a = ['%s %s'%(key,','.join(a[key])) for key in a]
    outputs = '\n'.join(outputs_a)
    with open(file_name,'w') as f:
        f.write(outputs)
    
    
def _pair_distances(data):
    text,imgs,rule = data
    scores = []
    for img in imgs:
        if rule=='cos':
            scores.append(cosine(text,img))
        else:
            scores.append(euclidean(text,img))
    return scores

def compute_scoremat(data):
    """
    data = (texts,images,rule) tuple
    texts: MxK embeddings of text
    images: NxK embeddings of images
    rules: "rect" or "cos" distances
    
    return np scoremat with MxN
    """
    start = time.time()
    texts,images,rule = data
    M = texts.shape[0]
    dist = DistanceMetric.get_metric('euclidean')
    scoremat = dist.pairwise(texts,images)
    
    #'''
    end = time.time()
    print('finish compute_scoremat in %.3f sec.'%(end-start))
    return scoremat


def parse_submission(submission_file):
    with open(submission_file) as f:
        lines = f.readlines()
    submission = []
    for line in lines:
        words = line.strip().split()
        if len(words) != 2:
            print('Format Error!')
            return None
        res = {}
        key = words[0].strip()
        ret = words[1].strip().split(',')
        unique_ret = []
        appeared_set = set()
        for x in ret:
            if x not in appeared_set:
                unique_ret.append(x)
                appeared_set.add(x)
        res[key] = ret
        submission.append(res)
    return submission


def get_topk(gt_dict, ret_list):
    kvals = [1, 5, 10]
    # Since each img has two sentences as labels, the query num is doubled of original test set size.
    query_num = len(ret_list)
    correct = {}
    for kval in kvals:
        correct[kval] = 0
    for ret in ret_list:
        img_name = list(ret.keys())[0]
        for kval in kvals:
            for i in range(kval):
                query_id = gt_dict[img_name]
                gallery_id = gt_dict[ret[img_name][i]]
                if query_id == gallery_id:
                    correct[kval] += 1
                    break
    acc = []

    for kval in kvals:
        acc.append(correct[kval] / float(query_num))

    return acc


def eval(submission_file, gt_file):
    with open(gt_file, 'r') as f:
        gt_dict = json.load(f)
    submission = parse_submission(submission_file)
    acc = get_topk(gt_dict, submission)
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gt", type=str)
    parser.add_argument("submission", type=str)
    args = parser.parse_args()

    submit_file = args.submission
    gt_file = args.gt

    acc = eval(submit_file, gt_file)
    out = {'top-1': acc[0], 'top-5': acc[1], 'top-10': acc[2]}
    keys = ['top-1', 'top-5', 'top-10']
    for k in keys:
        print('{}: {:.2%}'.format(k, out[k]))

        