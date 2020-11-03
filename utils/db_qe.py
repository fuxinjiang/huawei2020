# -*- coding: utf_8 -*-

import numpy as np
import pandas as pd
import multiprocessing
import os
import json
from joblib import Parallel, delayed
import multiprocessing
from scipy import sparse
import csv
import heapq
import time
import torch
import gc
from tqdm import tqdm

def retrieve(query_vecs, reference_vecs):
    #print('db过程')
    #query_vecs, reference_vecs = db_augmentation(query_vecs, reference_vecs, top_k=10)
    print("aqe过程")
    query_vecs, reference_vecs = average_query_expansion(query_vecs, reference_vecs, top_k=5)
    sim_matrix = calculate_sim_matrix(query_vecs, reference_vecs)
    return query_vecs, reference_vecs, sim_matrix
def db_augmentation(query_vecs, reference_vecs, top_k=10):
    """
    Database-side feature augmentation (DBA)
    Albert Gordo, et al. "End-to-end Learning of Deep Visual Representations for Image Retrieval,"
    International Journal of Computer Vision. 2017.
    https://link.springer.com/article/10.1007/s11263-017-1016-8
    """
    weights = np.logspace(0, -2., top_k+1)
    # Query augmentation
    print('Query augmentation')
    sim_mat = calculate_sim_matrix(query_vecs, reference_vecs)
    indices = np.argsort(sim_mat, axis=1)
    top_k_ref = reference_vecs[indices[:, :top_k], :]
    query_vecs = np.tensordot(weights, np.concatenate([np.expand_dims(query_vecs, 1), top_k_ref], axis=1), axes=(0, 1))
    print('Reference augmentation')
    # Reference augmentation
    sim_mat = calculate_sim_matrix(reference_vecs, reference_vecs)
    indices = np.argsort(sim_mat, axis=1)

    top_k_ref = reference_vecs[indices[:, :top_k+1], :]
    reference_vecs = np.tensordot(weights, top_k_ref, axes=(0, 1))

    return query_vecs, reference_vecs


def average_query_expansion(query_vecs, reference_vecs, top_k=5):
    """
    Average Query Expansion (AQE)
    Ondrej Chum, et al. "Total Recall: Automatic Query Expansion with a Generative Feature Model for Object Retrieval,"
    International Conference of Computer Vision. 2007.
    https://www.robots.ox.ac.uk/~vgg/publications/papers/chum07b.pdf
    """
    # Query augmentation
    print('Query augmentation')
    sim_mat = calculate_sim_matrix(query_vecs, reference_vecs)
    indices = np.argsort(sim_mat, axis=1)

    top_k_ref_mean = np.mean(reference_vecs[indices[:, :top_k], :], axis=1)
    query_vecs = np.concatenate([query_vecs, top_k_ref_mean], axis=1)
    print('Reference augmentation')
    # Reference augmentation
    sim_mat = calculate_sim_matrix(reference_vecs, reference_vecs)
    indices = np.argsort(sim_mat, axis=1)

    top_k_ref_mean = np.mean(reference_vecs[indices[:, 1:top_k+1], :], axis=1)
    reference_vecs = np.concatenate([reference_vecs, top_k_ref_mean], axis=1)

    return query_vecs, reference_vecs

    
def postprocess(query_vecs, reference_vecs):
    """
    Postprocessing:
    1) Moving the origin of the feature space to the center of the feature vectors
    2) L2-normalization
    """
    # centerize
    query_vecs, reference_vecs = _centerize(query_vecs, reference_vecs)
    # l2 normalization
    query_vecs = _l2_normalize(query_vecs)
    reference_vecs = _l2_normalize(reference_vecs)
    return query_vecs, reference_vecs


def _centerize(v1, v2):
    concat = np.concatenate([v1, v2], axis=0)
    center = np.mean(concat, axis=0)
    return v1-center, v2-center


def _l2_normalize(v):
    norm = np.expand_dims(np.linalg.norm(v, axis=1), axis=1)
    if np.any(norm == 0):
        return v
    return v / norm
def calculate_sim_matrix(query_vecs, reference_vecs):
    query_vecs, reference_vecs = postprocess(query_vecs, reference_vecs)
    #改用多线程进行计算，加快矩阵相乘速度
    return 2-2*np.dot(query_vecs, reference_vecs.T)

