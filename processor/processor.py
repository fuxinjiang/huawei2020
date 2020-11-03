import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import cv2
from utils.meter import AverageMeter
from utils.metrics import R1_mAP, R1_mAP_Pseudo
from utils.reranking import re_ranking, euclidean_distance
from utils.db_qe import retrieve, calculate_sim_matrix
import json
import datetime
from torch.cuda.amp import autocast as autocast, GradScaler
#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')

    if device:
        model.to(device)
        #print("cuda个数", torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    triplet_loss_meter = AverageMeter()
    # train
    scaler = GradScaler()
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        ce_loss_meter.reset()
        triplet_loss_meter.reset()

        model.train()
        for n_iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.cuda(non_blocking=True)
            target = vid.cuda(non_blocking=True)
            with autocast():
                score, feat = model(img, target)
                loss, ce_loss, triplet_loss = loss_fn(score, feat, target)
            scaler.scale(loss).backward()
            # optimizer.module.step()
            scaler.step(optimizer)
            scaler.update()
            acc = (score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            ce_loss_meter.update(ce_loss.item(), img.shape[0])
            triplet_loss_meter.update(triplet_loss.item(), img.shape[0])

            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Ce Loss: {:.3f}, Triplet Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, ce_loss_meter.avg, triplet_loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
        scheduler.step()
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return 1 - cosine
            
def do_inference(cfg, model, val_loader, num_query, query_name, gallery_name):
    time_start = time.time()
    model.eval()
    model.cuda()
    model = nn.DataParallel(model)
    feature = torch.FloatTensor().cuda()
    with torch.no_grad():
        for (img, pid) in val_loader:
            input_img = img.cuda()
            input_img_mirror = input_img.flip(dims=[3])
            outputs = model(input_img)
            outputs_mirror = model(input_img_mirror)
            f = outputs + outputs_mirror
            feature = torch.cat((feature, f), 0)  
    #feats = torch.nn.functional.normalize(features, dim=1, p=2)
    #query_vecs = feature[-num_query:, :]
    #reference_vecs = feature[:-num_query, :]
    del model
    feature_ = feature.cpu().numpy()
    np.save('ibn_a_600.npy', feature_)
    if  cfg.TEST.DB_QE:
        print("进行db_qe过程")
        query_vecs = feature[-num_query:, :].cpu().numpy()
        reference_vecs = feature[:-num_query, :].cpu().numpy()
        query_vecs, reference_vecs, distmat = retrieve(query_vecs, reference_vecs)
        if cfg.TEST.RE_RANKING:
            print("reranking过程")
            feature = torch.cat((torch.tensor(reference_vecs, dtype=torch.float32), torch.tensor(query_vecs, dtype=torch.float32)), 0)
            feature = torch.nn.functional.normalize(feature, dim=1, p=2)
            query_vecs = feature[-num_query:, :]
            reference_vecs = feature[:-num_query, :]
            ranking_parameter = cfg.TEST.RE_RANKING_PARAMETER
            k1 = ranking_parameter[0]
            k2 = ranking_parameter[1]
            lambda_value = ranking_parameter[2]
            distmat = re_ranking(query_vecs, reference_vecs, k1=k1, k2=k2, lambda_value=lambda_value)
        
    elif cfg.TEST.RE_RANKING:
        print("reranking过程")
        feature = torch.nn.functional.normalize(feature, dim=1, p=2)
        query_vecs = feature[-num_query:, :]
        reference_vecs = feature[:-num_query, :]
        ranking_parameter = cfg.TEST.RE_RANKING_PARAMETER
        k1 = ranking_parameter[0]
        k2 = ranking_parameter[1]
        lambda_value = ranking_parameter[2]
        distmat = re_ranking(query_vecs, reference_vecs, k1=k1, k2=k2, lambda_value=lambda_value)
    else :
        print("最原始的计算距离过程")
        query_vecs = feature[-num_query:, :].cpu().numpy()
        reference_vecs = feature[:-num_query, :].cpu().numpy()
        print(query_vecs.shape)
        distmat = calculate_sim_matrix(query_vecs, reference_vecs)
        #query_vecs = feature[-num_query:, :]
        #reference_vecs = feature[:-num_query, :]
        #distmat = cosine_dist(query_vecs, reference_vecs)
        #distmat = distmat.cpu().numpy()
    np.save('distmat_a_600.npy', distmat)
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    max_10_indices = indices[:, :10]
    res_dict = dict()
    for q_idx in range(num_q):
        filename = query_name[q_idx][0].split("\\")[-1]
        max_10_files = [gallery_name[i][0].split("\\")[-1] for i in max_10_indices[q_idx]]
        res_dict[filename] = max_10_files 
    with open('submission.csv', 'w') as file:
        for k, v in res_dict.items():
            writer_string = "%s,{%s,%s,%s,%s,%s,%s,%s,%s,%s,%s}\n"%(k, v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9])
            file.write(writer_string)
    file.close()
    time_final = time.time()
    print("总共用时:", time_final-time_start)
    
    

def do_inference_Pseudo(cfg,
                 model,
                val_loader,
                num_query
                 ):
    device = "cuda"

    evaluator = R1_mAP_Pseudo(num_query, max_rank=200, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    reranking_parameter = [14, 4, 0.4]

    model.eval()
    for n_iter, (img, pid, camid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.TEST.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            evaluator.update((feat, imgpath))

    distmat, img_name_q, img_name_g = evaluator.compute(reranking_parameter)

    return distmat, img_name_q, img_name_g