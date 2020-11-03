#开始进行模型融合
import numpy as np
import torch as nn
import torch
import torchvision
from torchvision import datasets, transforms
from config import default
from utils.logger import setup_logger
from utils.reranking import re_ranking, euclidean_distance
import os
import argparse
from glob import glob
#基于距离的模型融合
def model_ronghe_dist(dist_x, dist_y):
    return dist_x+dist_y
#基于预测向量相加然后在进行reranking的融合
def model_ronghe_vector(vec_x, vec_y):
    return vec_x+vec_y

def model_ronghe_concat(vec_x, vec_y):
    return np.concatenate((vec_x, vec_y), axis=1)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg = default._C

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    num_query = len(os.listdir("../test_data_final/test_data_A\\query"))
    '''
    query_name = np.load("querys_sort.npy").tolist()
    gallery_name = np.load("gallerys_sort.npy").tolist()
    query_name_chizhu = [x.split("/")[-1] for x in query_name]
    gallery_name_chizhu = [x.split("/")[-1] for x in gallery_name]
    query_name_jfx = np.load("query_name_jfx.npy").tolist()
    gallery_name_jfx = np.load("gallery_name_jfx.npy").tolist()
    transforms_query = [query_name_jfx.index(x) for x in query_name_chizhu]
    transforms_gallety = [gallery_name_jfx.index(x) for x in gallery_name_chizhu]
    '''
    query_name_ori = np.load("nurbs_querys.npy").tolist()
    gallery_name_ori = np.load("nurbs_gallerys.npy").tolist()
    query_name = [x.split("/")[-1] for x in query_name_ori]
    gallery_name = [x.split("/")[-1] for x in gallery_name_ori]
    query_name_jfx_ori = glob("../test_data_final/test_data_A\\query/*.jpg")
    gallery_name_jfx_ori = glob("../test_data_final/test_data_A\\gallery/*.jpg")
    query_name_jfx = [x.split("\\")[-1] for x in query_name_jfx_ori]
    gallery_name_jfx = [x.split("\\")[-1] for x in gallery_name_jfx_ori]
    transforms_query = [query_name_jfx.index(x) for x in query_name]
    transforms_gallety = [gallery_name_jfx.index(x) for x in gallery_name]

    #reranking部分改一下，尝试一下
    #filter_img = np.load("bad_gallery.npy").tolist()
    #after_filter_index = [gallery_name_chizhu.index(x) for x in gallery_name_chizhu if x not in filter_img]
    #after_filter_gallery = [x for x in gallery_name_chizhu if x not in filter_img]

    # print(query_name[0])
    #gallery_name = glob("../test_data_B\\gallery/*.jpg")
    '''
    transform_test = transforms.Compose([
        transforms.Resize(cfg.INPUT.SIZE_TEST),
        # 将图像中央的高和宽均为224的正方形区域裁剪出来
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.524, 0.4943, 0.473), (0.03477, 0.03015, 0.02478))
        transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    batch_size = 256
    testset = torchvision.datasets.ImageFolder(root='../test_data_B', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8,
                                              pin_memory=True)

    num_query = len(os.listdir("../test_data_B\\query"))
    query_name = testset.samples[-num_query:]
    # print(query_name[0])
    gallery_name = testset.samples[:-num_query]

    dist_a = np.load("distmat_a.npy")
    dist_b = np.load("distmat_b.npy")
    dist_se = np.load("distmat_se.npy")
    distmat = model_ronghe_dist(dist_a, dist_b)
    distmat = model_ronghe_dist(distmat, dist_se)

    feature_a = np.load("ibn_a.npy")
    feature_b = np.load("ibn_b.npy")
    feature_se = np.load("seibn_a.npy")
    feature = model_ronghe_concat(feature_a, feature_b)
    feature = model_ronghe_concat(feature, feature_se)
    print(feature.shape)
    feature = torch.tensor(feature, dtype=torch.float32)
    print("reranking过程")
    feature = torch.nn.functional.normalize(feature, dim=1, p=2)
    query_vecs = feature[-num_query:, :]
    reference_vecs = feature[:-num_query, :]
    ranking_parameter = cfg.TEST.RE_RANKING_PARAMETER
    k1 = ranking_parameter[0]
    k2 = ranking_parameter[1]
    lambda_value = ranking_parameter[2]
    query_total = np.load("query_pred_3_fold.npy")
    gallery_total = np.load("gallery_pred_3_fold.npy")
    query_pred_0 = query_total[:, :1000]
    gallery_pred_0 = gallery_total[:, :1000]
    query_pred_1 = query_total[:, 1000:2000]
    gallery_pred_1 = gallery_total[:, 1000:2000]
    query_pred_2 = query_total[:, 2000:3000]
    gallery_pred_2 = gallery_total[:, 2000:3000]
    del query_total
    del gallery_total
    
    
    feature0 = np.concatenate((query_pred_0, gallery_pred_0), axis=0)
    print(feature0.shape)
    feature0 = torch.tensor(feature0, dtype=torch.float32)
    feature0 = torch.nn.functional.normalize(feature0, dim=1, p=2)
    query_vecs0 = feature0[:num_query, :]
    reference_vecs0 = feature0[num_query:, :]
    distmat0 = re_ranking(query_vecs0, reference_vecs0, k1=k1, k2=k2, lambda_value=lambda_value)
    np.save('zhou_0.npy', distmat0)
    del feature0, query_pred_0, gallery_pred_0


    feature1 = np.concatenate((query_pred_1, gallery_pred_1), axis=0)
    print(feature1.shape)
    feature1 = torch.tensor(feature1, dtype=torch.float32)
    feature1 = torch.nn.functional.normalize(feature1, dim=1, p=2)
    query_vecs1 = feature1[:num_query, :]
    reference_vecs1 = feature1[num_query:, :]
    distmat1 = re_ranking(query_vecs1, reference_vecs1, k1=k1, k2=k2, lambda_value=lambda_value)
    np.save('zhou_1.npy', distmat1)
    del feature1, query_pred_1, gallery_pred_1
    
    feature2 = np.concatenate((query_pred_2, gallery_pred_2), axis=0)
    print(feature2.shape)
    feature2 = torch.tensor(feature2, dtype=torch.float32)
    feature2 = torch.nn.functional.normalize(feature2, dim=1, p=2)
    query_vecs2 = feature2[:num_query, :]
    reference_vecs2 = feature2[num_query:, :]
    distmat2 = re_ranking(query_vecs2, reference_vecs2, k1=k1, k2=k2, lambda_value=lambda_value)
    np.save('zhou_2.npy', distmat2)
    del feature2, query_pred_2, gallery_pred_2
    #distmat = distmat0 + distmat1 + distmat2
    #del distmat0 , distmat1 , distmat2
    '''
    '''
    query_chizhu = np.load("chizhu_efb6_querys_pred.npy")
    gallery_chizhu = np.load("chizhu_efb6_gallery_pred.npy")
    feature_chizhu = np.concatenate((query_chizhu, gallery_chizhu), axis=0)
    print(feature_chizhu.shape)
    feature_chizhu = torch.tensor(feature_chizhu, dtype=torch.float32)
    feature_chizhu = torch.nn.functional.normalize(feature_chizhu, dim=1, p=2)
    query_vecs_chizhu = feature_chizhu[:num_query, :]
    reference_vecs_chizhu = feature_chizhu[num_query:, :]
    distmat_chizhu = re_ranking(query_vecs_chizhu, reference_vecs_chizhu, k1=k1, k2=k2, lambda_value=lambda_value)
    np.save('chizhu_dist.npy', distmat_chizhu)
    del feature_chizhu
    '''
    '''
    #这个位置要先进行转换
    query_ibn_b = query_ibn_b[transforms_query, :]
    reference_ibn_b = reference_ibn_b[transforms_gallety, :]
    reference_ibn_b = reference_ibn_b[after_filter_index, :]
    distmat_ibn_b = re_ranking(query_ibn_b, reference_ibn_b, k1=k1, k2=k2, lambda_value=lambda_value)
    np.save('F:\\huawei_dist\\ibn_b_after.npy', distmat_ibn_b)
    del query_ibn_b, reference_ibn_b
    '''

    #distmat = distmat0
    print("三折")
    distmat0 = np.load("zhou_0.npy")
    distmat1 = np.load("zhou_1.npy")
    distmat2 = np.load("zhou_2.npy")
    distmat = distmat0 + distmat1 + distmat2
    #distmat = distmat0 + distmat2
    del distmat0 , distmat1 , distmat2

    print("ibnnet")
    #distmat_a = np.load("F:\\huawei_dist\\distmat_a.npy")
    #distmat_b = np.load("F:\\huawei_dist\\distmat_b.npy")
    distmat_a = np.load("distmat_a_600.npy")
    print(distmat_a)
    distmat_b = np.load("distmat_b_600.npy")
    print(distmat_b)
    #distmat_se = np.load("distmat_se_300.npy")
    #print(distmat_se)
    distmat_a_1 = distmat_a[:, transforms_gallety]
    distmat_a = distmat_a_1[transforms_query, :]
    distmat_b_1 = distmat_b[:, transforms_gallety]
    distmat_b = distmat_b_1[transforms_query, :]
    distmat = distmat + distmat_a + distmat_b
    #distmat = distmat_a + distmat_b + distmat_se
    del distmat_a_1, distmat_a , distmat_b_1, distmat_b
    '''
    print("eff6")
    distmat_chizhu = np.load("chizhu_dist.npy")
    distmat = distmat + 0.2*distmat_chizhu
    del distmat_chizhu
    '''
    '''
    distmat_se = np.load("F:\\huawei_dist\\distmat_se.npy")
    distmat_se_1 = distmat_se[:, transforms_gallety]
    distmat_se = distmat_se_1[transforms_query, :]
    distmat = distmat + distmat_se
    del distmat_se_1, distmat_se
    '''

    print(distmat.shape)
    np.save("dist_final.npy", distmat)
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    max_10_indices = indices[:, :10]
    res_dict = dict()
    for q_idx in range(num_q):
        #print(query_name[q_idx][0])
        filename = query_name[q_idx]
        max_10_files = [gallery_name[i] for i in max_10_indices[q_idx]]
        res_dict[filename] = max_10_files
    with open('submission.csv', 'w') as file:
        for k, v in res_dict.items():
            writer_string = "%s,{%s,%s,%s,%s,%s,%s,%s,%s,%s,%s}\n" % (
            k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9])
            file.write(writer_string)
    file.close()

    '''
    print(distmat.shape)
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    max_10_indices = indices[:, :20]
    res_dict = dict()
    for q_idx in range(num_q):
        # print(query_name[q_idx][0])
        filename = query_name[q_idx]
        max_10_files = [gallery_name[i] for i in max_10_indices[q_idx]]
        res_dict[filename] = max_10_files
    with open('submission_20.csv', 'w') as file:
        for k, v in res_dict.items():
            writer_string = "%s,{%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s}\n" % (
                k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16], v[17], v[18], v[19])
            file.write(writer_string)
    file.close()
    '''