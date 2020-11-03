import os
from config import default
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
import torch
import torchvision
from torchvision import datasets, transforms
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

    #train_loader, val_loader_green, val_loader_normal, num_query_green,num_query_normal, num_classes = make_dataloader(cfg)
    model = make_model(cfg, num_class=3094)
    model.load_param(cfg.TEST.WEIGHT)



    transform_test = transforms.Compose([
        transforms.Resize(cfg.INPUT.SIZE_TEST),
        # 将图像中央的高和宽均为224的正方形区域裁剪出来
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.524, 0.4943, 0.473), (0.03477, 0.03015, 0.02478))
        transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    batch_size = 128
    testset = torchvision.datasets.ImageFolder(root='../test_data_final/test_data_A', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)


    num_query = len(os.listdir("../test_data_final/test_data_A\\query"))
    query_name = testset.samples[-num_query:]
    #print(query_name[0])
    gallery_name = testset.samples[:-num_query]
    do_inference(cfg, model, test_loader, num_query, query_name, gallery_name)
