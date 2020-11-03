from .bases import BaseImageDataset
import os.path as osp
from collections import defaultdict
import os
class huawei(BaseImageDataset):
    '''get pid,camid,imgpathname info of train,query,gallery(test)'''
    #dataset_dir = '../train_data'
    def __init__(self, root='../train_data', verbose=True, **kwargs):
        super(huawei, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.root
        print(self.dataset_dir)
        assert os.path.exists(self.dataset_dir), 'huawei dataset dir:(%s) is wrong' % self.dataset_dir
        # self.download_dataset(self.dataset_dir, self.dataset_url)
        self.train_dir = self.dataset_dir
        #self.train_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/bounding_box_train')
        #self.query_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/query')
        #self.gallery_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/bounding_box_test')

        # self.check_before_run(required_files)

        self.train = self.process_dir(self.train_dir, relabel=True)
        #self.query = self.process_dir(self.query_dir, relabel=False)
        #self.gallery = self.process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> DukeMTMC-reID loaded")
            self.print_dataset_statistics(self.train)#, self.query, self.gallery)


        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        #self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        #self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def process_dir(self, dir_path, relabel=False):
        img_paths = []
        for root,dirs,files in os.walk(self.dataset_dir):
            if len(dirs)==0:
                if len(files)!=0:
                    for file in files:
                        img_paths.append(os.path.join(root, file))
        pid_container = set()
        for img_path in img_paths:
            #print(img_path)
            pid  = img_path.split('\\')[-2]
            #pid = img_path.split('/')[-2]
            pid_container.add(pid)
        pid_container = list(pid_container)
        pid_container.sort()
        self.pid2label = {pid: label for label, pid in enumerate(pid_container)}
        #print(len(pid_container))

        data = []
        for img_path in img_paths:
            pid  = img_path.split('\\')[-2]
            #pid = img_path.split('/')[-2]
            #assert 1 <= camid <= 8
            camid = 1  # index starts from 0
            if relabel: pid = self.pid2label[pid]
            data.append((img_path, pid, camid))

        return data