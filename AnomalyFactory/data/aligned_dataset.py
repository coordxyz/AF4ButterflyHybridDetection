import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize, \
    get_transform_base1, get_transform_tps, get_transform_base2,get_transform_base3, get_local
from data.image_folder import make_dataset
from PIL import Image, ImageChops
import torch
import imageio
from skimage import util, feature
from skimage.color import rgb2gray
import numpy as np
import pdb,random,cv2,glob
import albumentations
import torchvision.transforms as transforms
from options.test_options import TestOptions
from models.models import create_model
import util.util as deepsim_util

#---------------------------------------------------
#AF:AnomalyFactory use Single edge reference
class BootAF_I2DE_trainDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot.split('List')[0]   

        ## reference: color image
        Atxt_data = open(opt.dataroot + 'train_color_gt.txt', 'r')
        self.ref_rgb_paths = Atxt_data.readlines()

        ## target: edge image
        Atxt_data = open(opt.dataroot + 'train_edges.txt', 'r')
        self.tag_edge_paths = Atxt_data.readlines()

        self.dataset_size = len(self.tag_edge_paths)

    def adjust_input_size(self, opt):
        """
        change image size once when loading the image.
        :return:
        """
        # TODO: if we use instance map, support resize method = NEAREST
        ow, oh = 256, 256

        # ow, oh = self.A.size
        # # for cuda memory capacity
        # if max(ow, oh) > 1000:
        #     ow, oh = ow // 2, oh // 2

        if opt.resize_or_crop == 'none' or opt.resize_or_crop == "crop":
            # input size should fit architecture params
            base = float(2 ** opt.n_downsample_global)
            if opt.netG == 'local':
                base *= (2 ** opt.n_local_enhancers)
            new_h = int(round(oh / base) * base)
            new_w = int(round(ow / base) * base)

        elif 'resize' in opt.resize_or_crop:
            new_h, new_w = opt.loadSize, opt.loadSize

        elif 'scale_width' in opt.resize_or_crop:
            if ow != opt.loadSize:
                new_w = opt.loadSize
                new_h = int(opt.loadSize * oh / ow)

        self.ref_rgb = self.ref_rgb.resize((new_w, new_h), Image.BICUBIC)
        self.tag_edge = self.tag_edge.resize((new_w, new_h), Image.BICUBIC)

    def __getitem__(self, index):
        self.ref_rgb = Image.open(self.root + self.ref_rgb_paths[index].split('\n')[0]).convert('RGB')
        self.tag_edge = Image.open(self.root + self.tag_edge_paths[index].split('\n')[0]).convert('RGB')
        
        self.adjust_input_size(self.opt) 

        use_reflect=random.uniform(0,1)>0.5
        if use_reflect:
            ow, oh = self.tag_edge.size
            tw = random.randint(10,30)
            bw = random.randint(10,30)
            lh = random.randint(10,30)
            rh = random.randint(10,30)

            tmp_np = np.array(self.ref_rgb)
            tmp_np = cv2.copyMakeBorder(tmp_np, tw, bw, lh, rh, 
                        cv2.BORDER_REFLECT)
            tmp_np = cv2.resize(tmp_np, (ow, oh), interpolation=cv2.INTER_CUBIC) 
            self.ref_rgb = Image.fromarray(tmp_np)

            tmp_np = np.array(self.tag_edge)
            tmp_np = cv2.copyMakeBorder(tmp_np, tw, bw, lh, rh, 
                        cv2.BORDER_CONSTANT, value=(255,255,255))
            tmp_np = cv2.resize(tmp_np, (ow, oh), interpolation=cv2.INTER_CUBIC) 
            self.tag_edge = Image.fromarray(tmp_np)
            del tmp_np
                    
        ##---BASE transform for all
        params_base = get_params(self.opt, self.tag_edge.size, self.tag_edge)
        transform_base1 = get_transform_base1(self.opt, params_base, is_primitive=True, 
                                is_edges=self.opt.primitive == "edges", good_as_seg=False)    
        transform_base2 = get_transform_base2(self.opt, params_base, is_primitive=True, 
                                is_edges=self.opt.primitive == "edges", good_as_seg=False)

        ##---EDGE TPS wrap
        params_tps_edge = get_params(self.opt, self.tag_edge.size, self.tag_edge)
        transform_tps_edge = get_transform_tps(self.opt, params_tps_edge, is_primitive=True, 
                                is_edges=self.opt.primitive == "edges", good_as_seg=False)
        input_wrapped_edge = transform_base2(transform_tps_edge(transform_base1(self.tag_edge)))
                
        ##---RGB TPS wrap
        params_tps_rgb = get_params(self.opt, self.ref_rgb.size, self.ref_rgb)
        transform_tps_rgb = get_transform_tps(self.opt, params_tps_rgb, is_primitive=True, 
                                is_edges=self.opt.primitive == "edges", good_as_seg=False)
        input_wrapped_rgb = transform_base2(transform_tps_rgb(transform_base1(self.ref_rgb)))
        
        gt_wrapped_edge = transform_base2(transform_tps_edge(transform_base1(self.ref_rgb)))
        input_tensor = torch.cat((input_wrapped_edge,  input_wrapped_rgb), dim=0)

        clip_vfeatures = 0

        input_dict = {'label': input_tensor, 'inst': 0, 
                    'real_edge': gt_wrapped_edge, 
                    'feat': clip_vfeatures, 
                    'path': self.tag_edge_paths[index]} 

        return input_dict

    def __len__(self):
        return len(self.tag_edge_paths)

    def name(self):
        return 'BootAF_I2DE_trainDataset'

class BootAF_I2DE_testDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot.split('List')[0]   

        ## reference: color image
        Atxt_data = open(opt.dataroot + 'test_color_ref.txt', 'r')
        self.ref_rgb_paths = Atxt_data.readlines()
        
        ## target: edge image
        Atxt_data = open(opt.dataroot + 'test_edges.txt', 'r')
        self.tag_edge_paths = Atxt_data.readlines()

        self.dataset_size = len(self.tag_edge_paths)

    def adjust_input_size(self, opt):
        """
        change image size once when loading the image.
        :return:
        """
        # TODO: if we use instance map, support resize method = NEAREST
        ow, oh = 256, 256

        # ow, oh = self.A.size
        # # for cuda memory capacity
        # if max(ow, oh) > 1000:
        #     ow, oh = ow // 2, oh // 2

        if opt.resize_or_crop == 'none' or opt.resize_or_crop == "crop":
            # input size should fit architecture params
            base = float(2 ** opt.n_downsample_global)
            if opt.netG == 'local':
                base *= (2 ** opt.n_local_enhancers)
            new_h = int(round(oh / base) * base)
            new_w = int(round(ow / base) * base)

        elif 'resize' in opt.resize_or_crop:
            new_h, new_w = opt.loadSize, opt.loadSize

        elif 'scale_width' in opt.resize_or_crop:
            if ow != opt.loadSize:
                new_w = opt.loadSize
                new_h = int(opt.loadSize * oh / ow)

        self.ref_rgb = self.ref_rgb.resize((new_w, new_h), Image.BICUBIC)
        self.tag_edge = self.tag_edge.resize((new_w, new_h), Image.BICUBIC)

    def __getitem__(self, index):
        self.ref_rgb = Image.open(self.root + self.ref_rgb_paths[index].split('\n')[0]).convert('RGB')
        self.tag_edge = Image.open(self.root + self.tag_edge_paths[index].split('\n')[0]).convert('RGB')                
        gt_rgb_name = self.root + self.tag_edge_paths[index].split('\n')[0].replace('/pidinet_edges/','/good/').replace('png','jpg')
        gt_rgb = Image.open(gt_rgb_name).convert('RGB')
        self.adjust_input_size(self.opt) 

        ##---BASE transform for all
        params_base = get_params(self.opt, self.ref_rgb.size, self.ref_rgb)
        transform_base1 = get_transform_base1(self.opt, params_base, is_primitive=True, 
                                is_edges=self.opt.primitive == "edges", good_as_seg=False)    
        transform_base2 = get_transform_base2(self.opt, params_base, is_primitive=True, 
                                is_edges=self.opt.primitive == "edges", good_as_seg=False)

        ##---EDGE TPS wrap
        params_tps_edge = get_params(self.opt, self.tag_edge.size, self.tag_edge)
        transform_tps_edge = get_transform_tps(self.opt, params_tps_edge, is_primitive=True, 
                                is_edges=self.opt.primitive == "edges", good_as_seg=False)
        input_wrapped_edge = transform_base2(transform_tps_edge(transform_base1(self.tag_edge)))

        ##---RGB TPS wrap
        params_tps_rgb = get_params(self.opt, self.ref_rgb.size, self.ref_rgb)
        transform_tps_rgb = get_transform_tps(self.opt, params_tps_rgb, is_primitive=True, 
                                is_edges=self.opt.primitive == "edges", good_as_seg=False)
        input_wrapped_rgb = transform_base2(transform_tps_rgb(transform_base1(self.ref_rgb)))        
        
        gt_wrapped_edge = transform_base2(transform_tps_edge(transform_base1(gt_rgb)))
        input_tensor = torch.cat((input_wrapped_edge, input_wrapped_rgb), dim=0)

        clip_vfeatures = 0

        input_dict = {'label': input_tensor, 'inst': 0, 
                    'real_edge': gt_wrapped_edge,  
                    'feat': clip_vfeatures, 
                    'refpath': self.ref_rgb_paths[index],
                    'path':self.tag_edge_paths[index]}  

        return input_dict

    def __len__(self):
        return len(self.tag_edge_paths)

    def name(self):
        return 'BootAF_I2DE_testDataset'

#----------------------------------------------------------





