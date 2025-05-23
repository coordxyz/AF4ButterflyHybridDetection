import numpy as np
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pdb
# from .loss import FocalLoss

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

##-------------------
# AnomalyFactory Generator
##-------------------
class AnomalyFactoryPix2PixHDModel(BaseModel):
    def name(self):
        return 'AnomalyFactoryPix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        if self.train_mode=='BootAF_I2DE':
            flags = (True, use_gan_feat_loss, use_vgg_loss, True, True) 
            def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):                
                return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f] 
        else:
            flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True)
            def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, g_mask): 
                return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake, g_mask),flags) if f] 
        return loss_filter
    
    def initialize(self, opt):
        self.use_LR_fake = opt.use_LR_fake
        downsample4 = [
            nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False),
            nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False),
            nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False),
            nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        ]
        self.downsample4 = nn.Sequential(*downsample4)

        self.train_mode = opt.train_mode
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.encoder_nc 
        self.input_nc = input_nc 

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num                  

        self.netG = networks.define_G(netG_input_nc, opt.decoder_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids,train_mode=self.train_mode, 
                                      use_clipvfeat=opt.use_clipvfeat, use_LR_fake=opt.use_LR_fake)        

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = opt.input_nc + opt.output_nc * 1  
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, \
                                opt.n_layers_D, opt.norm, use_sigmoid, \
                                opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        ##---AM: load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
                   
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                
            # Names so we can breakout loss
            if self.train_mode=='BootAF_I2DE':
                self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake')
            else:
                ## add loss mask
                self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake', 'G_MASK')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            
            ##---optimizer G 
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            ##---optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input_for_test(self, label_map, inst_map=None, \
            real_edge=None, feat_map=None, edgemask=None,  infer=False):  
        if self.opt.label_nc == 0:
            if self.gpu_ids:
                input_label = label_map.data.to(torch.device(self.gpu_ids[0]))
            else:
                input_label = label_map.data
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
            if self.gpu_ids:
                input_label = input_label.to(torch.device(self.gpu_ids[0]))
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            if self.gpu_ids:
                inst_map = inst_map.data.to(torch.device(self.gpu_ids[0]))
            else:
                inst_map = inst_map.data
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)  

        with torch.no_grad():       
            input_label = Variable(input_label)

        # real images for training
        if real_edge is not None:
            if self.gpu_ids:
                real_edge = Variable(real_edge.data.to(torch.device(self.gpu_ids[0])))
            else:
                real_edge = Variable(real_edge.data)

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                if self.gpu_ids:
                    feat_map = Variable(feat_map.data.to(torch.device(self.gpu_ids[0])))
                else:
                    feat_map = Variable(feat_map.data)
            if self.opt.label_feat:
                if self.gpu_ids:
                    inst_map = label_map.to(torch.device(self.gpu_ids[0]))
                else:
                    inst_map = label_map

        # edgemask
        if self.opt.use_edgemask and edgemask is not None:
            if self.gpu_ids:
                edgemask = Variable(edgemask.data.to(torch.device(self.gpu_ids[0])))
            else:
                edgemask = Variable(edgemask.data)

        return input_label, inst_map, real_edge, feat_map, edgemask 

    def encode_input(self, label_map, inst_map=None, real_edge=None,  \
            feat_map=None, edgemask=None,  infer=False):  
        if self.opt.label_nc == 0:
            if self.gpu_ids:
                input_label = label_map.data.cuda()
            else:
                input_label = label_map.data
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
            if self.gpu_ids:
                input_label = input_label.cuda()
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            if self.gpu_ids:
                inst_map = inst_map.data.cuda()
            else:
                inst_map = inst_map.data
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)  

        # input_label = Variable(input_label, volatile=infer) #zy:volatile was removed and now has no effect
        with torch.no_grad():       
            input_label = Variable(input_label)

        # real images for training
        if real_edge is not None:
            if self.gpu_ids:
                real_edge = Variable(real_edge.data.cuda())
            else:
                real_edge = Variable(real_edge.data)

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                if self.gpu_ids:
                    feat_map = Variable(feat_map.data.cuda())
                else:
                    feat_map = Variable(feat_map.data)
            if self.opt.label_feat:
                if self.gpu_ids:
                    inst_map = label_map.cuda()
                else:
                    inst_map = label_map

        # edgemask
        if self.opt.use_edgemask and edgemask is not None:
            if self.gpu_ids:
                edgemask = Variable(edgemask.data.cuda())
            else:
                edgemask = Variable(edgemask.data)

        return input_label, inst_map, real_edge, feat_map, edgemask 

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1) 
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, edge, feat, edgemask=None, infer=False):
        # Encode Inputs
        if self.isTrain:
            input_label, inst_map, real_edge, feat_map, edgemask = self.encode_input(label, inst, edge,  feat, edgemask)  
        else:
            input_label, inst_map, real_edge, feat_map, edgemask = self.encode_input_for_test(label, inst, edge,  feat, edgemask)  

        # Fake Generation
        input_concat = input_label

        fake_edge, edge_ano = self.netG.forward(input_concat, feat_map, edgemask)
            
        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_edge, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, real_edge)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label,fake_edge), dim=1)) 
        loss_G_GAN = self.criterionGAN(pred_fake, True)                
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat = loss_G_GAN_Feat + D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat 
            
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_edge, real_edge, edgemask) * self.opt.lambda_feat  

        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake), \
                    None if not infer else fake_edge, edge_ano]

    def inference(self, label, inst=None, real_edge=None, real_depth=None, feat=None,fusion_mask=None):
        # Encode Inputs        
        inst = Variable(inst) if inst is not None else None
        real_edge = Variable(real_edge) if real_edge is not None else None
        real_depth = Variable(real_depth) if real_depth is not None else None
        feat = Variable(feat) if feat is not None else None
        # fusion_mask = Variable(fusion_mask) if fusion_mask is not None else None

        input_label, inst_map, real_edge, clip_vfeatures,fusion_mask = self.encode_input_for_test(Variable(label), inst, \
                                                                            real_edge, feat_map=feat,edgemask=fusion_mask, infer=True)

        # Fake Generation
        if self.use_features:
            if self.opt.use_encoded_image:
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # sample clusters from precomputed features             
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)                        
        else:
            input_concat = input_label        
           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_edge, edge_ano = self.netG.forward(input_concat, clip_vfeatures,fusion_mask)
        else:
            fake_edge, edge_ano = self.netG.forward(input_concat, clip_vfeatures,fusion_mask)
        
        return fake_edge, edge_ano

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        if self.gpu_ids:
            # image = Variable(image.to(torch.device(self.gpu_ids[0])), volatile=True)
            image = Variable(image.cuda(), volatile=True)
        else:
            image = Variable(image, volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        if self.gpu_ids:
            # feat_map = self.netE.forward(image, inst.to(torch.device(self.gpu_ids[0])))
            feat_map = self.netE.forward(image, inst.cuda())
        else:
            feat_map = self.netE.forward(image, inst)
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        

        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr

        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class AFInferenceModel(AnomalyFactoryPix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)       