import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch,pdb,random

import lpips
import itertools
import numpy as np
from tqdm import tqdm
import copy, cv2, time


seed = 202409 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def test():
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.resize_or_crop = "none"

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    device = opt.gpu_ids[0] 


    # test
    if not opt.engine and not opt.onnx:
        model = create_model(opt)
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)

        if opt.verbose:
            print(model)
    else:
        from run_engine import run_trt_engine, run_onnx

    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i >= opt.how_many:
                break

            if opt.data_type == 16:
                data['label'] = data['label'].half()
                data['inst']  = data['inst'].half()
            elif opt.data_type == 8:
                data['label'] = data['label'].uint8()
                data['inst']  = data['inst'].uint8()
            if opt.export_onnx:
                print ("Exporting to ONNX: ", opt.export_onnx)
                assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
                torch.onnx.export(model, [data['label'], data['inst']],
                                opt.export_onnx, verbose=True)
                exit(0)
            minibatch = 1
            if opt.engine:
                generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
            elif opt.onnx:
                generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
            else:
                start_time = time.process_time()
                fake_edge, edge_ano = model.inference(data['label'], None,None,None, data['feat'])
                end_time = time.process_time()

            visuals = OrderedDict([('fakeE', util.tensor2im(fake_edge.data[0]))]) 

            img_path = data['path']
            ref_path = data['refpath']
            
            print('process image... {}, {}seconds'.format(img_path, end_time-start_time))
            ## visualizer.save_images(webpage, visuals, img_path)#org
            # visualizer.save_separate_images(webpage, visuals, img_path,i)

            image_dir = webpage.get_image_dir()
            visualizer.save_all_images(image_dir, visuals, img_path, i, ref_path)
            # pdb.set_trace()
    webpage.save()

    if opt.vid_mode:
        util.frames_to_vid(webpage)

#----------------------------------------------------------------------------
def get_opt(testpath, use_edgemask):
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.resize_or_crop = "none"
    opt.dataroot = testpath
    opt.use_edgemask = use_edgemask    
    return opt

def init_model(opt):
    if not opt.engine and not opt.onnx:
        model = create_model(opt)
        # model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)

        if opt.verbose:
            print(model)
    else:
        from run_engine import run_trt_engine, run_onnx
    return model

def update_dataloader(opt):
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


if __name__ == "__main__":
    #single folder test
    test() 

#----------------------------------------------------------------------------
