import torch, pdb

def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import AnomalyFactoryPix2PixHDModel, AFInferenceModel
        if opt.isTrain:
            model = AnomalyFactoryPix2PixHDModel()
        else:
            model = AFInferenceModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    # pdb.set_trace()
    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
