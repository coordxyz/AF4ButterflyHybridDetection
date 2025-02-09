from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=300, help='how many test images to run')
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")

        self.parser.add_argument("--vid_mode", type=int, default=0, help="1 if input is video and desired output is video")
        self.parser.add_argument("--vid_input", type=int, default=0,
                                 help="1 if input is video and desired output is video")
        self.isTrain = False
        self.parser.add_argument('--test_mode', type=str, choices=['BootAF','BootAF_I2DE'])    
        self.parser.add_argument('--refImgExt', type=str, default='png', choices=['png', 'jpg', 'JPG'])    
        self.parser.add_argument('--use_LR_fake', action='store_true', help='use low resoution fake result')

        ###zy:for online AFmodel inference, not really use
        self.parser.add_argument('--save_latest_freq', type=int, default=20000, help='frequency of saving the latest results')
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--mask_loss_weight', type=int, default=1, help='zy: weight of edge mask loss, should also add to testOptions')    
        self.parser.add_argument('--train_mode', type=str, choices=['BootAF','BootAF_I2DE'])    

