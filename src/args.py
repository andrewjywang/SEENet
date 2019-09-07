import argparse


class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Run commands")
        self.parser.add_argument('--batch_size', default=2, type=int, help="Batch size")
        self.parser.add_argument('--motion_dim', default=128, type=int, help="motion code dimintion")
        self.parser.add_argument('--content_dim', default=128, type=int, help="content code dimention")
        self.parser.add_argument('--image_channel', default=3, type=int, help="image channel ")
        self.parser.add_argument('--image_size_h', default=128, type=int, help="Horizontal image size ")
        self.parser.add_argument('--image_size_v', default=128, type=int, help="Vertical image size ")
        self.parser.add_argument('--epoch_num', default=2000, type=int, help="Epoch number")
        self.parser.add_argument('--video_length', default=30, type=int, help="video frame length")
        self.parser.add_argument('--pred_num', default=20, type=int, help="predict frame number")
        self.parser.add_argument('--num_layers', default=2, type=int, help="number of LSTM layers")
        self.parser.add_argument('--hidden_size', default=64, type=int, help="LSTM cell hidden size ")
        self.parser.add_argument('--log_dir', default='./log/SEENET-KTH/', type=str, help="log save location")
        self.parser.add_argument('--pretrain_dir', default='../pretrained/', type=str, help="model save location")
        self.parser.add_argument('--sample_dir', default='./samples/SEENET-KTH/', type=str,
                                 help="sample save location")
        self.parser.add_argument('--train_real_data_dir', default='../data/KTH', type=str,
                                 help="trainning real data location")
        self.parser.add_argument('--train_flow_data_dir', default='../data/KTH-flow', type=str,
                                 help="trainning flow data location")
        self.parser.add_argument('--test_real_data_dir', default='../data/KTH_test', type=str,
                                 help="test real data location")
        self.parser.add_argument('--test_flow_data_dir', default='../data/KTH-flow_test', type=str,
                                 help="test flow data location")
        self.parser.add_argument('--lr_c', default=0.00001, type=float, help='content learning rate')
        self.parser.add_argument('--lr_m', default=0.00001, type=float, help='motion learning rate')
        self.parser.add_argument('--lr_g', default=0.00001, type=float, help='generator learning rate')
        self.parser.add_argument('--lr_d', default=0.000001, type=float, help='discriminator learning rate')
        self.parser.add_argument('--alpha', default=1.0, type=float, help='alpha')
        self.parser.add_argument('--beta', default=1.0, type=float, help='beta')
        self.parser.add_argument('--gamma', default=1.0, type=float, help='gamma')
        self.parser.add_argument('--delta', default=1.0, type=float, help='delta')
        self.parser.add_argument('--phi', default=1.0, type=float, help='phi')
        self.parser.add_argument('--chi', default=0.0001, type=float, help='chi')
        self.parser.add_argument('--margin', default=1.0, type=float, help='contrastive loss margin')
        self.parser.add_argument('--real_label', default=0.9, type=float, help='real label')
        self.parser.add_argument('--fake_label', default=0.1, type=float, help='fake label')
        self.parser.add_argument('--train_feature', default=False, type=bool, help='train feature or not')
        self.parser.add_argument('--test', default=True, type=bool, help='train feature or not')

    def parse(self):
        args = self.parser.parse_args()
        return args
