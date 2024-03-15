import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='Test results with different post-hoc methods',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
    parser.add_argument('--nb-run', default=3, type=int, help='Run n times, in order to compute std')
    parser.add_argument('--gpu', default='9', type=str, help='GPU id to use')
    parser.add_argument('--save-dir', default='./output', type=str, help='Output directory')
    parser.add_argument('--deit-path', default='/home/liyuting/lyt/deit_base_patch16_224-b5f2ef4d.pth', type=str,
                        help='Official DeiT checkpoints')
    ## cosine classifier
    parser.add_argument('--use-cosine', action='store_true', default=False, help='whether use cosine classifier ')
    parser.add_argument('--cos-temp', type=int, default=8, help='temperature for scaling cosine similarity')

    ## Model + optim method + data aug + loss + post-hoc
    parser.add_argument('--model-name', default='resnet18', type=str,
                        choices=['resnet18', 'resnet32', 'resnet50', 'densenet', 'wrn', 'vgg', 'vgg19bn', 'deit'],
                        help='Models name to use')
    parser.add_argument('--optim-name', default='baseline', type=str, choices=['baseline', 'sam', 'swa', 'fmfp'],
                        help='Supported methods for optimization process')
    parser.add_argument('--crl-weight', default=0.0, type=float, help='CRL loss weight')
    parser.add_argument('--mixup-weight', default=0.0, type=float, help='Mixup loss weight')

    ## Temperature Scaling
    parser.add_argument('--init-temp', default=1.5, type=float, help='Initial temperature')
    parser.add_argument('--lr-temp', default=0.01, type=float, help='Learning rate for learning temperature')
    parser.add_argument('--iters-temp', default=100, type=float, help='Max iterations for learning temperature')

    ## dataset setting
    subparsers = parser.add_subparsers(title="dataset setting", dest="subcommand")
    Cifar10 = subparsers.add_parser("Cifar10",
                                    description='Dataset parser for training on Cifar10',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    help="Dataset parser for training on Cifar10")
    Cifar10.add_argument('--data-name', default='cifar10', type=str, help='Dataset name')
    Cifar10.add_argument("--train-dir", type=str, default='./data/CIFAR10/train',
                         help="Cifar10 train directory, train_an_0.3 and train_sn_0.2")
    Cifar10.add_argument("--val-dir", type=str, default='./data/CIFAR10/val', help="Cifar10 val directory")
    Cifar10.add_argument("--test-dir", type=str, default='./data/CIFAR10/test', help="Cifar10 val directory")
    Cifar10.add_argument("--corruption-dir", type=str, default='./data', help="Cifar10 val directory")
    Cifar10.add_argument("--nb-cls", type=int, default=10, help="number of classes in Cifar10")
    Cifar10.add_argument("--imb-factor", type=float, default=1.0, help="imbalance rate in Cifar10")

    Cifar100 = subparsers.add_parser("Cifar100",
                                     description='Dataset parser for training on Cifar100',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     help="Dataset parser for training on Cifar100")
    Cifar100.add_argument('--data-name', default='cifar100', type=str, help='Dataset name')
    Cifar100.add_argument("--train-dir", type=str, default='./data/CIFAR100/train',
                          help="Cifar100 train directory")
    Cifar100.add_argument("--val-dir", type=str, default='./data/CIFAR100/val',
                          help="Cifar100 val directory")
    Cifar100.add_argument("--test-dir", type=str, default='./data/CIFAR100/test',
                          help="Cifar100 val directory")
    Cifar100.add_argument("--nb-cls", type=int, default=100, help="number of classes in Cifar100")
    Cifar100.add_argument("--imb-factor", type=float, default=1.0, help="imbalance rate in Cifar100")

    Cifar10_LT = subparsers.add_parser("Cifar10_LT",
                                       description='Dataset parser for training on Cifar10',
                                       add_help=True,
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       help="Dataset parser for training on Cifar10 Long Tail")
    Cifar10_LT.add_argument('--data-name', default='cifar10_LT', type=str, help='Dataset name')
    Cifar10_LT.add_argument("--train-dir", type=str, default='./data/CIFAR10_LT/train',
                            help="Cifar10 train directory, train_an_0.3 and train_sn_0.2")
    Cifar10_LT.add_argument("--val-dir", type=str, default='./data/CIFAR10_LT/test', help="Cifar10_LT val directory")
    Cifar10_LT.add_argument("--test-dir", type=str, default='./data/CIFAR10_LT/test', help="Cifar10_LT val directory")
    Cifar10_LT.add_argument("--nb-cls", type=int, default=10, help="number of classes in Cifar10_LT")
    Cifar10_LT.add_argument("--imb-factor", type=float, default=0.1, help="imbalance rate in Cifar10-LT")

    Cifar100_LT = subparsers.add_parser("Cifar100_LT",
                                        description='Dataset parser for training on Cifar100',
                                        add_help=True,
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                        help="Dataset parser for training on Cifar100 Long Tail")
    Cifar100_LT.add_argument('--data-name', default='cifar100_LT', type=str, help='Dataset name')
    Cifar100_LT.add_argument("--train-dir", type=str, default='./data/CIFAR100_LT/train',
                             help="Cifar100_LT train directory")
    Cifar100_LT.add_argument("--val-dir", type=str, default='./data/CIFAR100_LT/test', help="Cifar100_LT val directory")
    Cifar100_LT.add_argument("--test-dir", type=str, default='./data/CIFAR100_LT/test',
                             help="Cifar100_LT val directory")
    Cifar100_LT.add_argument("--nb-cls", type=int, default=100, help="number of classes in Cifar100_LT")
    Cifar100_LT.add_argument("--imb-factor", type=float, default=0.1, help="imbalance rate in Cifar100-LT")

    Cifar10_LT_50 = subparsers.add_parser("Cifar10_LT_50",
                                          description='Dataset parser for training on Cifar10',
                                          add_help=True,
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                          help="Dataset parser for training on Cifar10 Long Tail")
    Cifar10_LT_50.add_argument('--data-name', default='cifar10_LT', type=str, help='Dataset name')
    Cifar10_LT_50.add_argument("--train-dir", type=str, default='./data/CIFAR10_LT/train',
                               help="Cifar10 train directory, train_an_0.3 and train_sn_0.2")
    Cifar10_LT_50.add_argument("--val-dir", type=str, default='./data/CIFAR10_LT/test', help="Cifar10_LT val directory")
    Cifar10_LT_50.add_argument("--test-dir", type=str, default='./data/CIFAR10_LT/test',
                               help="Cifar10_LT val directory")
    Cifar10_LT_50.add_argument("--nb-cls", type=int, default=10, help="number of classes in Cifar10_LT")
    Cifar10_LT_50.add_argument("--imb-factor", type=float, default=0.02, help="imbalance rate in Cifar10-LT")

    Cifar100_LT_50 = subparsers.add_parser("Cifar100_LT_50",
                                           description='Dataset parser for training on Cifar100',
                                           add_help=True,
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                           help="Dataset parser for training on Cifar100 Long Tail")
    Cifar100_LT_50.add_argument('--data-name', default='cifar100_LT', type=str, help='Dataset name')
    Cifar100_LT_50.add_argument("--train-dir", type=str, default='./data/CIFAR100_LT/train',
                                help="Cifar100_LT train directory")
    Cifar100_LT_50.add_argument("--val-dir", type=str, default='./data/CIFAR100_LT/test',
                                help="Cifar100_LT val directory")
    Cifar100_LT_50.add_argument("--test-dir", type=str, default='./data/CIFAR100_LT/test',
                                help="Cifar100_LT val directory")
    Cifar100_LT_50.add_argument("--nb-cls", type=int, default=100, help="number of classes in Cifar100_LT")
    Cifar100_LT_50.add_argument("--imb-factor", type=float, default=0.02, help="imbalance rate in Cifar100-LT")

    Cifar10_LT_100 = subparsers.add_parser("Cifar10_LT_100",
                                           description='Dataset parser for training on Cifar10',
                                           add_help=True,
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                           help="Dataset parser for training on Cifar10 Long Tail")
    Cifar10_LT_100.add_argument('--data-name', default='cifar10_LT', type=str, help='Dataset name')
    Cifar10_LT_100.add_argument("--train-dir", type=str, default='./data/CIFAR10_LT/train',
                                help="Cifar10 train directory, train_an_0.3 and train_sn_0.2")
    Cifar10_LT_100.add_argument("--val-dir", type=str, default='./data/CIFAR10_LT/test',
                                help="Cifar10_LT val directory")
    Cifar10_LT_100.add_argument("--test-dir", type=str, default='./data/CIFAR10_LT/test',
                                help="Cifar10_LT val directory")
    Cifar10_LT_100.add_argument("--nb-cls", type=int, default=10, help="number of classes in Cifar10_LT")
    Cifar10_LT_100.add_argument("--imb-factor", type=float, default=0.01, help="imbalance rate in Cifar10-LT")

    Cifar100_LT_100 = subparsers.add_parser("Cifar100_LT_100",
                                            description='Dataset parser for training on Cifar100',
                                            add_help=True,
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                            help="Dataset parser for training on Cifar100 Long Tail")
    Cifar100_LT_100.add_argument('--data-name', default='cifar100_LT', type=str, help='Dataset name')
    Cifar100_LT_100.add_argument("--train-dir", type=str, default='./data/CIFAR100_LT/train',
                                 help="Cifar100_LT train directory")
    Cifar100_LT_100.add_argument("--val-dir", type=str, default='./data/CIFAR100_LT/test',
                                 help="Cifar100_LT val directory")
    Cifar100_LT_100.add_argument("--test-dir", type=str, default='./data/CIFAR100_LT/test',
                                 help="Cifar100_LT val directory")
    Cifar100_LT_100.add_argument("--nb-cls", type=int, default=100, help="number of classes in Cifar100_LT")
    Cifar100_LT_100.add_argument("--imb-factor", type=float, default=0.01,
                                 help="imbalance rate in Cifar100-LT imbalance factor = 100")


    Animal10N = subparsers.add_parser("Animal10N",
                                      description='Dataset parser for training on Animal10N',
                                      add_help=True,
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                      help="Dataset parser for training on Animal10N")
    Animal10N.add_argument('--data-name', default='Animal10N', type=str, help='Dataset name')
    Animal10N.add_argument("--train-dir", type=str, default='./data/Animal10N/train', help="Animal10N train directory")
    Animal10N.add_argument("--val-dir", type=str, default='./data/Animal10N/test', help="Animal10N test directory")
    Animal10N.add_argument("--test-dir", type=str, default='./data/Animal10N/test', help="Animal10N test directory")
    Animal10N.add_argument("--nb-cls", type=int, default=10, help="number of classes in Animal10N test")
    Animal10N.add_argument("--imb-factor", type=float, default=1.0, help="imbalance rate in Animal10N")

    Clothing1M = subparsers.add_parser("Clothing1M",
                                       description='Dataset parser for training on Clothing1M',
                                       add_help=True,
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       help="Dataset parser for training on Clothing1M")
    Clothing1M.add_argument('--data-name', default='Clothing1M', type=str, help='Dataset name')
    Clothing1M.add_argument("--train-dir", type=str, default='./data/Clothing1M/noisy_rand_subtrain',
                            help="Clothing1M train directory")
    Clothing1M.add_argument("--val-dir", type=str, default='./data/Clothing1M/clean_val',
                            help="Clothing1M val directory")
    Clothing1M.add_argument("--test-dir", type=str, default='./data/Clothing1M/clean_test',
                            help="Clothing1M test directory")
    Clothing1M.add_argument("--nb-cls", type=int, default=14, help="number of classes in Clothing1M test")
    Clothing1M.add_argument("--imb-factor", type=float, default=1.0, help="imbalance rate in Clothing1M")

    Food101N = subparsers.add_parser("Food101N",
                                     description='Dataset parser for training on Food101N',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     help="Dataset parser for training on Food101N")
    Food101N.add_argument('--data-name', default='Food101N', type=str, help='Dataset name')
    Food101N.add_argument("--train-dir", type=str, default='./data/Food101N/train', help="Food101N train directory")
    Food101N.add_argument("--val-dir", type=str, default='./data/Food101N/test', help="Food101N val directory")
    Food101N.add_argument("--test-dir", type=str, default='./data/Food101N/test', help="Food101N test directory")
    Food101N.add_argument("--nb-cls", type=int, default=101, help="number of classes in Food101N test")
    Food101N.add_argument("--imb-factor", type=float, default=1.0, help="imbalance rate in Food101N")

    TinyImgNet = subparsers.add_parser("TinyImgNet",
                                       description='Dataset parser for training on TinyImgNet',
                                       add_help=True,
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       help="Dataset parser for training on TinyImgNet")
    TinyImgNet.add_argument('--data-name', default='TinyImgNet', type=str, help='Dataset name')
    TinyImgNet.add_argument("--train-dir", type=str, default='/home/liyuting/Uncertainty/data/tinyImageNet/train',
                            help="TinyImgNet train directory")
    TinyImgNet.add_argument("--val-dir", type=str, default='/home/liyuting/Uncertainty/data/tinyImageNet/val', help="TinyImgNet val directory")
    TinyImgNet.add_argument("--test-dir", type=str, default='/home/liyuting/Uncertainty/data/tinyImageNet/test', help="TinyImgNet val directory")
    TinyImgNet.add_argument("--nb-cls", type=int, default=200, help="number of classes in TinyImgNet")
    TinyImgNet.add_argument("--imb-factor", type=float, default=1.0, help="imbalance rate in TinyImgNet")
    return parser.parse_args()

