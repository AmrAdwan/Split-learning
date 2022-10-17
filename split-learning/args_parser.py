
def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """

    parser.add_argument('--run_name', type=str, default='Split_Learning', metavar='RN',
                        help='Wandb run name')

    # Training settings
    parser.add_argument('--model', type=str, default='resnet18', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset_id', type=str, default=4134, metavar='N',
                        help='Dataset ID from openml')


    parser.add_argument('--dataset_index_id', type=str, default=0, metavar='N',
                        help='Dataset ID from openml')


    parser.add_argument('--cut_layer', type=str, default=3, metavar='N',
                        help='Cut layer number')


    parser.add_argument('--num_ln', type=str, default=5, metavar='N',
                        help='Cut layer number')


    parser.add_argument('--data_dir', type=str, default='./data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--noise_rate', type=float, default=0.0, metavar='NR',
                        help='Corrupt rate of samples')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--data_parallel', type=int, default=0,
                        help='if distributed training')


    parser.add_argument('--client_num_in_total', type=int, default=5, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=5, metavar='NN',
                        help='number of workers')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--frequency_of_train_acc_report', type=int, default=1,
                        help='the frequency of training accuracy report')

    parser.add_argument('--frequency_of_test_acc_report', type=int, default=1,
                        help='the frequency of test accuracy report')


    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--config_id', type=int, default=0,
                        help='CoI')
    return parser
