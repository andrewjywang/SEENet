import tensorflow as tf
from src.args import Args
from src.dataloader import DataLoader
from src.seenet import SEENET


def main():
    args = Args().parse()
    loader = DataLoader(args.train_real_data_dir, args.train_flow_data_dir,
                        video_length=args.video_length, batch_size=args.batch_size)
    test_loader = DataLoader(args.test_real_data_dir, args.test_flow_data_dir,
                             video_length=args.video_length, batch_size=args.batch_size)
    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        gan = SEENET(args, sess, graph)
        if args.test:
            gan.test_all(test_loader)
        else:
            gan.train(loader, test_loader)


if __name__ == "__main__":
    main()
