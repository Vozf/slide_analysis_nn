from slide_analysis_nn.train import Train
from slide_analysis_nn.train.datasets_preparation import DatasetPreparation


def main():
    dataset_preparation = DatasetPreparation()
    dataset_preparation.create_dataset()
    # dataset_preparation.generate_new_train_test_split_from_full_dataset()
    #
    train = Train()
    train.start_training()


if __name__ == '__main__':
    main()
