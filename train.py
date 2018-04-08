from train import Train
from train.datasets_preparation import DatasetPreparation


def main():
    dataset_preparation = DatasetPreparation()
    dataset_preparation.populate_prepared_datasets()
    #
    train = Train()
    train.start_training()


if __name__ == '__main__':
    main()
