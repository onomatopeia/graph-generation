from main import main_training
from evaluate import main_evaluation
from args import Args


if __name__ == '__main__':
    args = Args()
    main_training(args)
    main_evaluation(args)