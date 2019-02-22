#datasets:
DATASET_PATH = 'Dataset/'
TRAIN_PATH = 'Dataset/train.csv'
TEST_PATH = 'Dataset/test.csv'

#Is it necessary to clean train data at first?
NEED_CLEANING = True

CLEAN_WORDS_PATH = 'features/cleanwords.txt'
FASTTEXT_PATH = 'features/crawl-300d-2M.vec'

MODEL_CHECKPOINT_FOLDER = "checkpoints/"
TEMPORARY_CHECKPOINTS_PATH = 'temporary_checkpoints/'
MAX_SENTENCE_LENGTH = 350

num_classes = 6.