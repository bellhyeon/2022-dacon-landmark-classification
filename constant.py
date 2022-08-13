import pandas as pd


TRAIN_IMAGE_PATH = "./dataset/train"
TEST_IMAGE_PATH = "./dataset/test"

TRAIN_CSV_PATH = "./dataset/train.csv"
TEST_CSV_PATH = "./dataset/test.csv"
SUBMISSION_CSV_PATH = "./dataset/sample_submission.csv"

train_csv = pd.read_csv(TRAIN_CSV_PATH)

LABELS = train_csv.label.tolist()

SAVE_MODEL_NAME = "model.pt"
