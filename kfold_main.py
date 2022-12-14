import argparse
import os
from glob import glob
import shutil

import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold

from data.dataset import LandmarkDataset
from utils.get_path import load_path
from data.data_loader import get_data_loader
from utils.get_path import get_save_kfold_model_path

from models.model.models import Model
from models.runners.training_runner import TrainingRunner

from utils.fix_seed import seed_torch
from utils.translation import str2bool


from constant import LABELS, SAVE_MODEL_NAME, TRAIN_IMAGE_PATH, TEST_IMAGE_PATH


def kfold_main_loop(
    args,
    train_img_paths,
    train_labels,
    valid_img_paths,
    valid_labels,
    test_img_paths,
    fold_num,
):

    train_dataset = LandmarkDataset(
        img_paths=train_img_paths,
        labels=train_labels,
        training=True,
        img_size=args.img_size,
        use_augmentation=True,
    )
    valid_dataset = LandmarkDataset(
        img_paths=valid_img_paths,
        labels=valid_labels,
        training=True,
        img_size=args.img_size,
        use_augmentation=False,
    )

    test_dataset = LandmarkDataset(
        img_paths=test_img_paths,
        labels=None,
        training=False,
        img_size=args.img_size,
        use_augmentation=False,
    )

    # ===========================================================================
    train_data_loader, valid_data_loader, _ = get_data_loader(
        train_dataset, valid_dataset, test_dataset, args.batch_size, args.num_workers
    )

    model = Model.get_model(args.model_name, args.__dict__).to(device)

    # ===========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.T_max, eta_min=args.eta_min
    )

    # ===========================================================================
    save_model_path, save_folder_path = get_save_kfold_model_path(
        args.save_path, SAVE_MODEL_NAME, fold_num
    )

    # ===========================================================================
    train_runner = TrainingRunner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        device=device,
        max_grad_norm=args.max_grad_norm,
    )

    # ===========================================================================
    # prev_valid_acc: float = 1e-4
    prev_valid_loss: float = 1e4
    t_loss, t_acc = [], []
    v_loss, v_acc = [], []

    for epoch in range(args.epochs):
        print(f"Epoch : {epoch + 1}")

        train_loss, train_acc = train_runner.run(
            train_data_loader,
            epoch + 1,
            mixup=args.mixup,
            mixup_epochs=args.mixup_epochs,
            cutmix=args.cutmix,
            cutmix_epochs=args.cutmix_epochs,
        )

        t_loss.append(train_loss)
        t_acc.append(train_acc)

        print(f"Train loss : {train_loss}, Train acc : {train_acc}")

        valid_loss, valid_acc = train_runner.run(
            valid_data_loader, epoch + 1, training=False, mixup=False, mixup_epochs=None
        )
        v_loss.append(valid_loss)
        v_acc.append(valid_acc)
        print(
            f"Valid loss : {valid_loss}, Valid acc : {valid_acc}, BEST: {prev_valid_loss}"
        )

        train_runner.save_graph(save_folder_path, t_loss, t_acc, v_loss, v_acc)

        if prev_valid_loss > valid_loss:
            prev_valid_loss = valid_loss
            train_runner.save_model(save_path=save_model_path)
            train_runner.save_result(
                epoch,
                save_folder_path,
                train_loss,
                valid_loss,
                train_acc,
                valid_acc,
                args,
            )


if __name__ == "__main__":
    print(len(LABELS))
    args = argparse.ArgumentParser()

    args.add_argument("--backbone", type=str, default="resnet50d")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--num_classes", type=int, default=10)
    args.add_argument("--T_max", type=int, default=10)
    args.add_argument("--eta_min", type=float, default=1e-6)
    args.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    args.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    args.add_argument("--eps", type=float, default=1e-8)
    args.add_argument("--weight_decay", type=float, default=1e-3)
    args.add_argument("--epochs", type=int, default=200)
    args.add_argument("--max_grad_norm", type=float, default=1.0)
    args.add_argument("--img_size", type=int, default=224)
    args.add_argument("--num_workers", type=int, default=4)
    args.add_argument("--save_path", type=str, default="./models/saved_model/")
    args.add_argument("--model_name", type=str, default="timm_classification")
    args.add_argument("--label_smoothing", type=float, default=0.0)
    args.add_argument("--device", type=int, default=0)
    args.add_argument("--mixup", type=str2bool, default="True")
    args.add_argument(
        "--mixup_epochs", type=int, default=180, help="epochs to train with mixup"
    )
    args.add_argument("--cutmix", type=str2bool, default="False")
    args.add_argument(
        "--cutmix_epochs", type=int, default=0, help="epochs to train with cutmix"
    )
    args.add_argument(
        "--num_folds", type=int, default=5, help="number of cross validation"
    )
    args.add_argument(
        "--pretrained", type=str2bool, default="False", help="Transfer Learning"
    )
    args = args.parse_args()

    print(args.__dict__)

    seed_torch(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===========================================================================
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    num_folder = len(glob(args.save_path + "*"))
    args.save_path = os.path.join(args.save_path, str(num_folder + 1))

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        shutil.copyfile("kfold_main.py", os.path.join(args.save_path, "kfold_main.py"))
        shutil.copyfile("data/dataset.py", os.path.join(args.save_path, "dataset.py"))
        shutil.copyfile("constant.py", os.path.join(args.save_path, "constant.py"))
    # ===========================================================================
    img_paths, labels = load_path(TRAIN_IMAGE_PATH)

    print(f"img_paths : {len(img_paths)}")
    print(f"labels: {len(labels)}")

    print(img_paths[0])

    fold_list = []
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    for train, valid in skf.split(img_paths, labels):
        fold_list.append([train, valid])
        print("train", len(train), train)
        print("valid", len(valid), valid)
        print()

    for fold_num, fold in enumerate(fold_list):
        print(f"Fold num : {str(fold_num + 1)}, fold : {fold}")
        train_img_paths = [img_paths[i] for i in fold[0]]
        train_labels = [labels[i] for i in fold[0]]

        valid_img_paths = [img_paths[i] for i in fold[1]]
        valid_labels = [labels[i] for i in fold[1]]

        kfold_main_loop(
            args,
            train_img_paths,
            train_labels,
            valid_img_paths,
            valid_labels,
            TEST_IMAGE_PATH,
            fold_num,
        )
