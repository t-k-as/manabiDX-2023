import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import albumentations
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from skimage import io
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
import seaborn as sns
from mylib import (
    MyDataset,
    get_model,
    plt_result,
    get_f1,
    predict_model,
    train_model_custom,
    For_Submission_Datasets,
    data_transforms,
    Focal_MultiLabel_Loss,
)
import tqdm
import os

def main():

    # デバイスの指定（CUDA）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    #パラメータの設定
    SEED = 123
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01   #ADAM:0.001, SGD:0.01
    EPOCH = 100
    NUM_WORKERS = 6
    PARAM_DIR = "./params"
    TRAIN_DIR = "./cut/train/"
    TEST_DIR = "./cut/train/"
    IS_SAVED = False
    DRYRUN = False
    MODEL_NAME = "resnet101" #resnet18, resnet50, resnet101, resnet152, efficientnet_v2_l
    IS_PRETRAINED = True
    MULTI_CLASS = False #True:多クラス分類、False:二値分類
    SUBMIT = True
    CV = True
    FOLD_NUM = 5
    CLIP_THRESHOLD = 0

    #Tensorboardの設定
    writer = SummaryWriter()
    
    #乱数の固定
    seed = SEED
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinmistic = True

    #学習用テーブルデータファイルの読み込み
    if MULTI_CLASS:
        master = pd.read_csv("train_master_multi.tsv", sep="\t")
        TARGET_NUM = 4
    else:
        master = pd.read_csv("train_master.tsv", sep="\t")
        #master = pd.read_csv("train_master_crensing.tsv", sep="\t")
        TARGET_NUM = 2


    #入力データ、正解ラベルの取得
    image_name_list = master["file_name"].values
    label_list = master["flag"].values
    master_multi = pd.read_csv("train_master_multi.tsv", sep="\t")
    label_list_multi = master_multi["flag"].values

    #学習データ、検証データの分割
    if CV:
        # 戻り値はインデックス
        #folds = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=SEED).split(image_name_list, label_list)
        folds = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=SEED).split(image_name_list, label_list_multi)
    else:
        # _train⇒イメージの名前、 _val⇒正解ラベル
        x_train, x_val, y_train, y_val = train_test_split(image_name_list, label_list, test_size=0.2, random_state=SEED, stratify=label_list)


    if CV:
        models_preds = []
        #テストデータのデータセット準備
        subdataset = For_Submission_Datasets(data_transform=data_transforms['val'])
        subdataloader = torch.utils.data.DataLoader(subdataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, drop_last=True)

        for fold, (train_idx, val_idx) in enumerate(folds):
            print(f'==========Cross-Validation Fold {fold+1}==========')
            x_train = image_name_list[train_idx]
            x_val = image_name_list[val_idx]
            y_train = label_list[train_idx]
            y_val = label_list[val_idx]
            
            if DRYRUN:
                EPOCH = 3
                BATCH_SIZE = 5
                x_train = x_train[:11]
                x_val = x_val[:11]
                y_train = y_train[:11]
                y_val = y_val[:11]
                    
            #datasetの作成
            image_datasets = {
                "train" : MyDataset(x_train, y_train, phase="train", train_dir=TRAIN_DIR),
                "val" : MyDataset(x_val, y_val, phase="val", train_dir=TRAIN_DIR)
            }

            #datasetのサイズを取得
            dataset_sizes= {
                'train':len(image_datasets['train']),
                'val':len(image_datasets['val'])
            }

            #detaloader作成
            image_dataloaders = {
                'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True, persistent_workers=True),
                'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False, pin_memory=True, persistent_workers=True)
            }

            #モデルオブジェクト作成、最適化関数定義、損失関数定義
            model_ft = get_model(target_num=TARGET_NUM, isPretrained=IS_PRETRAINED, device=device, model_name=MODEL_NAME)
            #optimizer = optim.Adam(model_ft.parameters(),lr=LEARNING_RATE)
            optimizer = optim.SGD(model_ft.parameters(),lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
            #n_iterations = len(image_dataloaders["train"]) * EPOCH
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCH)
            criterion = nn.CrossEntropyLoss()
            #criterion = Focal_MultiLabel_Loss(gamma=2.0)
            #criterion = nn.BCELoss()
            
            best_model = train_model_custom(model=model_ft, criterion=criterion, optimizer=optimizer, 
                                            num_epochs=EPOCH, is_saved=IS_SAVED, device=device, param_dir="./params",
                                            dataloaders=image_dataloaders, dataset_sizes=dataset_sizes,
                                            lr=LEARNING_RATE, bs=BATCH_SIZE, save_model_name=MODEL_NAME,
                                            multi_class=MULTI_CLASS, scheduler=scheduler, fold=fold, writer=writer)
            #テストデータの予測結果
            # CVの各データセットのtrain後、そのまま予測を行う
            pred_fold = []
            pred_fun = torch.nn.Softmax(dim=1)
            for i,(inputs, labels) in enumerate(subdataloader):
                with torch.set_grad_enabled(False):
                    inputs = inputs.to(device)
                    best_model.eval()
                    pre_outputs = best_model(inputs)
                    outputs = pred_fun(best_model(inputs))
                outputs = outputs.cpu().numpy()
                outputs = outputs[:, 1] # OK:0, NG:1
                pred_fold.append(outputs)
            
            pred_fold = np.concatenate(pred_fold)
            models_preds.append(pred_fold)

            #モデルの初期化
            del model_ft, best_model, optimizer, x_train, x_val, y_train, y_val
            gc.collect()
            torch.cuda.empty_cache()
        
        #最終的な予測結果(平均)
        models_preds = np.mean(models_preds, axis=0)
        pred = np.clip(models_preds, CLIP_THRESHOLD, 1.0-CLIP_THRESHOLD).round().astype(int)
            
    else:
        if DRYRUN:
            EPOCH = 3
            BATCH_SIZE = 5
            x_train = x_train[:11]
            x_val = x_val[:11]
            y_train = y_train[:11]
            y_val = y_val[:11]

        #datasetの作成
        image_datasets = {
            "train" : MyDataset(x_train, y_train, phase="train", train_dir=TRAIN_DIR),
            "val" : MyDataset(x_val, y_val, phase="val", train_dir=TRAIN_DIR)
        }

        #datasetのサイズを取得
        dataset_sizes= {
            'train':len(image_datasets['train']),
            'val':len(image_datasets['val'])
        }

        #detaloader作成
        image_dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True, persistent_workers=True),
            'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False, pin_memory=True, persistent_workers=True)
        }

        #モデルオブジェクト作成、最適化関数定義、損失関数定義
        model_ft = get_model(target_num=TARGET_NUM, isPretrained=IS_PRETRAINED, device=device, model_name=MODEL_NAME)
        #optimizer = optim.Adam(model_ft.parameters(),lr=LEARNING_RATE, weight_decay=0.0001)
        optimizer = optim.SGD(model_ft.parameters(),lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
        #n_iterations = len(image_dataloaders["train"]) * EPOCH
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCH)
        criterion = nn.CrossEntropyLoss()
        #criterion = Focal_MultiLabel_Loss(gamma=2.0)
        #criterion = nn.BCELoss()
        
        best_model = train_model_custom(model=model_ft, criterion=criterion, optimizer=optimizer, 
                                        num_epochs=EPOCH, is_saved=IS_SAVED, device=device, param_dir="./params",
                                        dataloaders=image_dataloaders, dataset_sizes=dataset_sizes,
                                        lr=LEARNING_RATE, bs=BATCH_SIZE, save_model_name=MODEL_NAME,
                                        multi_class=MULTI_CLASS, scheduler=scheduler, writer=writer)
    
    writer.close()

    #######################
    ### 提出データの作成  ##
    #######################
    if SUBMIT:

        # CVの場合は、CV中に集計したテストデータの予測結果(pred)を使用
        if not CV:
            # テストデータの読み込み
            subdataset = For_Submission_Datasets(data_transform=data_transforms['val'])
            subdataloader = torch.utils.data.DataLoader(subdataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, drop_last=True)

            # 提出データで推論
            pred = []
            for i,(inputs, labels) in enumerate(subdataloader):
                inputs = inputs.to(device)
                best_model.eval()
                outputs = best_model(inputs)
                _, preds = torch.max(outputs, 1)
                pred.append(preds.item())

        #提出ファイルの作成
        submit = pd.read_csv("./sample_submit.tsv", sep="\t", header=None)
        submit[1] = pred
        submit.to_csv("submit.tsv", index=False, header=False, sep="\t")
        print("Create submit.tsv.")
    else:
        print("Not Create submit.tsv.")


if __name__ == "__main__":
    main()