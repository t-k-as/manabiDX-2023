
import torch
import pandas as pd
from mylib import (
    MyDataset,
    get_model,
    plt_result,
    get_f1,
    predict_model,
    train_model_custom,
    For_Submission_Datasets,
    data_transforms,
)


if __name__ == "__main__":

    # デバイスの指定（CUDA）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    #定数設定
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
    MODEL_NAME = "resnet101"
    MULTI_CLASS = False #True:多クラス分類、False:二値分類
    TARGET_NUM = 2

    #######################
    ### 提出データの作成  ##
    #######################
    #モデルのロード
    best_model = get_model(target_num=TARGET_NUM, device=device, model_name=MODEL_NAME)
    best_model.load_state_dict(torch.load('./runs/Sep17_18-26-48_kato-home/resnet50_foldNone_epoch98_2023_917_18-26_best.pkl', map_location=lambda storage, loc: storage), strict=True)

    #テストデータの読み込み
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