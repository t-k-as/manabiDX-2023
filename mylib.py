import io
import os
import torch
import torch.nn as nn
#import albumentations
from torch.utils.data import Dataset
from torchvision import models, transforms, utils
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import datetime
import copy
import time
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import ImageFilter

#データセットの分割
def split_dataset(image_name_list, label_list, test_size=0.2, random_state=123, stratify=None, dryrun=False):
    x_train, x_val, y_train, y_val = train_test_split(image_name_list, label_list, test_size=test_size, random_state=random_state, stratify=label_list)
    
    return x_train, x_val, y_train, y_val

#自作Transforms
def blur(img, radius=0.5, p=0.3):
    if np.random.rand() < p:
       img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img

#transformsの設定
data_transforms = {
    'train': transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        #transforms.Lambda(blur),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        #transforms.ColorJitter(brightness=0.5, saturation=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomAffine(degrees=[-5, 5], scale=(0.9, 1.0)),
        #transforms.RandomRotation(degrees=10),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}



#datasetクラスの設定
class MyDataset(Dataset):
    def __init__(self, image_name_list, label_list, phase=None, train_dir="./train/all"):
        self.image_name_list = image_name_list
        self.label_list = label_list
        self.phase = phase
        self.train_dir = train_dir
        
    def __len__(self):
        return len(self.image_name_list)
    
    def __getitem__(self, index):
        #index番目の画像を読み込み、前処理を行う
        image_path = self.image_name_list[index]
        image = Image.open(self.train_dir + str(image_path))
        image = data_transforms[self.phase](image)
                
        #画像のラベルを取得する
        label = self.label_list[index]
        
        #入力データと正解ラベルをセットで返す
        return image, label, image_path
    


#提出用データのdataloader作成
class For_Submission_Datasets(Dataset):
    
    def __init__(self, data_transform):
        self.df = pd.read_csv('./sample_submit.tsv', sep="\t", names=['file_name','flag'])
        self.data_transform = data_transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        file = self.df['file_name'][index]
        image = Image.open('./cut/test/'+ file)
        image = self.data_transform(image)

        return image,file


# 自作損失関数（Focal Loss）
class Focal_MultiLabel_Loss(nn.Module):
    def __init__(self, gamma):
      super(Focal_MultiLabel_Loss, self).__init__()
      self.gamma = gamma
      self.bceloss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets): 
      bce = self.bceloss(outputs, targets)
      bce_exp = torch.exp(-bce)
      focal_loss = (1-bce_exp)**self.gamma * bce
      return focal_loss.mean()


#モデル作成関数
def get_model(target_num,isPretrained=False, device="cpu", model_name="resnet18"):
    if model_name == "resnet18":
        if(isPretrained):
            model_ft = models.resnet18(pretrained=True)
            #model_ft.load_state_dict(torch.load('../input/resnet18-5c106cde.pth', map_location=lambda storage, loc: storage), strict=True)
        else:
            model_ft = models.resnet18(pretrained=False)
    
        model_ft.fc = nn.Linear(512, target_num) #resnet18
        
    elif model_name == "resnet50":
        if(isPretrained):
            model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            model_ft = models.resnet50(pretrained=False)
    
        model_ft.fc = nn.Linear(2048, target_num) #resne50

    elif model_name == "resnet101":
        if(isPretrained):
            model_ft = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        else:
            model_ft = models.resnet101(pretrained=False)
    
        model_ft.fc = nn.Linear(2048, target_num) #resnet101

    elif model_name == "resnet152":
        if(isPretrained):
            model_ft = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
        else:
            model_ft = models.resnet152(pretrained=False)
    
        model_ft.fc = nn.Linear(2048, target_num) #resnet152

    elif model_name == "efficientnet_v2_l":
        if(isPretrained):
            model_ft = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        else:
            model_ft = models.efficientnet_v2_l(pretrained=False)
    
        model_ft.classifier = nn.Linear(1280, target_num)
    
    else:
        print("model_name is not found.")
        return None

    print("★ {} is loaded. ★".format(model_name))
    print() 
    model_ft = model_ft.to(device)
    return model_ft




#モデル訓練用関数（Tensorboaard利用）
def train_model_custom(model, criterion, optimizer, num_epochs=1,is_saved=False, device="cpu",
                param_dir="./params", dataloaders=None, dataset_sizes=None,
                lr=0.001, bs=25, save_model_name="resnet18", multi_class=False, scheduler=None, fold=None, writer=None):
    
    save_model_dir = "./params"
    os.makedirs(save_model_dir, exist_ok=True)
    d = datetime.datetime.now()
    save_day = "{}_{}{}_{}-{}".format(d.year, d.month, d.day, d.hour, d.minute)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_precision = 0.0
    best_f1 = 0.0
    best_train_f1 = 0.0
      
    # エポック数だけ下記工程の繰り返し
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()  

            epoch_preds = []
            epoch_labels = []
            epoch_image_names = []
            running_loss = 0.0
            running_corrects = 0
            fn_img_list = []
            fp_img_list = []

            # dataloadersからバッチサイズだけデータ取り出し、下記工程（1−5）の繰り返し
            for inputs, labels, image_name in tqdm(dataloaders[phase]):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # 1. optimizerの勾配初期化
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                     # 2.モデルに入力データをinputし、outputを取り出す
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, axis=1)
                    epoch_preds.extend(preds.cpu())
                    epoch_labels.extend(labels.cpu())
                    epoch_image_names.extend(labels.cpu())                    

                    # 3. outputと正解ラベルから、lossを算出
                    loss = criterion(outputs, labels)

                    if phase == 'train':                        
                        # 4. 誤差逆伝播法により勾配の算出
                        loss.backward()
                        # 5. optimizerのパラメータ更新
                        optimizer.step()
                        
                #学習の評価指標を計算
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if epoch % 10 == 0:
                    # False NegativeとFalse Positiveの画像を特定
                    false_negative_mask = (preds == 0) & (labels == 1)
                    false_positive_mask = (preds == 1) & (labels == 0)

                    # False NegativeとFalse Positiveの画像をリストに格納
                    for i in range(len(inputs)):
                        if false_negative_mask[i]:
                            fn_img_list.append(inv_normalize(inputs[i].cpu()))
                        elif false_positive_mask[i]:
                            fp_img_list.append(inv_normalize(inputs[i].cpu()))
                        
            if phase == 'train':
                scheduler.step()
                print("epoch: {} lr: {:6f}".format(epoch, scheduler.get_last_lr()[0]))
                writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

            #評価項目（loss, accuracy, reacall, precision）
            #epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if multi_class:
                epoch_recall = recall_score(y_true=epoch_labels, y_pred=epoch_preds, average="macro")
                epoch_precision = precision_score(y_true=epoch_labels, y_pred=epoch_preds, average="macro")
                epoch_f1 = f1_score (y_true=epoch_labels, y_pred=epoch_preds,  average="macro")
            else:
                epoch_recall = recall_score(y_true=epoch_labels, y_pred=epoch_preds, pos_label=1)
                epoch_precision = precision_score(y_true=epoch_labels, y_pred=epoch_preds, pos_label=1)
                epoch_f1 = f1_score (y_true=epoch_labels, y_pred=epoch_preds, pos_label=1)
            
            if phase == "train":
                epoch_train_f1 = epoch_f1

            #Tensorboardに書き込み
            writer.add_scalar("loss/fold{}/{}".format(fold,phase), epoch_loss, epoch)
            writer.add_scalar("accuracy/fold{}/{}".format(fold,phase), epoch_acc, epoch)
            writer.add_scalar("recall/fold{}/{}".format(fold,phase), epoch_recall, epoch)
            writer.add_scalar("precision/fold{}/{}".format(fold,phase), epoch_precision, epoch)
            writer.add_scalar("f1-score/fold{}/{}".format(fold,phase), epoch_f1, epoch)
            print('{} Loss: {:.4f} Acc: {:.4f} Recall: {:.4f} Precision: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_recall, epoch_precision, epoch_f1))
            
            # 混同行列の保存（Tensorboard）
            #writer.add_figure("Confusion matrix/fold{}/{}".format(fold,phase), createConfusionMatrix(y_true=epoch_labels, y_pred=epoch_preds), epoch)    

            # 不正解画像のみ保存（Tensorboard）
            if len(fn_img_list) > 0:
                fn_grid_images = torch.stack(fn_img_list, dim=0)
                writer.add_images(f'False_Negative/fold{fold}/{phase}', fn_grid_images, epoch)
                            
            if len(fp_img_list) > 0:
                fp_grid_images = torch.stack(fp_img_list, dim=0)
                writer.add_images(f'False_Positive/fold{fold}/{phase}', fp_grid_images, epoch)
            
            #問答無用でモデルを保存
            # if phase == "val":
            #     torch.save(model.state_dict(), os.path.join(save_model_dir, save_model_name+"_{}_{}_recall_1.0.pkl".format(epoch, save_day)))
            #     print("save model recall=1.0 epoch :{}".format(epoch))

            # 今までのエポックの精度よりも高い場合はモデルの保存
            if phase == 'val' and epoch_f1 > best_f1:
                #torch.save(model.state_dict(), os.path.join(save_model_dir, save_model_name+"_{}_{}_{}.pkl".format(fold, epoch, save_day)))
                print("save model epoch :{}".format(epoch))
                best_acc = epoch_acc
                best_precision = epoch_precision
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_train_f1 = epoch_train_f1    
                
        print()

    time_elapsed = time.time() - since
    
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}, F1-score: {:4f}, epoch: {}".format(best_acc, best_f1, best_epoch))

    #ベストモデルのロード
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(save_model_dir, save_model_name+"_fold{}_epoch{}_{}_best.pkl".format(fold, best_epoch, save_day)))
    
    return model


# 標準化した画像の逆変換
def inv_normalize(img):
    img = img.mul(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    img = img.add(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))

    return img

# 混同行列の作成関数
def createConfusionMatrix(y_true, y_pred):
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    cf_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    cf_labels = np.asarray(cf_labels).reshape(2,2)
    plt.figure(figsize=(12, 7))   

    return sns.heatmap(cf_matrix, annot=cf_labels, fmt="", cmap='Blues').get_figure()


#結果の可視化
def plt_result(train_losses, train_acces, val_losses, val_acces, epoch):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.xlim([0, epoch])
    plt.ylim([0.0, 1.0])    
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.title('losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_acces, label="train_acc")
    plt.plot(val_acces, label="val_acc")
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()

    plt.tight_layout()
    plt.show()
    #plt.pause(0.1)
    #plt.clf()




#モデル予測用関数
def predict_model(model, dataloader, device="cpu"):
    pred = []
    pred_labels = []
    ground_truth = []

    for i,(inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device, non_blocking=True)
        model.eval()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        pred.append(preds.tolist())
        pred_labels.append(preds.cpu().detach().numpy())
        ground_truth.append(labels.cpu().detach().numpy())

    pred = torch.tensor(pred)
    return pred, pred_labels, ground_truth




#精度算出関数
def get_f1(true_labels_list: list,
           predictions_list: list,
           average_method: str,
          ):
    """This function will performs model inferencing using test data
    and stores the results into the lists.

    Args:
        true_labels_list (list): List of true labels.
        predictions_list (list): List of predictions.
        average_method (string): method to average score.

    Returns:
        f1 (float): return f1 metric.
        precision (float): return precision metric.
        recall (float): return recall metric.
    """
    f1 = f1_score(
        y_true=true_labels_list,
        y_pred=predictions_list,
        average=average_method
    )

    precision = precision_score(
        y_true=true_labels_list,
        y_pred=predictions_list,
        average=average_method,
    )

    recall = recall_score(
        y_true=true_labels_list,
        y_pred=predictions_list,
        average=average_method,
    )

    f1 = round(f1, 2)
    precision = round(precision, 2)
    recall = round(recall, 2)

    return f1, precision, recall
