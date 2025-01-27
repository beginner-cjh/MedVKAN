import os  
import numpy as np  
from skimage.io import imread  
import argparse 
    
def get_metrics(pred, mask):
    pred = (pred > 0.5).astype(np.float32)
    pred_positives = np.sum(pred)
    mask_positives = np.sum(mask)
    inter = np.sum(pred * mask)
    union = pred_positives + mask_positives
    dice = (2 * inter) / (union + 1e-6)
    iou = inter / (union - inter + 1e-6)
    acc = ((pred == mask).astype(np.float32)).mean()
    recall = inter / (mask_positives + 1e-6)
    precision = inter / (pred_positives + 1e-6)
    f2 = (5 * inter) / (4 * mask_positives + pred_positives + 1e-6)
    mae = np.abs(pred - mask).mean()

    return dice, iou, acc, recall, precision, f2, mae
  
def read_images(folder_true, folder_pred):  
    labels_true = []  
    labels_pred = []  

    for filename in os.listdir(folder_true):  
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            labels_path_true = os.path.join(folder_true, filename)  
            labels_true.append(imread(labels_path_true))  

            labels_path_pred = os.path.join(folder_pred, filename)  
            labels_pred.append(imread(labels_path_pred))
  
    return labels_true, labels_pred  
  
def main(true_folder_path, pred_folder_path):  
    labels_true, labels_pred = read_images(true_folder_path, pred_folder_path)  
    dice_list = []
    iou_list = []
    acc_list = []
    recall_list = []
    precision_list = []
    f2_list = []
    mae_list = []

    for i in range(len(labels_true)):  
        dice, iou, acc, recall, precision, f2, mae = get_metrics(labels_pred[i], labels_true[i][:,:,0])  
        dice_list.append(dice)
        iou_list.append(iou)
        acc_list.append(acc)
        recall_list.append(recall)
        precision_list.append(precision)
        f2_list.append(f2)
        mae_list.append(mae) 
        print(f"Image {i + 1}: Dice: {dice:.4f}, Iou: {iou:.4f}, Acc: {acc:.4f},Recall: {recall:.4f}")  
        print(f"Image {i + 1}: Precision: {precision:.4f}, F2: {f2:.4f},MAE: {mae:.4f}") 


    dice_mean = np.mean(dice_list)
    iou_mean = np.mean(iou_list)
    acc_mean = np.mean(acc_list)
    recall_mean = np.mean(recall_list)
    precision_mean = np.mean(precision_list)
    f2_mean = np.mean(f2_list)
    mae_mean = np.mean(mae_list)
    print(f"Dice: {dice_mean:.4f}, Iou: {iou_mean:.4f}, Acc: {acc_mean:.4f},Recall: {recall_mean:.4f}")  
    print(f"Precision: {precision_mean:.4f}, F2: {f2_mean:.4f},MAE: {mae_mean:.4f}")  

if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description="Calculate metrics from labels.")  
    parser.add_argument("-p", type=str, required=True, help="The base path for predicted labels folders.")  
    parser.add_argument("-t", type=str, required=True, help="The base path for true labels folders.") 
    args = parser.parse_args()  

    true_folder_path = os.path.join(args.t)    
    pred_folder_path = os.path.join(args.p)  

    main(true_folder_path, pred_folder_path)
