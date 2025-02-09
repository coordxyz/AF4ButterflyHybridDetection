import csv, pdb, random, os
from pathlib import Path

import torch
import pickle 
from torch.utils.data import DataLoader

from dataset import ButterflyDataset
from data_utils import data_transforms, load_data
from evaluation import evaluate, print_evaluation
from model_utils import get_feats_and_meta, get_dino_model
from classifier import train, get_scores, OneLayerClassifier

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
# Configuration         
ROOT_DATA_DIR = Path("../../Datasets/Images4TrainClassifier")
CSV_DIR = Path("../../Datasets/Images4TrainClassifier")
DATA_FILE = CSV_DIR / "List" / "butterfly_anomaly_AF.csv"  
IMG_DIR = ROOT_DATA_DIR / "Images"
CLF_SAVE_DIR = Path("../submission/dino2B_contestAF_1024linear2cls") 
CTN_SAVE_DIR = Path("../submission/dino2B_contestAF_1024linear2cls")

DEVICE = "cuda:0"
BATCH_SIZE = 20
NCLS = 2
NEPOCH = 200
CTN = False

def setup_data_and_model():
    # Load Data
    train_data, test_data = load_data(DATA_FILE, IMG_DIR, 0.2)

    # Model setup
    model = get_dino_model(dino_name='facebook/dinov2-base')
    return model.to(DEVICE), train_data, test_data


def prepare_data_loaders(train_data, test_data, class_nums=2):
    train_sig_dset = ButterflyDataset(train_data, IMG_DIR, phase='train', transforms=data_transforms(), class_nums=class_nums)
    tr_sig_dloader = DataLoader(train_sig_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    test_dset = ButterflyDataset(test_data, IMG_DIR, phase='train', transforms=data_transforms(), class_nums=class_nums)
    test_dl = DataLoader(test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    return tr_sig_dloader, test_dl


def extract_features(tr_sig_dloader, test_dl, model):
    tr_features, tr_labels,_ = get_feats_and_meta(tr_sig_dloader, model, DEVICE)
    test_features, test_labels,_ = get_feats_and_meta(test_dl, model, DEVICE)
    return tr_features, tr_labels, test_features, test_labels


def train_and_evaluate(tr_features, tr_labels, test_features, test_labels):
    configs = ["sgd"] #["svm","sgd","knn"]
    csv_output = []
    score_output = []

    for con in configs:
        print(f"Training and evaluating {con}...")
        clf, acc, h_acc, nh_acc = train(tr_features, tr_labels, con)

        # Save model to the specified path
        model_filename = CLF_SAVE_DIR / f"{con}_clf.pkl"
        with open(model_filename, 'wb') as model_file:
            pickle.dump(clf, model_file)
        print(f"Saved {con} classifier to {model_filename}")
        print(f"{con}: Acc - {acc:.4f}, Hacc - {h_acc:.4f}, NHacc - {nh_acc:.4f}")
        
        # Get scores for the test dataset
        scores = get_scores(clf, test_features)
        eval_scores = evaluate(scores, test_labels, reversed=False)
        print_evaluation(*eval_scores)
        csv_output.append([f"DiNO Features + {con}"] + list(eval_scores))
        
        # Save individual scores for analysis
        for idx, score in enumerate(scores):
            score_output.append([idx, score, test_labels[idx]])
            
    return csv_output, score_output


def train_sgd_clf():
    model, train_data, test_data = setup_data_and_model()
    tr_sig_dloader, test_dl = prepare_data_loaders(train_data, test_data)
    tr_features, tr_labels, test_features, test_labels = extract_features(tr_sig_dloader, test_dl, model)
    csv_output, score_output = train_and_evaluate(tr_features, tr_labels, test_features, test_labels)
    
    # Save evaluation results
    csv_filename = CLF_SAVE_DIR / "classifier_evaluation_results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Configuration", "Precision", "Recall", "F1-score","ROCAUC", "ACC","PRAUC"])
        writer.writerows(csv_output)
    
    # Save individual scores
    scores_filename = CLF_SAVE_DIR / "classifier_scores.csv"
    with open(scores_filename, mode='w', newline='') as score_file:
        score_writer = csv.writer(score_file)
        score_writer.writerow(["Index", "Score", "True Label"])
        score_writer.writerows(score_output)

def train_clf1024():
    # Model setup
    model = get_dino_model(dino_name='facebook/dinov2-base')
    model = model.to(DEVICE)
    continue_train = CTN

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')    
    class_nums = NCLS
    clfmodel = OneLayerClassifier(feat_dim=1536, class_nums=class_nums) #base:1536, large:2048, giant:3072
    clfmodel = clfmodel.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clfmodel.parameters(), lr=0.0001)
    if continue_train:
        clfmodel.load_state_dict(torch.load(CTN_SAVE_DIR/f"clf1024.pth"))

    num_epoch = NEPOCH
    loop = tqdm(range(num_epoch))
    for epoch in loop:
        train_acc = 0
        train_loss = 0

        # Load Data
        train_data, test_data = load_data(DATA_FILE, IMG_DIR, 0.1, random_state=None)
        tr_sig_dloader, test_dl = prepare_data_loaders(train_data, test_data, class_nums)
        tr_features, tr_labels,_ = get_feats_and_meta(tr_sig_dloader, model, DEVICE)
        features = torch.tensor(tr_features).to(device)

        ##0:non-hybrid, 1:hybrid
        labels = []
        for idx,ll in enumerate(tr_labels):
            tmplabel = [0]*class_nums
            tmplabel[int(ll)] = 1  
            labels.append(tmplabel)

        labels = torch.tensor(labels).to(device)

        optimizer.zero_grad()
        outputs = clfmodel(features)
        loss = criterion(outputs, labels.float())

        predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
        gts = labels.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == gts).sum().item()
        accuracy = correct / labels.shape[0]

        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch [{epoch}/{num_epoch}]")
        loop.set_postfix(loss=loss.item(), acc=accuracy)
        if epoch%50==0:
            torch.save(clfmodel.cpu().state_dict(), CLF_SAVE_DIR/ f"clf1024_latest.pth")
            clfmodel.cuda()
    torch.save(clfmodel.cpu().state_dict(), CLF_SAVE_DIR/ f"clf1024_last.pth")
    print('done')


if __name__ == "__main__":
    if not os.path.exists(CLF_SAVE_DIR):
        os.makedirs(CLF_SAVE_DIR)
        print('>>>makedirs: ',CLF_SAVE_DIR)
    print('>>>Training sgd classifier...')
    train_sgd_clf()   
    print('>>>Training linear classifier...')
    train_clf1024()
    print('Done. Models are saved to ',CLF_SAVE_DIR)