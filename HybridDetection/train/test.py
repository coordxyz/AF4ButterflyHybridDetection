import csv
from pathlib import Path

import torch
import pickle 
from torch.utils.data import DataLoader

from dataset import ButterflyDataset
from data_utils import data_transforms, load_data
from evaluation import evaluate, print_evaluation
from model_utils import get_feats_and_meta, get_dino_model
from classifier import train, get_scores,OneLayerClassifier
import os,pdb,shutil
import numpy as np
import torch.nn.functional as F
# Configuration         
ROOT_DATA_DIR = Path("/ns_data/DATA/AD/butterflySpeciesVal")
CSV_DIR = Path("/ns_data/DATA/AD/butterflyContest")
DATA_FILE = CSV_DIR / "ref" / "butterfly_anomaly_val.csv"
IMG_DIR = ROOT_DATA_DIR 
CLF_SAVE_DIR = Path("/local_data/data/CODE/AD/Imageomics/HDR-anomaly-challenge-sample-main/DINO_train/dino2B_clean_2k_linear")
DEVICE = "cuda:1"
BATCH_SIZE = 4


def setup_data_and_model():
    # Load Data
    _, test_data = load_data(DATA_FILE, IMG_DIR, 0.99, phase='train')

    # Model setup
    model = get_dino_model()
    return model.to(DEVICE), test_data


def prepare_data_loaders(test_data):
    test_dset = ButterflyDataset(test_data, IMG_DIR, phase='train', transforms=data_transforms())
    test_dl = DataLoader(test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    return test_dl


def extract_features(test_dl, model):
    test_features, test_labels, test_paths = get_feats_and_meta(test_dl, model, DEVICE)
    return test_features, test_labels, test_paths


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

def run_evaluate(test_features, test_labels):
    configs = ["sgd"] #["svm","sgd","knn"]
    csv_output = []
    score_output = []

    for con in configs:
        print(f"Evaluating {con}...")
            # Load Classifier weights
        with open(os.path.join(CLF_SAVE_DIR, con+"_clf.pkl"), "rb") as f:
            clf = pickle.load(f)
        # Get scores for the test dataset
        scores = get_scores(clf, test_features)
        eval_scores = evaluate(scores, test_labels, reversed=False)
        print_evaluation(*eval_scores)
        csv_output.append([f"DiNO Features + {con}"] + list(eval_scores))
        
        # Save individual scores for analysis
        for idx, score in enumerate(scores):
            score_output.append([idx, score, test_labels[idx]])
            
    return csv_output, score_output

def run_evaluate_layers(test_features, test_labels):
    configs = ["sgd"] #["svm","sgd","knn"]
    csv_output = []
    score_output = []

    clfmodel = OneLayerClassifier(feat_dim=1536, class_nums=196)
    clfmodel = clfmodel.to(DEVICE)
    clfmodel.load_state_dict(torch.load(CLF_SAVE_DIR/f"trained_onelayer_classifier_latest.pth"))
    clfmodel.eval()
    test_features = torch.tensor(test_features).to(DEVICE)
    outputs = clfmodel(test_features)
    # Get scores for the test dataset
    # scores = outputs.argmax(dim=1, keepdim=True).squeeze()
    values,scores = torch.max(outputs.data,1)
    scores[scores<14]=0
    scores[scores>13]=1
    # scores = F.softmax(outputs, dim=1)
    # pdb.set_trace()
    eval_scores = evaluate(scores.cpu(), test_labels, reversed=False)
    print_evaluation(*eval_scores)
    csv_output.append([f"DINO Features + onelayer"] + list(eval_scores))
    
    # Save individual scores for analysis
    for idx, score in enumerate(scores):
        score_output.append([idx, score, test_labels[idx]])
            
    return csv_output, score_output

def run_layers(test_features, test_labels, test_paths):
    configs = ["sgd"] #["svm","sgd","knn"]
    score_output = []

    classes = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13']
    for ii in range(14):
        for jj in range(14):
            if ii==jj:
                continue
            classes.append(str(ii)+'_'+str(jj))


    clfmodel = OneLayerClassifier(feat_dim=1536, class_nums=196)
    clfmodel = clfmodel.to(DEVICE)
    clfmodel.load_state_dict(torch.load(CLF_SAVE_DIR/f"trained_onelayer_classifier_latest.pth"))
    clfmodel.eval()
    test_features = torch.tensor(test_features).to(DEVICE)
    outputs = clfmodel(test_features)
    # Get scores for the test dataset
    # scores = outputs.argmax(dim=1, keepdim=True).squeeze()
    values,predid = torch.max(outputs.data,1)
    # scores[scores<14]=0
    # scores[scores>13]=1
    scores = F.softmax(outputs, dim=1)
    
    # Save individual scores for analysis
    for idx, score in enumerate(scores):
        savescore = score[predid[idx]].detach().cpu().numpy()
        saveid = predid[idx].detach().cpu().numpy()
        parents = classes[test_labels[idx]].split('_')
        filename=test_paths[idx].split('/')[-1]
        # if savescore < 0.9:           
        #     if len(parents)==2:
        #         if classes[saveid]==parents[0] or classes[saveid]==parents[1]:
        #             ##---filtered
        #             # score_output.append([test_paths[idx], savescore, saveid, test_labels[idx], classes[saveid], classes[test_labels[idx]] ])
        #             ##---save for training                    
        #             score_output.append(['','','hybrid','train','','',classes[test_labels[idx]],parents[0],parents[1],filename])
        #             shutil.move(test_paths[idx], test_paths[idx].replace('genmcls', 'genmcls2'))
        
        # if savescore > 0.7:
        #     score_output.append(['','','hybrid','train','','',classes[test_labels[idx]],classes[test_labels[idx]],classes[test_labels[idx]],filename])
            # shutil.move(test_paths[idx], test_paths[idx].replace('genscls3', 'genscls'))
        if saveid>14:
            score_output.append([test_paths[idx], savescore, saveid, test_labels[idx], classes[saveid], classes[test_labels[idx]] ])

        # pdb.set_trace()

        # if savescore < 0.5:
        #     score_output.append([test_paths[idx], savescore, saveid, test_labels[idx], classes[saveid], classes[test_labels[idx]] ])
            

    return score_output



def run_evaluate_ensamble(test_features, test_labels):
    configs = ["sgd"] #["svm","sgd","knn"]
    csv_output = []
    score_output = []

    with open(os.path.join(CLF_SAVE_DIR, "trained_"+configs[0]+"_classifier.pkl"), "rb") as f:
        clf = pickle.load(f)
    # Get scores for the test dataset
    sgdscores = get_scores(clf, test_features)

    clfmodel = OneLayerClassifier(feat_dim=1536)
    clfmodel = clfmodel.to(DEVICE)
    clfmodel.load_state_dict(torch.load(CLF_SAVE_DIR/f"trained_onelayer_classifier_latest.pth"))
    clfmodel.eval()
    test_features = torch.tensor(test_features).to(DEVICE)
    outputs = clfmodel(test_features)
    # Get scores for the test dataset
    # scores = outputs.argmax(dim=1, keepdim=True).squeeze()
    values,scores = torch.max(outputs.data,1)
    # scores = F.softmax(outputs, dim=1)
    # pdb.set_trace()
    scores = np.maximum(scores.cpu().numpy(), sgdscores)
    
    eval_scores = evaluate(scores, test_labels, reversed=False)
    print_evaluation(*eval_scores)
    csv_output.append([f"DINO Features + onelayer"] + list(eval_scores))
    
    # Save individual scores for analysis
    for idx, score in enumerate(scores):
        score_output.append([idx, score, test_labels[idx]])
            
    return csv_output, score_output

def main():
    model, test_data = setup_data_and_model()
    test_dl = prepare_data_loaders(test_data)
    test_features, test_labels, test_paths = extract_features(test_dl, model)
    # csv_output, score_output = run_evaluate(test_features, test_labels)
    # csv_output, score_output = run_evaluate_layers(test_features, test_labels)
    # csv_output, score_output = run_evaluate_ensamble(test_features, test_labels)
    score_output = run_layers(test_features, test_labels, test_paths)

    # Save evaluation results
    # csv_filename = CLF_SAVE_DIR / "classifier_evaluation_results_layer.csv"
    # with open(csv_filename, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Configuration", "Recall", "Precision", "F1-score","ROCAUC", "ACC","PRAUC"])
    #     writer.writerows(csv_output)
    
    # Save individual scores
    scores_filename = CLF_SAVE_DIR / "classifier_scores_layer.csv"
    with open(scores_filename, mode='w', newline='') as score_file:
        score_writer = csv.writer(score_file)
        # score_writer.writerow(["Index", "Score",'PredID', "GtID", 'PredCls', 'GtCls'])
        score_writer.writerows(score_output)
    
if __name__ == "__main__":
    main()