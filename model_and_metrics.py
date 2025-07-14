# coding: utf-8

import time
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.metrics import confusion_matrix
import matplotlib
import argparse
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, GINConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
# from torch.utils.data import random_split
# from torch_geometric.data import InMemoryDataset
# import matplotlib.pyplot as plt
# import pandas as pd

def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def plot_multi_label_confusion_matrix(_save_path, y_true, y_pred, labels, normalize=False, title=None, cmap=plt.cm.Blues):
    plt.close('all')
    plt.style.use("ggplot")
    plt.rcParams.update({'font.size': 8})
    plt.rcParams.update({'font.family':'Arial'})
    conf_mat_dict={}

    class_num = len(labels)

    plot_rows = int(class_num/4)+1
    plot_cols = 4 if class_num>=4 else class_num

    for label_col in range(class_num):
        y_true_label = y_true[:, label_col]
        y_pred_label = y_pred[:, label_col]

        print(y_true_label)
        print(y_pred_label)
        conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)


    fig, axes = plt.subplots(nrows=plot_rows, ncols=plot_cols, sharex=False, sharey=False,gridspec_kw = {'wspace':0.5, 'hspace':0.05},figsize=(10,10))
    axes = trim_axs(axes, class_num)
    for ii in range(len(labels)):
        _label = labels[ii]
        _matrix = conf_mat_dict[_label]
        axes[ii].imshow(_matrix,interpolation='nearest', cmap=plt.cm.Blues)
        axes[ii].set(xticks=np.arange(_matrix.shape[1]),
               yticks=np.arange(_matrix.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"],
               title=_label,
               ylabel='True label',
               xlabel='Predicted label')
        fmt = 'd'
        thresh = _matrix.max() / 2.
        for i in range(_matrix.shape[0]):
            for j in range(_matrix.shape[1]):
                axes[ii].text(j, i, format(_matrix[i, j], fmt),
                    ha="center", va="center", fontsize=16,
                    color="red")# if _matrix[i, j] > thresh else "black")

    plt.savefig(_save_path, dpi=400,pad_inches = 0.1,bbox_inches = 'tight')


# In[ ]:

def calculate_metrics(gts, ops, preds, class_num, labels, outputs, mode):
    if mode:
        gts = np.vstack([gts, labels.cpu()]) if gts.size else labels.cpu()
        y_pred = outputs.unsqueeze(1)
        y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
        y_pred = torch.max(y_pred, dim=1)[1]
        # print("Predict is %s"%y_pred)
        preds = np.vstack([preds, y_pred.cpu()]) if preds.size else y_pred.cpu()
    else:
        _labels = labels.cpu()
        tmp = torch.zeros(len(_labels), class_num)
        for idx, ele in enumerate(_labels):
            tmp[idx][ele] = 1
        gts = np.vstack([gts, tmp]) if gts.size else tmp
        view = outputs.view(-1, class_num)
        y_pred = (view == view.max(dim=1, keepdim=True)[0]).view_as(outputs).type(torch.ByteTensor)
            # y_pred = torch.max(outputs, 1)[1].view(labels.size())
            # y_pred = np.argmax(y_pred.cpu())
            # print(y_pred)
        preds = np.vstack([preds, y_pred.cpu()]) if preds.size else y_pred.cpu()

    acc_list = []
    auc_list = []
    f1 = f1_score(gts, preds, average="micro")
    for j in range(0, class_num):
        gts_i = gts[:,j]
        preds_i = preds[:,j]
        ops_i = ops[:,j]
        fpr, tpr, thresholds = roc_curve(gts_i, ops_i)
        acc_score = accuracy_score(gts_i, preds_i)
        auc_score = auc(fpr, tpr)
        acc_list.append(acc_score)
        auc_list.append(auc_score)
        print("class_num: %d, acc_score: %f, auc_score: %f"%(j, acc_score, auc_score))
    return acc_list, auc_list, f1, gts, ops, preds


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):

    plot_multi_label_confusion_matrix('%s_Confusion_matrix.png' % title, y_true, y_pred, classes)

    
def plot_roc_curve(pred_y, test_y, class_label, n_classes, fig_name="roc_auc.png"):
    #pred_y = pred_y/pred_y.max(axis=0)
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000", "#66CC99", "#999999"]
    plt.close('all')
    plt.style.use("ggplot")
    matplotlib.rcParams['font.family'] = "Arial"
    plt.figure(figsize=(8, 8), dpi=400)
    for i in range(n_classes):
        _tmp_pred = pred_y
        _tmp_label = test_y
        #print(_tmp_label[:, 0], _tmp_pred[:, 0])
        _fpr, _tpr, _ = roc_curve(_tmp_label[:, i], _tmp_pred[:, i])
        _auc = auc(_fpr, _tpr)
        plt.plot(_fpr, _tpr, color=colors[i],
                 label=r'%s ROC (AUC = %0.3f)' % (class_label[i], _auc), lw=2, alpha=.9)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC curve of')
    plt.legend(loc="lower right")
    plt.savefig(fig_name, dpi=400)
    plt.close('all')

##Define Model Class
class GCNTopK(torch.nn.Module):
    def __init__(self, num_feature, num_class, nhid=256, pooling_ratio=0.75):
        super(GCNTopK, self).__init__()
        self.nhid = nhid
        self.pooling_ratio = pooling_ratio
        self.conv1 = GraphConv(int(num_feature), self.nhid)
        self.pool1 = TopKPooling(self.nhid, ratio = self.pooling_ratio) # edited by Ming with concern for further extension
        self.conv2 = GraphConv(self.nhid, self.nhid)
        self.pool2 = TopKPooling(self.nhid, ratio = self.pooling_ratio)
        self.conv3 = GraphConv(self.nhid, self.nhid)
        self.pool3 = TopKPooling(self.nhid, ratio = self.pooling_ratio)
        #add one more conv-pooling block, i.e., conv4 and pool4
        self.conv4 = GraphConv(self.nhid, self.nhid)
        self.pool4 = TopKPooling(self.nhid, ratio = self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)   # edited by Ming with concern for further extension
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, num_class)  # edited by Ming with concern for further extension

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #add one more conv-pooling block, corresponding to conv4 and pool4
        x = F.relu(self.conv4(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool4(x, edge_index, edge_attr, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4
#         x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
#         print('shape of x before log_softmax: ',x.shape)
        y1 = F.log_softmax(self.lin3(x), dim=-1)
#         print('shape of x after log_softmax: ',x.shape)
        y2 = torch.sigmoid(self.lin3(x))

        return y1, y2
    
##GINTopK
class GINTopK(torch.nn.Module):
    def __init__(self, num_feature, num_class, nhid, pooling_ratio=0.75, dropout_ratio=0.7):
        super(GINTopK, self).__init__()
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio
        self.conv1 = GINConv(Seq(Lin(num_feature, nhid), ReLU(), Lin(nhid, nhid)))
        self.pool1 = TopKPooling(nhid, ratio=self.pooling_ratio)
        self.conv2 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        self.pool2 = TopKPooling(nhid, ratio=self.pooling_ratio)
        self.conv3 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        self.pool3 = TopKPooling(nhid, ratio=self.pooling_ratio)
        self.conv4 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        self.pool4 = TopKPooling(nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(2*nhid, nhid)
        self.lin2 = torch.nn.Linear(nhid, nhid//2)
        self.lin3 = torch.nn.Linear(nhid//2, num_class)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv4(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        y1 = F.log_softmax(self.lin3(x), dim=-1)
        y2 = torch.sigmoid(self.lin3(x))

        return y1, y2