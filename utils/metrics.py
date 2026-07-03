#Copyright 2026 Mücahit Sahin
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import numpy as np
import plotly.express as px
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score    


def binary_metrics(y_true, y_pred, class_map, decimal_places=3, return_results=False):
    if not return_results:
        print(f'\n######### Binarization Metrics #########\n')
        
    nr_classes = len(list(class_map.values()))
    metrics = {}
    
    for c in class_map.values():
        metrics[c] = {}

    for c in class_map.keys():
        binary_y_true = list(y_true)
        binary_y_pred = list(y_pred)
        for x in range(len(y_true)):
            if binary_y_true[x] == class_map[c]:
                binary_y_true[x] = 1
            else:
                binary_y_true[x] = 0
            if binary_y_pred[x] == class_map[c]:
                binary_y_pred[x] = 1
            else:
                binary_y_pred[x] = 0
                
        
        acc = accuracy_score(binary_y_true, binary_y_pred)
        prec = precision_score(binary_y_true, binary_y_pred, pos_label=1)
        rec = recall_score(binary_y_true, binary_y_pred, pos_label=1)
        f1 = f1_score(binary_y_true, binary_y_pred, pos_label=1)

        metrics[class_map[c]]["Accuracy"] = acc if not decimal_places else round(acc, decimal_places)
        metrics[class_map[c]]["Precision"] = prec if not decimal_places else round(prec, decimal_places)
        metrics[class_map[c]]["Recall"] = rec if not decimal_places else round(rec, decimal_places)
        metrics[class_map[c]]["F1-Score"] = f1 if not decimal_places else round(f1, decimal_places)

        if not return_results:
            print(f'## Class: {class_map[c]} ##\n')
            print(f'F1-Score: {f1 if not decimal_places else round(f1, decimal_places)}')
            print(f'Precision: {prec if not decimal_places else round(prec, decimal_places)}')
            print(f'Recall: {rec if not decimal_places else round(rec, decimal_places)}')
            print(f'Accuracy: {acc if not decimal_places else round(acc, decimal_places)}')
            print(f'\n////////////////////////////////////////\n')
            
    if return_results:
        return metrics
        


def model_performance(y_true, y_pred, class_map, decimal_places=2, class_average='macro'):
    print(f'######### General model performance #########\n')
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=class_average)
    prec = precision_score(y_true, y_pred, average=class_average)
    rec = recall_score(y_true, y_pred, average=class_average)

    print(f'F1-Score: {round(f1, decimal_places)}')
    print(f'Precision: {round(prec, decimal_places)}')
    print(f'Recall: {round(rec, decimal_places)}')
    print(f'Accuracy: {round(acc, decimal_places)}')

    return prec, rec, acc, f1



def plot_heatmap(y_true, y_pred, class_map_true, class_map_pred=None, decimal_places=2, font_size=22, save_path=None):
    if class_map_pred == None:
        class_map_pred = class_map_true

    nr_true_classes = len(list(class_map_true.values()))
    nr_pred_classes = len(list(class_map_pred.values())) 
    heatmap = np.zeros((nr_true_classes, nr_pred_classes))
    
    for i in range(nr_true_classes):
        for k in range(nr_pred_classes):
            for x in range(len(y_true)):
                if y_true[x] == class_map_true[i] and y_pred[x] == class_map_pred[k]:
                    heatmap[i, k] += 1
                    
    # normalize with respect to ground truth
    for i in range(nr_true_classes):
        for k in range(nr_pred_classes):
            if not heatmap[i, k] == 0:
                heatmap[i, k] = round(heatmap[i, k] / list(y_true).count(class_map_true[i]), decimal_places)

    fig = px.imshow(heatmap, labels=dict(x="Prediction", y="Ground truth", color='%'),
                    text_auto=True, x=list(class_map_pred.values()), y=list(class_map_true.values()), zmin=0, zmax=1)
    fig.update_layout(width=800, height=600)
    fig.show()

    if save_path:
        fig.update_layout(
            font=dict(size=font_size),  # increase font size
        )
        fig.update_xaxes(
            tickangle=35,
            tickfont=dict(size=font_size)
        )
        
        fig.update_yaxes(
            title_font=dict(size=font_size),
            tickfont=dict(size=font_size),
        )
        
        # Save figure as PDF
        fig.write_image(save_path)



def LLM_features_performance(featurized_df, features, class_map):
    df = featurized_df.copy()
    for i in features:
        label = ""
        for c in class_map:
            if class_map[c].replace('-', '_') in features[i].__name__.lower():
                label = class_map[c]
        if not 'heuristic' in features[i].__name__.lower():
            if features[i].__name__ in df.columns.tolist():
                print(f'\t\t######### Performance of feature: {features[i].__name__} #########\n')
    
                if 'all_at_once' in features[i].__name__.lower():
                    plot_heatmap(df['y_act'], df[features[i].__name__], class_map_true=class_map, decimal_places=3)
                    prec, rec, acc, f1 = model_performance(df['y_act'], df[features[i].__name__], class_map, decimal_places=3)
                    print('////////////////////////////////////////////////////////////////////////////\n')
                else:
                    pos_label = label
                    
                    if label == "not-generalizable":
                        neg_label = label.replace("not-", "")
                    else:
                        neg_label = "not-" + label
        
            
                    df.loc[(df[features[i].__name__] == 0), features[i].__name__] = neg_label
                    df.loc[(df[features[i].__name__] == 1), features[i].__name__] = pos_label
            
                    plot_heatmap(df['y_act'], df[features[i].__name__], class_map_true=class_map,
                                 class_map_pred={0: neg_label, 1: pos_label}, decimal_places=3)
                    
                    binary_metrics(df['y_act'], df[features[i].__name__], class_map={0: label})
                    print('////////////////////////////////////////////////////////////////////////////\n')