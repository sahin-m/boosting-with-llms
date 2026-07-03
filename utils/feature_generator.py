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

import os
import json
from utils import gpt
from utils import sortinghat_featurization
import pandas as pd


def create_batchfiles(df, feature, stopped_index=0):
    batchfile = 'batches/' + feature.__name__ + '_batch.jsonl'
    if not os.path.isfile(batchfile):
        for idx in range(stopped_index, len(df)):
            row = df[idx:idx+1].reset_index(drop=True)
            feature(row, as_batch=True) # add the request to the batchfile
        print(f"Batch for feature '{feature.__name__}' added.\n" + \
        f'Make sure now to run the script /batches/run_llm.sh to retrieve batch results!\n')
    else:
        print(f"Batch for feature '{feature.__name__}' already exists!\n" + \
        f'Make sure now to run the script /batches/run_llm.sh to retrieve batch results!\n')
    return


def get_feature_results(df, feature, pipe=None, as_batch=False, stopped_index=0, save_path=None):
    featurized_df = sortinghat_featurization.FeatureExtraction(df)
    if save_path:
        if os.path.isfile(save_path):
            featurized_df = pd.read_csv(save_path, low_memory=False)

    if feature.__name__ in featurized_df.columns.tolist() and featurized_df[feature.__name__].isna().sum() == 0:
        print(f'Feature {feature.__name__} already done.')
        return featurized_df
        
    if as_batch and "GPT" in feature.__name__:
        result_batchfile = 'batches/' + feature.__name__ + '_batch_results.jsonl'
        if os.path.isfile(result_batchfile):
            with open(result_batchfile, "r", encoding="utf-8") as f:
                pred = [
                    json.loads(line)['response']['body']['choices'][0]['message']['content'].strip()
                    for line in f
                    if line.strip()
                ]
            if 'detect_all' in feature.__name__:
                featurized_df.loc[range(stopped_index, len(df)), feature.__name__] = [str(i).lower() for i in pred]
            else:
                pred = [0 if 'no' in str(i).lower() else str(i).lower() for i in pred]
                pred = [1 if 'yes' in str(i).lower() else str(i).lower() for i in pred]
                featurized_df.loc[range(stopped_index, len(df)), feature.__name__] = pred
                
            if save_path:
                featurized_df.to_csv(save_path, index=False)
                
            return featurized_df
        else:
            batchfile = 'batches/' + feature.__name__ + '_batch.jsonl'
            print(f"Batch '{batchfile}' exists but was not run yet.\n" + \
            f"Make sure now to run the script /batches/run_llm.sh to retrieve batch results!\n")
            return featurized_df
            
    else:
        if feature == None:
            print('Please specify the feature you want to retrieve results from.')
            return featurized_df
            
        if not feature.__name__ in featurized_df.columns.tolist() or featurized_df[feature.__name__].isna().sum() > 0:
            for idx in range(stopped_index, len(df)):
                row = df[idx:idx+1].reset_index(drop=True)

                # without batching, direct prompt
                if "GPT" in feature.__name__:
                    print(f"Index {idx}, Doing Feature '{row.loc[0, 'Attribute_name']}'\n")
                    pred = None
                    for i in range(6):
                        if not pred == None:
                            break                    
                        elif i == 5:
                            print(f'////////// Stopped at index {idx} //////////')
                            return featurized_df
                        else:
                            pred = feature(row)

                # local models
                elif "Gemma" in feature.__name__ or "Qwen" in feature.__name__:
                    if not pipe:
                        print(f"Failed to make predictions for feature '{feature.__name__}'!\n" + \
                              f"Model was not loaded / Pipeline was not set.\n")
                        return featurized_df
                    print(f"Index {idx}, Doing Feature '{row.loc[0, 'Attribute_name']}'\n")
                    pred = None
                    for i in range(10):
                        if not pred == None:
                            break
                        elif i == 9:
                            print(f'////////// Stopped at index {idx} //////////')
                            return featurized_df
                        else:
                            pred = feature(row, pipe)
                            
                if pred == True:
                    pred = 1
                elif pred == False:
                    pred = 0
    
                featurized_df.loc[idx, feature.__name__] = pred
                
                if save_path:
                    featurized_df.to_csv(save_path, index=False)
                    
                print('------------------------------------------------------------------\n')
            
    return featurized_df