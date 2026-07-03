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

from sklearn.model_selection import StratifiedGroupKFold


def create_split(df, test_size=0.2, target_col='y_act', group_col='Record_id', random_state=42):
    fold = {}
    X = df
    y = df[target_col]
    groups = df[group_col]
    
    cv = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
    train_idx, test_idx = list(cv.split(X, y, groups))[0] # take first fold
    fold['train'] = train_idx
    fold['test'] = test_idx
        
    return fold