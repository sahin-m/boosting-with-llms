# Original Source Code from SortingHat. For more details, see reference:
# https://github.com/vraj-ucsd/SortingHatLib/blob/master/sortinghat/pylib.py

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import joblib
import regex as re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


del_pattern = r'([^,;\|]+[,;\|]{1}[^,;\|]+){1,}'
del_reg = re.compile(del_pattern)

delimeters = r"(,|;|\|)"
delimeters = re.compile(delimeters)

url_pat = r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
url_reg = re.compile(url_pat)

email_pat = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b"
email_reg = re.compile(email_pat)

stop_words = set(stopwords.words('english'))


vectorizerName = joblib.load(
    PROJECT_ROOT / "resources/Dictionary/dictionaryName.pkl"
)


def FeatureExtraction(data, useSamples=0):

    data1 = data[['total_vals', 'num_nans', '%_nans', 'num_of_dist_val', '%_dist_val', 'mean', 'std_dev', 'min_val', 'max_val','has_delimiters',
                  'has_url', 'has_email', 'has_date', 'mean_word_count',
                  'std_dev_word_count', 'mean_stopword_total', 'stdev_stopword_total',
                  'mean_char_count', 'stdev_char_count', 'mean_whitespace_count',
                  'stdev_whitespace_count', 'mean_delim_count', 'stdev_delim_count',
                  'is_list', 'is_long_sentence']]
    data1 = data1.reset_index(drop=True)
    data1 = data1.fillna(0)

    arr = data['Attribute_name'].values
    arr = [str(x) for x in arr]
    
    X = vectorizerName.transform(arr)
    attr_df = pd.DataFrame(X.toarray())

    if useSamples:
        arr1 = data['sample_1'].values
        arr1 = [str(x) for x in arr1]
        arr2 = data['sample_2'].values
        arr2 = [str(x) for x in arr2]

        X1 = vectorizerSample.transform(arr1)
        X2 = vectorizerSample.transform(arr2)    

        sample1_df = pd.DataFrame(X1.toarray())
        sample2_df = pd.DataFrame(X2.toarray())
        data2 = pd.concat([data1, attr_df, sample1_df, sample2_df], axis=1, sort=False)
        data2 = data2.replace({True: 1, False: 0})
    else:
        data2 = pd.concat([data1, attr_df], axis=1, sort=False)
        data2 = data2.replace({True: 1, False: 0})
        
    return data2


def FeaturizeFile(df):

	stats = []
	attribute_name = []
	sample = []
	id_value = []
	i = 0

	castability = []
	number_extraction = []

	avg_tokens = []
	ratio_dist_val = []
	ratio_nans = []

	keys = list(df.keys())

	attribute_name.extend(keys)
	summary_stat_result = _summary_stats(df, keys)
	stats.extend(summary_stat_result)
	samples = _get_sample(df,keys)
	sample.extend(samples)

	ratio_dist_val.extend(_get_ratio_dist_val(summary_stat_result))
	ratio_nans.extend(_get_ratio_nans(summary_stat_result))


	csv_names = ['Attribute_name', 'total_vals', 'num_nans', 'num_of_dist_val', 'mean', 'std_dev', 'min_val',
	             'max_val', '%_dist_val', '%_nans', 'sample_1', 'sample_2', 'sample_3','sample_4','sample_5'
	            ]
	golden_data = pd.DataFrame(columns = csv_names)

	for i in range(len(attribute_name)):
	    val_append = []
	    val_append.append(attribute_name[i])
	    val_append.extend(stats[i])
	    
	    val_append.append(ratio_dist_val[i])
	    val_append.append(ratio_nans[i])    
	    
	    val_append.extend(sample[i])
	
	    golden_data.loc[i] = val_append


	curdf = golden_data

	for row in curdf.itertuples():

	    is_list = False
	    curlst = [row[11],row[12],row[13],row[14],row[15]]
	    
	    delim_cnt,url_cnt,email_cnt,date_cnt =0,0,0,0
	    chars_totals,word_totals,stopwords,whitespaces,delims_count = [],[],[],[],[]
	    
	    for value in curlst: 
	        word_totals.append(len(str(value).split(' ')))
	        chars_totals.append(len(str(value)))
	        whitespaces.append(str(value).count(' '))
	        
	        if del_reg.match(str(value)):  delim_cnt += 1    
	        if url_reg.match(str(value)):  url_cnt += 1
	        if email_reg.match(str(value)):  email_cnt += 1
	        
	        delims_count.append(len(delimeters.findall(str(value))))        
	    
	        tokenized = word_tokenize(str(value))
	        stopwords.append(len([w for w in tokenized if w in stop_words]))    
	    
	        try:
	            _ = pd.Timestamp(value)
	            date_cnt += 1
	        except ValueError: date_cnt += 0    
	    
	    if delim_cnt > 2:  curdf.at[row.Index, 'has_delimiters'] = True
	    else: curdf.at[row.Index, 'has_delimiters'] = False

	    if url_cnt > 2:  curdf.at[row.Index, 'has_url'] = True
	    else: curdf.at[row.Index, 'has_url'] = False
	        
	    if email_cnt > 2:  curdf.at[row.Index, 'has_email'] = True
	    else: curdf.at[row.Index, 'has_email'] = False   
	        
	    if date_cnt > 2:  curdf.at[row.Index, 'has_date'] = True
	    else: curdf.at[row.Index, 'has_date'] = False           
	        
	    curdf.at[row.Index, 'mean_word_count'] = np.mean(word_totals)
	    curdf.at[row.Index, 'std_dev_word_count'] = np.std(word_totals)
	    
	    curdf.at[row.Index, 'mean_stopword_total'] = np.mean(stopwords)
	    curdf.at[row.Index, 'stdev_stopword_total'] = np.std(stopwords)
	    
	    curdf.at[row.Index, 'mean_char_count'] = np.mean(chars_totals)    
	    curdf.at[row.Index, 'stdev_char_count'] = np.std(chars_totals)
	    
	    curdf.at[row.Index, 'mean_whitespace_count'] = np.mean(whitespaces)
	    curdf.at[row.Index, 'stdev_whitespace_count'] = np.std(whitespaces)    
	    
	    curdf.at[row.Index, 'mean_delim_count'] = np.mean(whitespaces)
	    curdf.at[row.Index, 'stdev_delim_count'] = np.std(whitespaces)      
	    
	    if curdf.at[row.Index, 'has_delimiters'] and curdf.at[row.Index, 'mean_char_count'] < 100: curdf.at[row.Index, 'is_list'] = True    
	    else: curdf.at[row.Index, 'is_list'] = False
	    
	    if curdf.at[row.Index, 'mean_word_count'] > 10: curdf.at[row.Index, 'is_long_sentence'] = True    
	    else: curdf.at[row.Index, 'is_long_sentence'] = False    
	    
	golden_data = curdf

	return golden_data	


def _summary_stats(dat, key_s):
    b_data = []
    for col in key_s:
        nans = np.count_nonzero(pd.isnull(dat[col]))
        dist_val = len(pd.unique(dat[col].dropna()))
        Total_val = len(dat[col])
        mean = 0
        std_dev = 0
        var = 0
        min_val = 0
        max_val = 0
        if is_numeric_dtype(dat[col]):
            mean = np.mean(dat[col])
            
            if pd.isnull(mean):
                mean = 0
                std_dev = 0
                #var = 0
                min_val = 0
                max_val = 0           
            else:    
                std_dev = np.std(dat[col])
                var = np.var(dat[col])
                min_val = float(np.min(dat[col]))
                max_val = float(np.max(dat[col]))
        b_data.append([Total_val, nans, dist_val, mean, std_dev, min_val, max_val])
    return b_data
    

def _get_sample(dat, key_s):
    rand = []
    for name in key_s: # TODO Omg this is bad. Should use key_s.
        rand_sample = list(pd.unique(dat[name]))
        rand_sample = rand_sample[:5]
        while len(rand_sample) < 5:
            rand_sample.append(list(pd.unique(dat[name]))[np.random.randint(len(list(pd.unique(dat[name]))))])
        rand.append(rand_sample[:5])
    return rand

def _get_ratio_dist_val(summary_stat_result):
    ratio_dist_val = []
    for r in summary_stat_result:
        ratio_dist_val.append(r[2]*100.0 / r[0])
    return ratio_dist_val

def _get_ratio_nans(summary_stat_result):
    ratio_nans = []
    for r in summary_stat_result:
        ratio_nans.append(r[1]*100.0 / r[0])
    return ratio_nans
