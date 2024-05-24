import pandas as pd
from tqdm import tqdm
import swifter as sw
import numpy as np
from sklearn.model_selection import train_test_split

# Load train and test metadata
pa_train = pd.read_csv('/home/gigotleandri/Documents/GLC_24/data/GLC24-PA-metadata-train.csv',dtype={'speciesId':int, 'surveyId':int})

location = pa_train.drop_duplicates('surveyId').drop('speciesId',axis = 1)
target_arr = np.zeros([len(location),11255])

for i,survey_id in tqdm(enumerate(location.surveyId.values), total = len(location)):
    presense_species_ids = pa_train.loc[pa_train.surveyId == survey_id,'speciesId'].values
    target_arr[i,[ids-1 for ids in presense_species_ids]] = 1
    target_arr[i,-1] = int(survey_id)

species_target_columns = [str(sp_id) for sp_id in range(1,11255)] + ['surveyId']
train_df = pd.DataFrame(target_arr)
train_df.columns = species_target_columns
train_df = train_df.merge(location, on = 'surveyId')

train, test = train_test_split(train_df, test_size=0.05)
del train_df

def score(submission,answer):
    TP = 0
    FP = 0
    FN = 0
    for survey, prediction in tqdm(zip(*submission.T), total= len(submission)):
        tp = 0
        species = prediction.split(' ')
        for specie in species :
            if answer.loc[answer.surveyId == survey,specie].values[0] == 1 :
                tp += 1
            else :
                FP += 1
        FN += np.sum(answer.loc[answer.surveyId == survey].values[0,:-1]) - tp
        TP += tp
    
    return 2*TP/(2*TP + FP + FN)


tqdm.pandas()

def get_topk_pa_species(country, k=25):
    query = list(pa_train.loc[pa_train.country == country, "speciesId"].value_counts().nlargest(k).index)
    query.sort()
    return " ".join([str(c) for c in query]) if len(query) > 0 else "0"

'''
test["predictions"] = test.swifter.apply(lambda test_obs: get_topk_pa_species(test_obs["country"]), axis=1)
del pa_train
submission = test[["surveyId","predictions"]]
print(score(submission.values,test.iloc[:,:11255]))
'''


pa_test= pd.read_csv('/home/gigotleandri/Documents/GLC_24/data/GLC24-PA-metadata-test.csv',dtype={'speciesId':int, 'surveyId':int})
pa_test["predictions"] = pa_test.swifter.apply(lambda test_obs: get_topk_pa_species(test_obs["country"]), axis=1)
submission = pa_test[["surveyId","predictions"]]
submission.to_csv("submission-pa-country-top25.csv", index=False)
