"""Reproduce adolescent-only EFA-derived binary random-forest classification.

Uses the *fixed* configuration in adolescents_final.ipynb. This is NOT a
hyperparameter search. Feature selection is fitted within each training fold.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

SEEDS = [900,42,137,600,800,12,389,512,728,801,913,1024,1189,1344,1599,
         2047,2519,3421,4057,4829,5902,6593,7412,50613,84207,61729,
         45081,91236,76402,12589]
RF = dict(n_estimators=600, max_features='sqrt', max_depth=10,
          min_samples_leaf=12, min_samples_split=40, bootstrap=True,
          oob_score=True, class_weight='balanced_subsample', n_jobs=-1)

def load_features(root):
    betas = pd.read_csv(root/'data'/'betas_multi_voxels.csv')
    rois = pd.read_csv(root/'data'/'ROIs.csv').iloc[:,0].dropna().astype(str).tolist()
    betas = betas[betas['roi'].astype(str).isin(rois)].copy()
    betas['_ix'] = betas.groupby(['subject_id','roi','coef']).cumcount()
    betas['feature'] = (betas['roi'].astype(str)+'_'+betas['coef'].astype(str)
                        +'_'+betas['_ix'].astype(int).astype(str))
    return betas.pivot_table(index='subject_id',columns='feature',values='beta',aggfunc='first').reset_index()

def make_data(root, wide):
    labels = pd.read_csv(root/'data'/'adolescents_labels_filtered.csv')
    ids = pd.read_csv(root/'data'/'subject_id_adolescents_filtered.csv')[['subject_id']]
    if labels.subject_id.duplicated().any() or ids.subject_id.duplicated().any():
        raise ValueError('Duplicate subject IDs')
    if set(labels.subject_id) != set(ids.subject_id):
        raise ValueError('Binary label IDs differ from the filtered adolescent cohort')
    df = ids.merge(labels[['subject_id','impulsive']],on='subject_id',validate='one_to_one')
    df = df.merge(wide,on='subject_id',how='left',validate='one_to_one')
    if len(df)!=98: raise ValueError(f'Expected 98 adolescents, got {len(df)}')
    X = df.drop(columns=['subject_id','impulsive']).to_numpy()
    if not np.isfinite(X).all(): raise ValueError('Missing/nonfinite fMRI features')
    if df.impulsive.isna().any(): raise ValueError('Missing binary labels')
    return X,df.impulsive.to_numpy(),df[['subject_id','impulsive']]

def preprocess(Xtr,Xte,ytr):
    selector=SelectKBest(f_classif,k=10).fit(Xtr,ytr)
    a=selector.transform(Xtr); b=selector.transform(Xte)
    poly=PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
    a=poly.fit_transform(a); b=poly.transform(b)
    scaler=StandardScaler().fit(a)
    return scaler.transform(a),scaler.transform(b)

def evaluate(X,y,seed):
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,stratify=y,random_state=seed)
    # Inner CV is retained for procedural parity; only ONE candidate exists.
    inner=[]
    skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)
    for fold,(tr,va) in enumerate(skf.split(Xtr,ytr),1):
        a,b=preprocess(Xtr[tr],Xtr[va],ytr[tr])
        model=RandomForestClassifier(**RF,random_state=seed+fold).fit(a,ytr[tr])
        inner.append(accuracy_score(ytr[va],model.predict(b)))
    a,b=preprocess(Xtr,Xte,ytr)
    model=RandomForestClassifier(**RF,random_state=seed).fit(a,ytr)
    pred=model.predict(b)
    kw={'average':'binary','zero_division':0}
    out=dict(seed=seed,accuracy=accuracy_score(yte,pred),
             balanced_accuracy=balanced_accuracy_score(yte,pred),
             precision=precision_score(yte,pred,**kw),recall=recall_score(yte,pred,**kw),
             f1=f1_score(yte,pred,**kw),test_n=len(yte),inner_cv_accuracy=np.mean(inner))
    return out,confusion_matrix(yte,pred,labels=np.unique(y))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=30,choices=range(1,31),metavar='1..30')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    wide=load_features(root)
    X,y,labels=make_data(root,wide)
    print('EFA binary: X',X.shape,'class sizes',pd.Series(y).value_counts().sort_index().to_dict())
    rows=[]
    confusion=np.zeros((2,2),dtype=int)
    for seed in SEEDS[:args.seeds]:
        result,cm=evaluate(X,y,seed)
        rows.append(result)
        confusion+=cm
        print('seed',seed,'accuracy',round(result['accuracy'],4),flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(root/'results'/'efa_per_seed.csv',index=False)
    pd.DataFrame(confusion,index=['true_0','true_1'],columns=['pred_0','pred_1']).to_csv(root/'results'/'efa_confusion.csv')
    labels.to_csv(root/'results'/'efa_labels.csv',index=False)
    print('Mean accuracy',df.accuracy.mean(),'SD',df.accuracy.std(ddof=1),
          'Mean balanced accuracy',df.balanced_accuracy.mean(),flush=True)
if __name__=='__main__': main()
