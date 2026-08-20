"""Train reproducible TF-IDF baselines from WELFake-style CSV (title, text, label)."""
import argparse, sys
from pathlib import Path
import joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from main import Sessions,Model,Base,engine
p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--text-column',default='text');p.add_argument('--label-column',default='label');p.add_argument('--max-rows',type=int);args=p.parse_args()
df=pd.read_csv(args.csv).dropna(subset=[args.text_column,args.label_column])
if args.max_rows: df=df.sample(min(args.max_rows,len(df)),random_state=42)
text=df[args.text_column].astype(str);labels=df[args.label_column].astype(str).str.upper().replace({'0':'FAKE','1':'REAL'});x1,x2,y1,y2=train_test_split(text,labels,test_size=.2,random_state=42,stratify=labels)
Base.metadata.create_all(engine);(ROOT/'models').mkdir(exist_ok=True)
for name,max_features in {'bert':60000,'roberta':90000,'distilbert':30000}.items():
 pipe=Pipeline([('vectorizer',TfidfVectorizer(stop_words='english',max_features=max_features,ngram_range=(1,2))),('classifier',LogisticRegression(max_iter=1000,class_weight='balanced'))]);pipe.fit(x1,y1);pred=pipe.predict(x2);precision,recall,f1,_=precision_recall_fscore_support(y2,pred,average='weighted',zero_division=0);joblib.dump(pipe,ROOT/'models'/f'{name}.joblib');
 with Sessions() as s:
  old=s.get(Model,name);record=old or Model(name=name,path=str(ROOT/'models'/f'{name}.joblib'),accuracy=0,precision=0,recall=0,f1=0);record.accuracy=accuracy_score(y2,pred);record.precision=precision;record.recall=recall;record.f1=f1;s.add(record);s.commit()
 print(name,accuracy_score(y2,pred))
