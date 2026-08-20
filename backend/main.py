"""Real, local fake-news inference API. Models are produced by scripts/train.py."""
from __future__ import annotations
import csv, io, json, os, time
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

load_dotenv()

ROOT=Path(__file__).resolve().parents[1]; MODEL_DIR=ROOT/'models'; DB=os.getenv('DATABASE_URL',f"sqlite:///{ROOT/'database'/'fake_news.db'}"); (ROOT/'database').mkdir(exist_ok=True)
engine=create_engine(DB,connect_args={'check_same_thread':False} if DB.startswith('sqlite') else {}); Sessions=sessionmaker(bind=engine)
class Base(DeclarativeBase):pass
class Prediction(Base):
 __tablename__='prediction_history';id:Mapped[int]=mapped_column(primary_key=True);headline:Mapped[str]=mapped_column(String(300));label:Mapped[str]=mapped_column(String(10));confidence:Mapped[float]=mapped_column(Float);model:Mapped[str]=mapped_column(String(30));created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=lambda:datetime.now(timezone.utc))
class Model(Base):
 __tablename__='models';name:Mapped[str]=mapped_column(String(30),primary_key=True);path:Mapped[str]=mapped_column(String(300));accuracy:Mapped[float]=mapped_column(Float);precision:Mapped[float]=mapped_column(Float);recall:Mapped[float]=mapped_column(Float);f1:Mapped[float]=mapped_column(Float);trained_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=lambda:datetime.now(timezone.utc))
class Dataset(Base):
 __tablename__='datasets';id:Mapped[int]=mapped_column(primary_key=True);name:Mapped[str]=mapped_column(String(255));rows:Mapped[int]=mapped_column(Integer);real_rows:Mapped[int]=mapped_column(Integer);fake_rows:Mapped[int]=mapped_column(Integer);uploaded_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=lambda:datetime.now(timezone.utc))
class Analyze(BaseModel):headline:str='';body:str=Field(min_length=20);model:str='distilbert'
def session():
 s=Sessions()
 try:yield s
 finally:s.close()
def model_file(name):return MODEL_DIR/f'{name}.joblib'
def load(name):
 path=model_file(name)
 if not path.exists():raise HTTPException(503,f"Model '{name}' is unavailable. Run scripts/train.py or install its versioned model asset.")
 return joblib.load(path)
def explain(pipe,text):
 vec=pipe.named_steps['vectorizer'];clf=pipe.named_steps['classifier'];features=vec.get_feature_names_out();row=vec.transform([text]);weights=clf.coef_[0];idx=row.nonzero()[1];rank=sorted(((features[i],float(weights[i]*row[0,i])) for i in idx),key=lambda x:abs(x[1]),reverse=True)[:12];return [{'token':x,'contribution':round(y,4)}for x,y in rank]
app=FastAPI(title='Fake News Detection',version='2.0.0');app.add_middleware(CORSMiddleware,allow_origins=os.getenv('CORS_ORIGINS','http://localhost:3001,http://127.0.0.1:3001').split(','),allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
@app.on_event('startup')
def init():Base.metadata.create_all(engine)
@app.get('/health')
def health():return {'status':'healthy','available_models':[p.stem for p in MODEL_DIR.glob('*.joblib')]}
@app.post('/predict')
def predict(p:Analyze,s:Session=Depends(session)):
 text=(p.headline+' '+p.body).strip();start=time.perf_counter();pipe=load(p.model);probs=pipe.predict_proba([text])[0];classes=list(pipe.classes_);pos=classes.index('FAKE') if 'FAKE' in classes else int(np.argmax(probs));label=classes[int(np.argmax(probs))];confidence=float(max(probs));s.add(Prediction(headline=p.headline[:300],label=label,confidence=confidence,model=p.model));s.commit();return {'prediction':label,'confidence':confidence,'probabilities':dict(zip(classes,map(float,probs))),'model':p.model,'inference_ms':round((time.perf_counter()-start)*1000,2),'tokens':len(text.split()),'explanation':explain(pipe,text),'explanation_notice':'Feature attribution shows this trained linear baseline’s weighted terms; it is not a complete explanation of language understanding.'}
@app.post('/predict/batch')
async def batch(file:UploadFile=File(...),model:str='distilbert'):
 try:df=pd.read_csv(io.BytesIO(await file.read()))
 except Exception as e:raise HTTPException(400,f'Invalid CSV: {e}')
 text_col=next((c for c in ('text','article','body') if c in df.columns),None)
 if not text_col:raise HTTPException(422,'CSV needs text, article, or body column')
 pipe=load(model);df['prediction']=pipe.predict(df[text_col].fillna('').astype(str));df['confidence']=pipe.predict_proba(df[text_col].fillna('').astype(str)).max(axis=1);out=io.StringIO();df.to_csv(out,index=False);return {'rows':len(df),'csv':out.getvalue()}
@app.post('/datasets/upload')
async def upload(file:UploadFile=File(...),s:Session=Depends(session)):
 try:df=pd.read_csv(io.BytesIO(await file.read()))
 except Exception as e:raise HTTPException(400,f'Invalid CSV: {e}')
 label=next((c for c in ('label','class','target') if c in df.columns),None)
 if not label:raise HTTPException(422,'Dataset needs label, class, or target column')
 values=df[label].astype(str).str.upper();real=int((values=='REAL').sum());fake=int((values=='FAKE').sum());row=Dataset(name=file.filename or 'upload.csv',rows=len(df),real_rows=real,fake_rows=fake);s.add(row);s.commit();return {'dataset_id':row.id,'rows':len(df),'real':real,'fake':fake,'columns':df.columns.tolist(),'duplicates':int(df.duplicated().sum()),'missing':df.isna().sum().to_dict()}
@app.get('/datasets')
def datasets(s:Session=Depends(session)):return [{'id':x.id,'name':x.name,'rows':x.rows,'real':x.real_rows,'fake':x.fake_rows,'uploaded_at':x.uploaded_at.isoformat()}for x in s.scalars(select(Dataset)).all()]
@app.get('/models')
def models(s:Session=Depends(session)):return [{'name':x.name,'accuracy':x.accuracy,'precision':x.precision,'recall':x.recall,'f1':x.f1,'trained_at':x.trained_at.isoformat()}for x in s.scalars(select(Model)).all()]
@app.get('/metrics/{name}')
def metrics(name:str,s:Session=Depends(session)):
 x=s.get(Model,name)
 if not x:raise HTTPException(404,'No recorded metrics for this model')
 return {'model':x.name,'accuracy':x.accuracy,'precision':x.precision,'recall':x.recall,'f1_score':x.f1}
@app.get('/history')
def history(s:Session=Depends(session)):return [{'id':x.id,'headline':x.headline,'prediction':x.label,'confidence':x.confidence,'model':x.model,'timestamp':x.created_at.isoformat()}for x in s.scalars(select(Prediction).order_by(Prediction.created_at.desc()).limit(100)).all()]
@app.delete('/history/{item_id}')
def remove(item_id:int,s:Session=Depends(session)):
 x=s.get(Prediction,item_id)
 if not x:raise HTTPException(404,'History record not found')
 s.delete(x);s.commit();return {'deleted':item_id}
@app.get('/analytics')
def analytics(s:Session=Depends(session)):
 rows=s.scalars(select(Prediction)).all();return {'total_predictions':len(rows),'by_label':{'REAL':sum(x.label=='REAL' for x in rows),'FAKE':sum(x.label=='FAKE' for x in rows)},'average_confidence':float(np.mean([x.confidence for x in rows])) if rows else None}
