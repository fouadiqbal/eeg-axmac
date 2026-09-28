from pathlib import Path
import os, time, json, hashlib, sys
import numpy as np, pandas as pd, tensorflow as tf, tf_keras as keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
assert Path('/kaggle').exists(), 'Kaggle only'
base=Path('/kaggle/working/bciiv2a_eegtcnet_validation')
repo=base/'eeg-tcnet'
sys.path.insert(0,str(repo))
from utils.data_loading import prepare_features
run=Path('/kaggle/working/bciiv2a_eegnet82_session'); run.mkdir(exist_ok=True)
config=dict(model='EEGNet-8,2',kernel=64,pools=[8,8],dropout=.5,epochs=750,batch=64,lr=.001,seed=42,reference=.724,protocol='artifact-rejected official loader, T->E, train-only per-channel/time scaling',qualification='near reproduction; comparator-specific training code not released')
manifest=run/'config.json'
if manifest.exists(): assert json.loads(manifest.read_text())==config
else: manifest.write_text(json.dumps(config,indent=2))
# Preserve the released backbone; change legacy import paths only.
source=(repo/'utils/models.py').read_text()
source=chr(10).join(line for line in source.splitlines() if not line.startswith('from '))
imports=chr(10).join(['from tf_keras.models import Model','from tf_keras.layers import Dense,Activation,Conv1D,Conv2D,AveragePooling2D,SeparableConv2D,BatchNormalization,Dropout,Add,Lambda,DepthwiseConv2D,Input,Permute','from tf_keras.constraints import max_norm',''])
ns={}; exec(imports+source,ns)
def make_net():
    inp=keras.Input((1,22,1125))
    x=keras.layers.Permute((3,2,1))(inp)
    x=ns['EEGNet'](input_layer=x,F1=8,kernLength=64,D=2,Chans=22,dropout=.5)
    x=keras.layers.Flatten()(x)
    out=keras.layers.Dense(4,activation='softmax',kernel_constraint=keras.constraints.max_norm(.25),name='classifier')(x)
    model=keras.Model(inp,out)
    model.compile(optimizer=keras.optimizers.Adam(.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model
class Persist(keras.callbacks.Callback):
    def __init__(self,folder): super().__init__(); self.folder=folder; self.start=time.perf_counter()
    def on_epoch_end(self,epoch,logs=None):
        if (epoch+1)%50==0:
            checkpoint=self.folder/f'epoch_{epoch+1:04d}.keras'
            tmp=self.folder/f'epoch_{epoch+1:04d}.pending.keras'
            self.model.save(tmp); os.replace(tmp,checkpoint)
            state=dict(epoch=epoch+1,elapsed_seconds=time.perf_counter()-self.start,logs={k:float(v) for k,v in (logs or {}).items()})
            state_tmp=self.folder/f'epoch_{epoch+1:04d}.pending.json'
            state_tmp.write_text(json.dumps(state)); os.replace(state_tmp,self.folder/f'epoch_{epoch+1:04d}.json')
            print('SUBJECT',self.folder.name,'EPOCH',epoch+1,'SECONDS',round(state['elapsed_seconds'],1),flush=True)
records=[]
for subject in range(1,10):
    folder=run/f'subject{subject}';folder.mkdir(exist_ok=True)
    result=folder/'result.json'
    if result.exists(): records.append(json.loads(result.read_text())); print('REUSE',subject,flush=True); continue
    keras.backend.clear_session(); keras.utils.set_random_seed(42+subject)
    Xtr,_,ytr,Xte,_,yte=prepare_features(str(base/'data')+'/',subject-1,False)
    means=[];scales=[]
    for ch in range(22):
        scaler=StandardScaler().fit(Xtr[:,0,ch,:]);means.append(scaler.mean_);scales.append(scaler.scale_)
        Xtr[:,0,ch,:]=scaler.transform(Xtr[:,0,ch,:]);Xte[:,0,ch,:]=scaler.transform(Xte[:,0,ch,:])
    Xtr=Xtr.astype('float32');Xte=Xte.astype('float32')
    np.savez_compressed(folder/'preprocessed.npz',X_train=Xtr,y_train=ytr.argmax(1),X_test=Xte,y_test=yte.argmax(1),mean=means,scale=scales)
    initial=0
    complete=sorted(p for p in folder.glob('epoch_*.json') if p.with_suffix('.keras').exists())
    if complete:
        latest=complete[-1]; initial=json.loads(latest.read_text())['epoch']
        model=keras.models.load_model(latest.with_suffix('.keras'))
    else: model=make_net()
    print('START',subject,'PARAMS',model.count_params(),'FROM_EPOCH',initial,flush=True)
    model.fit(Xtr,ytr,batch_size=64,epochs=750,initial_epoch=initial,shuffle=True,verbose=0,callbacks=[Persist(folder)])
    pred=model.predict(Xte,batch_size=64,verbose=0).argmax(1); y=yte.argmax(1)
    row=dict(subject=subject,accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),n_train=len(Xtr),n_test=len(Xte),seed=42+subject,epochs=750)
    model.save(folder/'final.keras'); np.savez_compressed(folder/'predictions.npz',y_true=y,y_pred=pred)
    result.write_text(json.dumps(row,indent=2));records.append(row)
    pd.DataFrame(records).to_csv(run/'subject_results.partial.csv',index=False); print('COMPLETED',row,flush=True)
results=pd.DataFrame(records);results.to_csv(run/'subject_results.csv',index=False)
summary=dict(mean_accuracy=float(results.accuracy.mean()),std_accuracy=float(results.accuracy.std()),mean_macro_f1=float(results.macro_f1.mean()),reference=.724,within_3pp=bool(abs(results.accuracy.mean()-.724)<=.03),protocol_qualification=config['qualification'])
(run/'summary.json').write_text(json.dumps(summary,indent=2)); print('EEGNET_SESSION_SUMMARY',summary,flush=True)
