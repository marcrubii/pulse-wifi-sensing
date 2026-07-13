import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from features import window_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def load_esp32_csv(path):
    df=pd.read_csv(path); W=[]
    for s in df['CSI_DATA']:
        v=np.fromstring(s.strip('[]'),sep=' '); W.append(v[1::2]+1j*v[0::2])
    return np.array(W)

def make_windows(csi, size=128):
    n=len(csi)//size
    return [csi[i*size:(i+1)*size] for i in range(n)]

X,y=[],[]
for path,label in [('data/raw/sample_quieto.csv',0),('data/raw/sample_movimiento.csv',1)]:
    for win in make_windows(load_esp32_csv(path)):
        X.append(window_features(win)); y.append(label)
X=np.array(X); y=np.array(y)
print("Dataset:",X.shape,"clases:",np.bincount(y))
clf=make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=200,random_state=0))
print("accuracy (cv5):",round(cross_val_score(clf,X,y,cv=5).mean(),3))
