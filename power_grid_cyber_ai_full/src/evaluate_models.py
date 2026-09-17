import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_DIR
def main():
 d=pd.read_csv(RESULTS_DIR/'predictions_test.csv');labels=['Normal','Fault','Attack'];(RESULTS_DIR/'confusion_matrices').mkdir(parents=True,exist_ok=True);(RESULTS_DIR/'classification_reports').mkdir(parents=True,exist_ok=True);rows=[]
 for name,col in [('random_forest','rf_prediction'),('xgboost','xgb_prediction')]:
  cm=confusion_matrix(d.y_true,d[col],labels=labels);pd.DataFrame(cm,index=labels,columns=labels).to_csv(RESULTS_DIR/'confusion_matrices'/f'{name}.csv');r=classification_report(d.y_true,d[col],labels=labels,output_dict=True,zero_division=0);pd.DataFrame(r).T.to_csv(RESULTS_DIR/'classification_reports'/f'{name}.csv');rows.append({'model':name,'attack_as_fault':int(((d.y_true=='Attack')&(d[col]=='Fault')).sum()),'fault_as_attack':int(((d.y_true=='Fault')&(d[col]=='Attack')).sum()),'macro_f1':r['macro avg']['f1-score']})
 pd.DataFrame(rows).to_csv(RESULTS_DIR/'fault_attack_analysis.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
