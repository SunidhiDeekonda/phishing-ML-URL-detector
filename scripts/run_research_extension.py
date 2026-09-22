"""Generate, train, select, and evaluate the seed-42 robustness extension."""

from __future__ import annotations
import json, random, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import joblib, lightgbm as lgb, matplotlib.pyplot as plt, numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from app.inference import URLInference
from src.adversarial_urls import MUTATION_TYPES, MutationRecord, mutate_records
from src.char_tokenizer import encode_urls
from src.continuous_learning import ModelRegistry
from src.drift_monitor import create_drift_report, save_drift_report
from src.features import build_features_dataframe
from src.train_charcnn import CharCNN

SEED = 42
# CPU is deliberately used here: the Python 3.14/PyTorch MPS combination can
# stall during repeated validation, while CPU execution is deterministic.
DEVICE = torch.device("cpu")


def dump(path, value): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
def score(y, p):
    pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"roc_auc":float(roc_auc_score(y,p)),"pr_auc":float(average_precision_score(y,p)),"false_positives":int(fp),"false_negatives":int(fn),"fpr":float(fp/max(fp+tn,1)),"fnr":float(fn/max(fn+tp,1))}
def suite(data, idx):
    selected=data.iloc[idx]; rows=[]
    for i,r in selected[selected.label==0].iterrows(): rows.append(MutationRecord(int(i),str(r.url),str(r.url),"none",0).to_dict())
    rows += [r.to_dict() for r in mutate_records([(int(i),str(r.url),1) for i,r in selected[selected.label==1].iterrows()])]
    return pd.DataFrame(rows).sort_values("source_row_index").reset_index(drop=True)
def original_inputs(service, x, encoded):
    bundle=service._bundle
    matrix=x[bundle.feature_columns].to_numpy(dtype=np.float32)
    lgb_output=bundle.lightgbm_model.session.run(None,{"input":matrix})[1]
    raw=np.array([float(item[1]) for item in lgb_output])
    calibrated=1/(1+np.exp(np.clip(bundle.lightgbm_model.calibration_a*raw+bundle.lightgbm_model.calibration_b,-40,40)))
    logits=bundle.cnn_model.run(None,{"input_ids":np.asarray(encoded,dtype=np.int64)})[0].reshape(-1)
    cnn=1/(1+np.exp(-np.clip(logits,-40,40)))
    return {"lightgbm":calibrated,"char_cnn":cnn,"reference_60_40":.6*cnn+.4*calibrated,"selected_95_5":.95*cnn+.05*calibrated}
def suite_inputs(frame, frozen_features, frozen_sequences, vocab):
    urls=frame.mutated_url.astype(str).tolist()
    x=build_features_dataframe(urls)[list(frozen_features.columns)]
    encoded=encode_urls(urls,vocab)
    untouched=np.flatnonzero(frame.mutation_type.to_numpy()=="none")
    source=frame.iloc[untouched].source_row_index.to_numpy(dtype=int)
    if len(untouched):
        x.iloc[untouched]=frozen_features.iloc[source].to_numpy()
        encoded[untouched]=frozen_sequences[source]
    return x,encoded
def train_lgb(x,y,xv,yv):
    model=lgb.LGBMClassifier(objective="binary",n_estimators=1000,learning_rate=.03,num_leaves=31,subsample=.9,colsample_bytree=.9,reg_lambda=1,random_state=SEED,deterministic=True,force_col_wise=True,verbosity=-1,n_jobs=1)
    model.fit(x,y,eval_set=[(xv,yv)],eval_metric="auc",callbacks=[lgb.early_stopping(50,verbose=False)])
    raw=np.clip(model.predict_proba(xv)[:,1],1e-6,1-1e-6); cal=LogisticRegression(random_state=SEED).fit(np.log(raw/(1-raw)).reshape(-1,1),yv)
    return {"model":model,"calibrator":cal,"feature_columns":list(x.columns),"seed":SEED}
def lgbp(bundle,x):
    raw=np.clip(bundle["model"].predict_proba(x[bundle["feature_columns"]])[:,1],1e-6,1-1e-6)
    return bundle["calibrator"].predict_proba(np.log(raw/(1-raw)).reshape(-1,1))[:,1]
def cnnp(model, seq, batch=512):
    model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(seq),batch): out.append(torch.sigmoid(model(torch.as_tensor(seq[start:start+batch],dtype=torch.long,device=DEVICE)).reshape(-1)).cpu().numpy())
    return np.concatenate(out)
def train_cnn(train_seq,y,val_seq,yv):
    ckpt=torch.load(ROOT/"models/char_cnn.pt",map_location="cpu",weights_only=False); model=CharCNN().to(DEVICE); model.load_state_dict(ckpt["model_state_dict"])
    loader=DataLoader(TensorDataset(torch.as_tensor(train_seq,dtype=torch.long),torch.as_tensor(y,dtype=torch.float32)),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); loss_fn=nn.BCEWithLogitsLoss(); best=(-1,None,0)
    for epoch in range(1,4):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(set_to_none=True); loss_fn(model(xb).reshape(-1),yb).backward(); opt.step()
        auc=roc_auc_score(yv,cnnp(model,val_seq))
        if auc>best[0]: best=(auc,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch)
    model.load_state_dict(best[1]); return model,best[2],best[0]
def replace_section(path,heading,body):
    text=path.read_text(); marker=f"\n## {heading}\n"
    if marker in text: text=text.split(marker,1)[0].rstrip()+"\n"
    path.write_text(text+marker+"\n"+body.strip()+"\n")
def write_docs(payload,aug,n):
    b=payload["adversarial_test"]["original_selected_95_5"]; r=payload["adversarial_test"]["robust_candidate_95_5"]; c=payload["clean_test"]["robust_candidate_95_5"]; delta=r["recall"]-b["recall"]
    novelty=f"""# Novelty and Research Extension

## Reference paper implemented
Character-level CNN, 36 engineered URL features, LightGBM, and a weighted ensemble. It reported approximately 99.819% accuracy, 100% precision, 99.635% recall, and 99.947% ROC-AUC.

## Original reproduction
Independent training on a deterministic 20,000-URL subset, strict root-domain-separated splits, validation-only ensemble selection, held-out evaluation, and a network-free FastAPI demo.

## New implemented extensions
- Eight deterministic offline mutation categories with seed 42.
- Conservative adversarial training with {aug} mutations derived only from training phishing rows.
- Separate clean and {n}-row adversarial held-out evaluation.
- Safe caller-supplied HTML/email context, never mixed into validated probability.
- Quarantined human-verified feedback with deduplication and test protection.
- PSI drift monitoring and a versioned validation-gated model registry.

Original adversarial recall was {b['recall']:.4%} with {b['false_negatives']} false negatives; robust-candidate recall was {r['recall']:.4%} with {r['false_negatives']} false negatives ({delta:+.4%}). Robust clean accuracy was {c['accuracy']:.4%}. The contribution is robustness and controlled adaptability, not a claim of higher saturated precision. Synthetic results do not prove protection against every attacker.
"""
    (ROOT/"docs/NOVELTY_AND_EXTENSION.md").write_text(novelty)
    q=["Paper already has 100% precision. What is new?","Is this copied?","Which code did you implement?","Why 20,000 URLs?","How prevent domain leakage?","What are the 36 features?","What does CNN add?","Why 95/5?","Why did LightGBM score higher cleanly?","Why keep validation selection?","What is adversarial training?","How generate mutations?","Did you visit phishing sites?","Could mutations leak?","How did robustness change?","Did clean performance decrease?","Why not retrain every report?","What is model poisoning?","How does validation gating work?","How does drift monitoring work?","Is context in validated accuracy?","Why not fetch HTML?","How is deployment different?","Biggest limitations?","What next?"]
    proofs=["docs/NOVELTY_AND_EXTENSION.md","RUN_LOG.md","src/adversarial_urls.py","docs/FINAL_REPORT.md","tests/test_phase1.py","src/features.py","src/train_charcnn.py","src/select_ensemble.py","results/test_metrics.json","docs/FINAL_REPORT.md","scripts/run_research_extension.py","src/adversarial_urls.py","src/adversarial_urls.py","data/processed/adversarial_test.csv","results/robustness_final_metrics.json","results/robustness_selection.json","src/continuous_learning.py","src/continuous_learning.py","results/robustness_selection.json","src/drift_monitor.py","src/context_features.py","app/research_api.py","app/main.py","docs/NOVELTY_AND_EXTENSION.md","docs/FINAL_REPORT.md"]
    short=["Robustness and adaptability, not higher precision.","No: independent reproduction plus measured extensions.","The reproducibility and extension pipeline.","Feasible deterministic college-scale study.","Root domains are split-disjoint.","Lexical and structural URL measurements.","Learned local character motifs.","Validation-only selection chose it.","Engineered signals were strong here.","To avoid test-set tuning.","Training with label-preserving perturbations.","Eight seeded string transformations.","No; URLs remained inert strings.","No; sources remain split-specific.",f"Recall changed {delta:+.4%}.",f"Robust clean accuracy was {c['accuracy']:.4%}.","Unverified feedback enables poisoning.","Maliciously corrupting training feedback.","Mean clean/adversarial AUC plus clean floor.","PSI over all 36 features.","No; it is separate evidence.","Safety, privacy, SSRF, reproducibility.","Safe extension APIs and controls.","Synthetic attacks and no labeled content corpus.","External temporal and labeled content evaluation."]
    parts=["# Viva / Professor Defense","","## Part 1 — 60-second project explanation","","We reproduced a CNN-LightGBM phishing URL detector with domain-separated splits, then added deterministic adversarial stress testing/training, safe context evidence, verified feedback, drift monitoring, and model versioning. No submitted URL is fetched.","","## Part 2 — 3-minute technical explanation","",f"The CNN consumes 200 characters and LightGBM consumes 36 features. Eight seed-42 mutation families augment training only; validation freezes selection before a {n}-row adversarial test. Original adversarial recall was {b['recall']:.4%}; robust recall was {r['recall']:.4%}. Context stays separate because no labeled content corpus supports calibrated fusion."]
    sections=[("Part 3 — What was in the original paper?","CNN, 36 URL features, LightGBM, weighted ensemble."),("Part 4 — What did we reproduce?","Independent preprocessing, domain splits, training, selection, test, deployment."),("Part 5 — What is actually new?","Robustness benchmark/training, context, verified adaptation, drift, versioning."),("Part 6 — Why not higher precision?","The reference already reports 100%; novelty answers different questions."),("Part 7 — Adversarial robustness","Label-preserving surface changes; clean and adversarial tests stay separate."),("Part 8 — Content/context","Only supplied text is parsed; it is not probability fusion."),("Part 9 — Continuous learning","Quarantine, human approval, deduplication, test protection, offline retraining, validation gate."),("Part 10 — Data leakage","Domains and mutation sources remain split-specific."),("Part 11 — Clean vs adversarial test","Ordinary generalization versus declared perturbation resilience."),("Part 12 — Why never fetch URLs","Avoid harm, SSRF, privacy loss, and nondeterminism."),("Part 13 — Limitations","Synthetic coverage, fixed subset, heuristic context, ephemeral serverless feedback."),("Part 14 — Files to open","`src/adversarial_urls.py`, `scripts/run_research_extension.py`, `src/context_features.py`, `src/continuous_learning.py`, `src/drift_monitor.py`, `results/robustness_final_metrics.json`, `tests/test_research_extensions.py`.")]
    for h,t in sections: parts += ["",f"## {h}","",t]
    parts += ["","## Part 15 — 25 likely professor questions",""]
    for i,(question,answer,proof) in enumerate(zip(q,short,proofs),1): parts += [f"### {i}. {question}","",f"**Short answer:** {answer}","",f"**Detailed follow-up:** {answer} The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.","",f"**Proof to open:** `{proof}`",""]
    (ROOT/"docs/VIVA_DEFENSE.md").write_text("\n".join(parts))
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data=pd.read_csv(ROOT/"data/processed/dataset.csv"); feats=pd.read_csv(ROOT/"data/processed/features.csv"); seq=np.load(ROOT/"data/processed/char_sequences.npy"); vocab=json.loads((ROOT/"data/processed/char_vocab.json").read_text())
    train=np.load(ROOT/"data/processed/train_idx.npy"); val=np.load(ROOT/"data/processed/val_idx.npy"); test=np.load(ROOT/"data/processed/test_idx.npy")
    av,at=suite(data,val),suite(data,test); av.to_csv(ROOT/"data/processed/adversarial_validation.csv",index=False); at.to_csv(ROOT/"data/processed/adversarial_test.csv",index=False)
    service=URLInference(); service.load_models(); yav=av.label.to_numpy(int); avf,avs=suite_inputs(av,feats,seq,vocab); baseav=original_inputs(service,avf,avs); dump(ROOT/"results/adversarial_baseline_validation.json",{k:score(yav,v) for k,v in baseav.items()})
    phishing=data.iloc[train]; phishing=phishing[phishing.label==1]; aug=round(len(phishing)*.25); sampled=phishing.sample(n=aug,random_state=SEED); muts=mutate_records([(int(i),str(r.url),1) for i,r in sampled.iterrows()]); urls=[r.mutated_url for r in muts]
    ytrain=np.r_[data.iloc[train].label.to_numpy(int),np.ones(aug,int)]; xtrain=pd.concat([feats.iloc[train].reset_index(drop=True),build_features_dataframe(urls)],ignore_index=True)
    yval=np.r_[data.iloc[val].label.to_numpy(int),yav]; xval=pd.concat([feats.iloc[val].reset_index(drop=True),avf],ignore_index=True)
    rl=train_lgb(xtrain[list(feats.columns)],ytrain,xval,yval); joblib.dump(rl,ROOT/"models/lightgbm_robust.pkl")
    trainseq=np.concatenate([seq[train],encode_urls(urls,vocab)]); valseq=np.concatenate([seq[val],encode_urls(av.mutated_url.astype(str),vocab)]); rc,epoch,cauc=train_cnn(trainseq,ytrain,valseq,yval); torch.save({"model_state_dict":rc.state_dict(),"seed":SEED,"best_epoch":epoch,"combined_validation_roc_auc":cauc,"architecture":"CharCNN"},ROOT/"models/char_cnn_robust.pt")
    ycv=data.iloc[val].label.to_numpy(int); ocv=original_inputs(service,feats.iloc[val],seq[val])["selected_95_5"]; rcv=.95*cnnp(rc,seq[val])+.05*lgbp(rl,feats.iloc[val]); rav=.95*cnnp(rc,avs)+.05*lgbp(rl,avf)
    oa,ob,ra,rb=roc_auc_score(ycv,ocv),roc_auc_score(yav,baseav["selected_95_5"]),roc_auc_score(ycv,rcv),roc_auc_score(yav,rav); selected="robust_candidate_95_5" if (ra+rb)/2>(oa+ob)/2 and ra>=oa-.002 else "original_selected_95_5"
    selection={"frozen_before_adversarial_test":True,"criterion":"mean clean/adversarial validation ROC-AUC; clean AUC floor original minus 0.002","original":{"clean_validation_roc_auc":oa,"adversarial_validation_roc_auc":ob,"selection_score":(oa+ob)/2},"robust_candidate":{"clean_validation_roc_auc":ra,"adversarial_validation_roc_auc":rb,"selection_score":(ra+rb)/2},"selected":selected,"seed":SEED}; dump(ROOT/"results/robustness_selection.json",selection)
    yct=data.iloc[test].label.to_numpy(int); oct=original_inputs(service,feats.iloc[test],seq[test])["selected_95_5"]; rct=.95*cnnp(rc,seq[test])+.05*lgbp(rl,feats.iloc[test]); yat=at.label.to_numpy(int); atf,ats=suite_inputs(at,feats,seq,vocab); oat=original_inputs(service,atf,ats)["selected_95_5"]; rat=.95*cnnp(rc,ats)+.05*lgbp(rl,atf)
    payload={"configuration":{"seed":SEED,"mutation_types":list(MUTATION_TYPES),"training_augmentation_count":aug,"selected_by_validation":selected},"clean_test":{"original_selected_95_5":score(yct,oct),"robust_candidate_95_5":score(yct,rct)},"adversarial_test":{"original_selected_95_5":score(yat,oat),"robust_candidate_95_5":score(yat,rat)}}; dump(ROOT/"results/robustness_final_metrics.json",payload)
    pd.DataFrame([{"suite":s,"model":m,**v} for s in ("clean_test","adversarial_test") for m,v in payload[s].items()]).to_csv(ROOT/"results/robustness_comparison.csv",index=False)
    plots=ROOT/"results/plots"; plots.mkdir(parents=True,exist_ok=True); names=["Original\nclean","Robust\nclean","Original\nadversarial","Robust\nadversarial"]; acc=[payload["clean_test"]["original_selected_95_5"]["accuracy"],payload["clean_test"]["robust_candidate_95_5"]["accuracy"],payload["adversarial_test"]["original_selected_95_5"]["accuracy"],payload["adversarial_test"]["robust_candidate_95_5"]["accuracy"]]; rec=[payload["clean_test"]["original_selected_95_5"]["recall"],payload["clean_test"]["robust_candidate_95_5"]["recall"],payload["adversarial_test"]["original_selected_95_5"]["recall"],payload["adversarial_test"]["robust_candidate_95_5"]["recall"]]; x=np.arange(4); fig,ax=plt.subplots(figsize=(9,5)); ax.bar(x-.18,acc,.36,label="Accuracy"); ax.bar(x+.18,rec,.36,label="Recall"); ax.set_ylim(.8,1.005); ax.set_xticks(x,names); ax.legend(); ax.set_title("Clean and adversarial held-out performance"); fig.tight_layout(); fig.savefig(plots/"robustness_comparison.png",dpi=180); plt.close(fig)
    matrix=confusion_matrix(yat,(rat>=.5).astype(int)); fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(matrix,cmap="Blues"); ax.set_xticks([0,1],["Legitimate","Phishing"]); ax.set_yticks([0,1],["Legitimate","Phishing"]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Robust candidate: adversarial test"); [ax.text(j,i,str(v),ha="center",va="center") for (i,j),v in np.ndenumerate(matrix)]; fig.tight_layout(); fig.savefig(plots/"adversarial_confusion_matrix.png",dpi=180); plt.close(fig)
    registry=ModelRegistry(ROOT/"models/model_registry.json"); registry.register({"version":"url-ensemble-original-1.0","training_date":"2026-08-28","data_count":int(len(train)),"clean_validation_metrics":{"roc_auc":oa},"robustness_validation_metrics":{"roc_auc":ob},"status":"production" if selected.startswith("original") else "superseded"}); registry.register({"version":"url-ensemble-robust-1.0","training_date":datetime.now(timezone.utc).date().isoformat(),"data_count":int(len(ytrain)),"clean_validation_metrics":{"roc_auc":ra},"robustness_validation_metrics":{"roc_auc":rb},"status":"validated_candidate" if selected.startswith("robust") else "rejected_by_validation_gate"})
    save_drift_report(create_drift_report(feats.iloc[train],[]),ROOT/"results/drift_report.json"); write_docs(payload,aug,len(at)); body=f"""Eight deterministic offline mutation families were evaluated. Only {aug} training-derived phishing mutations entered robust training; validation froze selection before the {len(at)}-row adversarial test.

Original adversarial recall was {payload['adversarial_test']['original_selected_95_5']['recall']:.3%} with {payload['adversarial_test']['original_selected_95_5']['false_negatives']} false negatives. The robust candidate achieved {payload['adversarial_test']['robust_candidate_95_5']['recall']:.3%} with {payload['adversarial_test']['robust_candidate_95_5']['false_negatives']} false negatives and {payload['clean_test']['robust_candidate_95_5']['accuracy']:.3%} clean accuracy. Validation selected `{selected}`. These are synthetic-mutation results, not universal protection.

Caller-supplied HTML/email produces separate context evidence and is never fetched or mixed into validated probability. Feedback is quarantined for human verification, deduplicated, protected from test contamination, and intended for explicit offline retraining. PSI and model versioning support controlled adaptation. Vercel local feedback storage is ephemeral unless external persistence is configured."""; replace_section(ROOT/"README.md","Research Extensions Beyond Reference Reproduction",body); replace_section(ROOT/"docs/FINAL_REPORT.md","Extensions Beyond the Reference Study",body); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
