from __future__ import annotations
import json,sys,zipfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));import contract_hardening as C

def test_directory_dataset_identity_is_path_aware_and_order_stable(tmp_path):
 (tmp_path/"b").mkdir();(tmp_path/"a").mkdir();(tmp_path/"b"/"train.csv").write_bytes(b"x,y\n1,2\n");(tmp_path/"a"/"val.csv").write_bytes(b"x,y\n3,4\n");first=C.dataset_identity(tmp_path);assert first and first["kind"]=="tree";assert first["files"]==["a/val.csv","b/train.csv"];assert C.dataset_identity(tmp_path)==first;(tmp_path/"a"/"val.csv").rename(tmp_path/"a"/"other.csv");assert C.dataset_identity(tmp_path)["sha256"]!=first["sha256"]
def test_archive_identity_hashes_exact_selected_zip(tmp_path):
 p=tmp_path/"dataset.zip";
 with zipfile.ZipFile(p,"w") as zf:zf.writestr("train.csv","x,target\n1,a\n")
 assert C.dataset_identity(tmp_path)["sha256"]==C._sha256_file(p)
def test_validator_payload_is_versioned_and_task_locked(tmp_path,monkeypatch):
 monkeypatch.setenv("DIMER_TASK_TYPE","object_detection");cfg=SimpleNamespace(dataset_dir=tmp_path);p={"successful":False,"message":"bad","metadata":{}};C.normalize_payload(p,task_type="tabular_regression",role="validator",cfg=cfg);assert p["schemaVersion"]==1 and p["code"]=="INVALID_DATASET";assert p["metadata"]["taskType"]=="tabular_regression" and p["metadata"]["platformTaskType"]=="object_detection" and p["metadata"]["classNames"]==[]
def test_finetuner_payload_preserves_dataset_identity(tmp_path):
 (tmp_path/"train.csv").write_text("x,target\n1,2\n");cfg=SimpleNamespace(dataset_dir=tmp_path);p={"successful":True,"metadata":{},"provenance":{}};C.normalize_payload(p,task_type="tabular_classification",role="finetuner",cfg=cfg);assert p["code"]=="SUCCEEDED" and p["metadata"]["datasetSha256"]==p["provenance"]["datasetSha256"]
def test_artifact_metadata_versions_secondary_json(tmp_path,monkeypatch):
 out=tmp_path/"out";(out/"artifacts").mkdir(parents=True);(out/"evaluation").mkdir();(out/"logs").mkdir();(out/"progress").mkdir();best=out/"artifacts"/"best.pt"
 with zipfile.ZipFile(best,"w") as zf:zf.writestr("mitra_predictor/predictor.pkl",b"fake")
 for p in [out/"evaluation"/"report.json",out/"logs"/"run-summary.json",out/"progress"/"epoch_0001.json"]:p.write_text("{}")
 cfg=SimpleNamespace(output_dir=out,dataset_dir=tmp_path,target_column="target",drop_columns=[],max_train_rows=10000,validation_split=.2,time_limit=60,seed=0,eval_metric="accuracy",fine_tune=False,fine_tune_steps=0,run_id="r1",session_id="s1",train_device="cpu");arts={"modelArtifact":{"path":"fine-tuning/r1/artifacts/best.pt","name":"best.pt","sizeBytes":best.stat().st_size},"evaluationReport":{"path":"x","name":"report.json","sizeBytes":2},"logArtifact":{"path":"x","name":"run-summary.json","sizeBytes":2}};monkeypatch.setattr(C,"verify_packaged_predictor",lambda *a:None);m=C._harden_artifacts(SimpleNamespace(),cfg,{"mode":"zero-shot"},{"datasetSha256":"abc"},arts,"tabular_classification")["modelArtifact"];assert m["logicalFormat"]=="autogluon-tabular-predictor" and m["packageFormat"]=="zip" and m["reloadVerified"] is True;assert json.loads((out/"evaluation"/"report.json").read_text())["schemaVersion"]==1
