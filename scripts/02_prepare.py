"""Run from project root: .venv/bin/python scripts/02_prepare.py --help"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.prepare import build,make_splits,prepare_fold,write_fold
from src.feature_map import field_map

def main():
    p=argparse.ArgumentParser(description='P2 feature preparation; no final model training')
    p.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw');p.add_argument('--out-dir',type=Path,default=ROOT/'data/processed');p.add_argument('--split-dir',type=Path,default=ROOT/'data/splits')
    p.add_argument('--scope',choices=['target','common'],default='target');p.add_argument('--with-source',action='store_true');p.add_argument('--source-limit',type=int)
    p.add_argument('--fold',choices=['0','1','2','3','4','time','all'],default='0');p.add_argument('--no-dedup',action='store_true');p.add_argument('--no-derived',action='store_true');p.add_argument('--reduction',choices=['pca','svd','none'],default='pca');p.add_argument('--early-policy',choices=['rank','raw'],default='rank');p.add_argument('--include-domain',action='store_true');p.add_argument('--exclude-early',action='store_true');p.add_argument('--format',choices=['parquet','csv'],default='parquet');p.add_argument('--run-name',default='p2_target');p.add_argument('--report-dir',type=Path,default=ROOT/'reports/tables')
    a=p.parse_args();mapping=field_map(); target,source,test=build(a.raw_dir,a.out_dir,a.format,mapping);folds,time=make_splits(target,a.split_dir)
    print('Aligned shapes:',target.shape,source.shape,test.shape,flush=True)
    selected=folds+[time] if a.fold=='all' else [time if a.fold=='time' else folds[int(a.fold)]]
    for split in selected:
        out=a.out_dir/a.run_name/f'fold_{split["fold"]}'
        result=prepare_fold(target,source,test,split,mapping,scope=a.scope,use_source=a.with_source,dedup=not a.no_dedup,derived=not a.no_derived,reduction=a.reduction,early_policy=a.early_policy,include_domain=a.include_domain,exclude_early=a.exclude_early,source_limit=a.source_limit)
        write_fold(result,out,a.format)
        report=a.report_dir/a.run_name/f'fold_{split["fold"]}';report.mkdir(parents=True,exist_ok=True)
        for k in ['feat_audit','psi_train_fold','dup_clusters','dup_counts','reduction']:
            ext='.json' if k=='reduction' else '.csv';(report/(k+ext)).write_bytes((out/(k+ext)).read_bytes())
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()};(out/'settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
        print(out, {k:v.shape for k,v in result['matrices'].items()},result['reduction'],flush=True)
if __name__=='__main__':main()
