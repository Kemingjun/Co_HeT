"""Rebuild industrial BKS and summaries from saved results, without optimization."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'docs/experiments/industrial'
METRICS=('objective','travel_time','tardiness','runtime_s')

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def build(data=DATA):
    rows=[]
    for family,name,count in [('drl','per_instance_results.csv',2400),('alns','per_rep_results.csv',2000),('gurobi','per_instance_results.csv',300)]:
        selected=read_csv(data/family/name)
        assert len(selected)==count,(family,len(selected))
        for row in selected:
            r=dict(row); r['size']=int(r['size']);r['instance_index']=int(r['instance_index'])
            r['repetition']=int(r.get('repetition') or 1)
            assert r['dataset_id']=='real_world_test100_seed20260906'
            for key in METRICS:r[key]=float(r[key]);assert math.isfinite(r[key])
            assert abs(r['objective']-.5*(r['travel_time']+r['tardiness']))<=1e-6
            rows.append(r)
    identities={(r['method'],r['size'],r['instance_index'],r['repetition']) for r in rows}
    assert len(identities)==4700
    for r in rows:
        assert r['size'] in (10,20,30,40) and 1 <= r['instance_index'] <= 100
        assert r['instance']==f"RW_N{r['size']}_K3_M16_I{r['instance_index']}.xlsx"
        assert (ROOT/'instances/real_world_test100_seed20260906'/r['instance']).is_file()
    bks={}
    for r in rows:
        key=(r['size'],r['instance_index'])
        bks[key]=min(bks.get(key,math.inf),r['objective'])
    assert len(bks)==400
    grouped=defaultdict(list)
    for r in rows:grouped[r['method'],r['size'],r['instance_index']].append(r)
    per_instance=[]
    for (method,n,i),runs in grouped.items():
        assert len(runs)==(5 if method=='ALNS' else 1)
        row=dict(method=method,size=n,instance_index=i,repetitions=len(runs),bks=bks[n,i])
        row.update({m:statistics.mean(r[m] for r in runs) for m in METRICS})
        row['rpd']=100*(row['objective']-row['bks'])/row['bks']
        per_instance.append(row)
    summaries=[]
    for method,n in sorted({(r['method'],r['size']) for r in per_instance}):
        cell=[r for r in per_instance if r['method']==method and r['size']==n]
        assert len(cell)==100
        row=dict(method=method,size=n,instances=len(cell))
        for m in METRICS+('rpd',):
            row[m+'_mean']=statistics.mean(r[m] for r in cell)
            row[m+'_sd']=statistics.stdev(r[m] for r in cell)
        summaries.append(row)
    return per_instance,summaries,[dict(size=n,instance_index=i,bks=v) for (n,i),v in sorted(bks.items())]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); tables=build()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name,rows in zip(['per_instance.csv','summary.csv','bks.csv'],tables):
        with (args.output_dir/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'raw_records':4700,'instances':400,'method_size_cells':len(tables[1]),'rows':len(tables[0])}))

if __name__=='__main__':main()
