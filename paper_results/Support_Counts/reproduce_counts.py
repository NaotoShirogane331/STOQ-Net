#!/usr/bin/env python3
"""Recompute support counts from the supplied compact snapshots.

Usage: python reproduce_counts.py
Requires numpy and pandas. Does not read/write the original submission package.
"""
from pathlib import Path
import csv
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
with (root/'support_summary.csv').open(encoding='utf-8-sig',newline='') as f:
    expected={(r['table'],r['split']):r for r in csv.DictReader(f)}
with (root/'daily_valid_counts.csv').open(encoding='utf-8-sig',newline='') as f:
    daily=list(csv.DictReader(f))
check_count=0
for split in ['validation','test']:
    with np.load(root/'data'/f'valid_support_{split}.npz',allow_pickle=False) as s:
        ni=len(s['origin_grid_index'])
        grid=s['grid_start_time_utc_minutes']
        local_dates=pd.to_datetime(grid,unit='m',utc=True).tz_convert('Australia/Darwin').tz_localize(None).to_numpy().astype('datetime64[D]').astype(str)
        for table,key in [('Table 7','table7_indices'),('Table 8','table8_indices'),('Table 10','main_daylight_indices')]:
            ids=s[key]
            array_id,within_array=np.divmod(ids,ni*24)
            issue_id,lead0=np.divmod(within_array,24)
            target_index=s['origin_grid_index'][issue_id]+lead0
            frame=pd.DataFrame({'array_index':array_id,'date':local_dates[target_index]})
            unique_days=len(frame.drop_duplicates(['array_index','date']))
            exp=expected[table,split]
            assert len(ids)==int(exp['source_reported_cells'])==int(exp['recomputed_cells'])
            assert len(np.unique(ids))==len(ids)
            assert unique_days==int(exp['target_local_array_days'])
            sid=table.replace(' ','')+'_'+split
            rows=[r for r in daily if r['support_id']==sid]
            got={(int(r['array_index']),r['date']):int(r['cells']) for r in rows}
            assert len(got)==len(rows)==unique_days
            reference=frame.groupby(['array_index','date']).size().to_dict()
            assert got==reference
            assert sum(got.values())==len(ids)
            check_count+=6
            print(f'{table} {split}: {len(ids):,} cells; {unique_days:,} target-local array-days')
print(f'PASS: {check_count} aggregate, uniqueness and every-day comparisons.')
