"""Historical fixed development rows only; no process/SSH/clock policy."""

def candidates():
    return [dict(name='C2-'+tag+'-s'+str(s),model='C2',variant=v,epochs=1,seed=s,
                 batch_size=64,learning_rate=.001,weight_decay=.0001,precision='amp_float16')
            for s in (29,47) for tag,v in (('MS','multiscale'),('SS','single_scale'))]

def validate_candidate(row):
    if row not in candidates(): raise ValueError('candidate outside approved C2 four-run probe')
    return row
