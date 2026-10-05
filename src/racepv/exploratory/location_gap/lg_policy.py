"""Historical fixed development rows only; no process/SSH/clock policy."""

def candidates():
    return [dict(name=tag+'-s'+str(seed),architecture=arch,display_model_id=arch,seed=seed,
        epochs=1,batch_size=64,learning_rate=.001,weight_decay=.0001,temperature=1.,
        precision='amp_float16',training_loss_protocol='original_projected_qcrps')
        for seed in (29,47) for tag,arch in [('C0','C'),('C-LG','C-LG')]]

def validate_candidate(row):
    if row not in candidates():raise ValueError('outside approved four-group screen')
    return row
