"""Historical fixed development rows only; no process/SSH/clock policy."""

def candidates():
    return [dict(name=tag+'-s'+str(s),architecture=arch,display_model_id=display,seed=s,
        epochs=1,batch_size=64,learning_rate=.001,weight_decay=.0001,temperature=1.,
        precision='amp_float16',training_loss_protocol=loss)
        for s in (29,47) for tag,arch,display,loss in
        [('C0','C','C','original_projected_qcrps'),('C3','C3','C3-TimeXer-PV','raw_ordered_qcrps')]]

def validate_candidate(row):
    if row not in candidates():raise ValueError('outside approved C0/C3 matrix')
    return row
