"""Plot all recorded development histories, without running a model."""
from pathlib import Path
import json,csv,os
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reproduction_outputs/convergence';OUT.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig,axs=plt.subplots(1,2,figsize=(10.5,3.9),layout='constrained');rows=[];summary=[]
for seed in range(5101,5106):
    h=json.loads((ROOT/f'outputs/e6/modern_recovery/training/s{seed}_epochs.json').read_text())['history']
    assert len(h)==120
    best=min(h,key=lambda r:r['validation_mse'])
    summary.append(dict(seed=seed,epochs=len(h),best_epoch=best['epoch'],best_validation_mse=best['validation_mse'],final_validation_mse=h[-1]['validation_mse']))
    for a,key,label in zip(axs,['train_mse','validation_mse'],['Training MSE','Validation MSE']):
        a.plot([r['epoch'] for r in h],[r[key] for r in h],label=str(seed),linewidth=1.3)
        a.set(xlabel='Recorded epoch',ylabel=label,yscale='log');a.grid(alpha=.2)
    rows.extend(dict(seed=seed,**r) for r in h)
axs[1].legend(title='Seed',fontsize=9,ncol=2)
fig.savefig(OUT/'helena_all_five_histories.png',dpi=220);fig.savefig(OUT/'helena_all_five_histories.pdf');plt.close(fig)
for name,data in [('all_epochs.csv',rows),('checkpoint_summary.csv',summary)]:
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
print('PASS: all 600 recorded epochs plotted; seed 5104 retained; no training or simulation.')
