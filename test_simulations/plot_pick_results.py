"""Run: python plot_pick_results.py results/pick_trial"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
a = p.parse_args()
with (a.directory/'timeseries.csv').open() as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit('No logged samples.')
def column(name):
    return np.array([float(r[name]) for r in rows])
t = column('time')
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
for i, ax in enumerate(axes.flat, 1):
    q, qh, sigma = column(f'q{i}'), column(f'qhat{i}'), column(f'q_sigma{i}')
    ax.plot(t, q, label='True angle', linewidth=1)
    ax.plot(t, qh, label='Estimated angle', linewidth=1)
    ax.plot(t, column(f'qref{i}'), '--', label='Reference', linewidth=1)
    ax.fill_between(t, qh-2*sigma, qh+2*sigma, alpha=0.25, label='Estimate +/- 2 sigma')
    ax.set_ylabel(f'Joint {i} (rad)'); ax.grid(alpha=0.3)
axes[0,0].legend(fontsize=8)
for ax in axes[-1]: ax.set_xlabel('Simulation time (s)')
fig.tight_layout(); fig.savefig(a.directory/'joint_tracking_uncertainty.png', dpi=160)
fig2, ax = plt.subplots(figsize=(10,4))
ax.plot(t, column('block_height'))
ax.set(xlabel='Simulation time (s)',ylabel='True block height (m)',title='Task evaluation only')
ax.grid(alpha=0.3); fig2.tight_layout()
fig2.savefig(a.directory/'object_lift.png',dpi=160)
print((a.directory/'summary.json').read_text())
plt.show()
