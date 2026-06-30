#!/usr/bin/env python
"""Generate fingerprints.jpg, a conceptual 3D fingerprint-layout figure.
The file existed in Paper-Charts but no generating cell/filename reference was
found in all-original-scripts, so this script reconstructs it from the reference
point coordinate grid used in the manuscript.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper_outputs" / "figures" / "fingerprints.jpg"
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family":"Times New Roman", "axes.linewidth": 1.2})
points=[]
for prefix,y in [('A',0.0),('C',2.4),('E',6.08)]:
    for i in range(1,24,2):
        x=2.4*(i//2) + (0.05 if prefix=='E' else 0.0)
        points.append((f"{prefix}{i}",x,y))
fig=plt.figure(figsize=(16,8),dpi=220)
ax=fig.add_subplot(111,projection='3d')
for label,x,y in points:
    ax.scatter(x,y,0,s=170,c='green',alpha=.80,edgecolors='darkgreen',linewidths=1.1)
    ax.text(x,y,0.18,label,fontsize=14,ha='center',va='bottom')
# central corridor
xx=np.linspace(0.5,25.8,2); yy=np.array([3.5,4.9])
X,Y=np.meshgrid(xx,yy); Z=np.zeros_like(X)-0.08
ax.plot_surface(X,Y,Z,color='gray',alpha=.38,edgecolor='gray')
# grid-like floor
for x in np.arange(0,27,2.4): ax.plot([x,x],[0,6.1],[0,0],':',color='gray',lw=.7,alpha=.7)
for y in [0,2.4,6.08]: ax.plot([0,26.5],[y,y],[0,0],':',color='gray',lw=.7,alpha=.7)
ax.scatter([],[],[],s=170,c='green',label='Fingerprints')
ax.legend(loc='upper left',fontsize=24,frameon=False)
ax.set_xlabel('X (m)',fontsize=26,fontweight='bold',labelpad=18)
ax.set_ylabel('Y (m)',fontsize=26,fontweight='bold',labelpad=20)
ax.set_zlim(-.4,1); ax.set_zticks([])
ax.set_xlim(-.5,27); ax.set_ylim(-.5,7.0)
ax.view_init(elev=30,azim=-130)
ax.set_box_aspect((4,1.1,.25))
ax.grid(True)
plt.tight_layout()
fig.savefig(OUT,dpi=300,bbox_inches='tight',pad_inches=0.02)
print(f"Saved {OUT}")
