"""Render RGB, semantic and instance views with identical sampled geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plyfile import PlyData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path)
    args = parser.parse_args()
    root = args.result
    clouds = [PlyData.read(root/name)['vertex'].data for name in
              ('map_labeled.ply','map_semantic.ply','map_instance.ply')]
    ix = np.arange(len(clouds[0]))
    if len(ix)>100000:
        ix = np.random.default_rng(42).choice(ix,100000,replace=False)
    xyz = np.column_stack([clouds[0][k][ix] for k in ('x','y','z')])
    xyz -= np.median(xyz,axis=0)
    _,axes = np.linalg.eigh(xyz.T@xyz)
    xyz = xyz@axes[:,::-1]
    fig,grid = plt.subplots(2,3,figsize=(15,9),facecolor='white')
    for col,(data,title) in enumerate(zip(clouds,('Original RGB geometry','Predicted semantics','Predicted instances'))):
        rgb = np.column_stack([data[k][ix] for k in ('red','green','blue')])/255
        for row,(u,v,d) in enumerate(((0,1,2),(0,2,1))):
            order = np.argsort(xyz[:,d])
            ax = grid[row,col]
            ax.scatter(xyz[order,u],xyz[order,v],c=rgb[order],s=.5,rasterized=True)
            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title(title if row==0 else 'Second orthographic view')
    result = json.loads((root/'result.json').read_text())
    fig.suptitle(f"{result['dataset']} / {result['sequence']} | "
                 f"{result['raw_frame_count']} frames | label coverage {result['label_coverage']:.1%}\n"
                 'Gray = unknown / ambiguous. Geometry and original RGB are unchanged.',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(root/'preview.png',dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
