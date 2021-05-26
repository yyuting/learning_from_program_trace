import os
import numpy
import numpy as np
import sys

ntiles_h = 2
ntiles_w = 3
tile_size = 320

bg_ratio = 0.05

def main():
    orig_dataset = sys.argv[1]
    camera_pos_pool_dir = sys.argv[2]
    mode = sys.argv[3]
    
    col_var = np.load(os.path.join(orig_dataset, 'tiles_var_col.npy'))
    
    orig_camera_pos = np.load(os.path.join(orig_dataset, 'pyramid_%s.npy' % mode))
    orig_render_t = np.load(os.path.join(orig_dataset, 'pyramid_%s_time.npy' % mode))
    orig_tile_start = np.load(os.path.join(orig_dataset, 'pyramid_%s_start.npy' % mode))
    raw_idx = np.load(os.path.join(orig_dataset, 'pyramid_sampled_idx_mode.npy'))
    
    camera_pos_pool = np.load(os.path.join(camera_pos_pool_dir, '%s.npy' % mode))
    render_t_pool = np.load(os.path.join(camera_pos_pool_dir, '%s_time.npy' % mode))
    
    assert orig_camera_pos.shape[0] == orig_render_t.shape[0] == orig_tile_start.shape[0]
    assert col_var.shape[0] == orig_camera_pos.shape[0] * 4
    assert camera_pos_pool.shape[0] == render_t_pool.shape[0] == col_var.shape[0] // 6
    
    orig_idx = np.arange(orig_camera_pos.shape[0])
    assert orig_idx.shape[0] % 4 == 0
    
    valid_bg_idx = np.where(col_var == 0)[0]
    
    n_bg_tiles = int((min(orig_idx.shape[0] * bg_ratio, valid_bg_idx.shape[0]) // 4) * 4)
    
    new_idx = []
    
    # uniformly sample 1 - bg_ratio of tiles in each quater of the original dataset
    quater_size = orig_idx.shape[0] // 4
    for i in range(4):
        sampled_idx = np.random.choice(np.arange(quater_size) + i * quater_size, int(quater_size - n_bg_tiles // 4), replace=False)
        new_idx.append(sorted(sampled_idx))
        
    new_idx = np.concatenate(new_idx)
    new_camera_pos = orig_camera_pos[new_idx]
    new_render_t = orig_render_t[new_idx]
    new_tile_start = orig_tile_start[new_idx]
    
    html_str = """
<html>
<head>
<style>
div.gallery {
border: 1px solid #ccc;
}

div.gallery:hover {
border: 1px solid #777;
}

div.gallery img {
width: 100%;
height: auto;
}

div.desc {
padding: 15px;
text-align: center;
}

* {
box-sizing: border-box;
}

.responsive {
padding: 0 6px;
float: left;
width: 16.666666%;
}

@media only screen and (max-width: 700px) {
.responsive {
width: 49.99999%;
margin: 6px 0;
}
}

@media only screen and (max-width: 500px) {
.responsive {
width: 100%;
}
}

.clearfix:after {
content: "";
display: table;
clear: both;
}
</style>
</head>
<body>
"""
    
    for i in range(new_idx.shape[0]):
        idx = raw_idx[new_idx[i]]
        img_idx = idx // (ntiles_h * ntiles_w)
        h_idx = (idx - img_idx * (ntiles_h * ntiles_w)) // ntiles_w
        w_idx = idx - img_idx * (ntiles_h * ntiles_w) - h_idx * ntiles_w
        imgfile = 'tile_%05d_%d_%d.png' % (img_idx, h_idx, w_idx)
        
        html_str += """
<div class="responsive">
<div class="gallery">
<a target="_blank">
<img src="{imgfile}">
</a>
<div class="desc">{i}, {imgfile}</div>
</div>
</div>
""".format(**locals())
        
    if n_bg_tiles > 0:
        sampled_bg_idx = np.random.choice(valid_bg_idx, n_bg_tiles, replace=False)

        bg_camera_pos = np.empty([n_bg_tiles, 6])
        bg_render_t = np.empty(n_bg_tiles)
        bg_tile_start = np.empty([n_bg_tiles, 2])
    
        for i in range(n_bg_tiles):
            idx = sampled_bg_idx[i]
            img_idx = idx // (ntiles_h * ntiles_w)
            h_idx = (idx - img_idx * (ntiles_h * ntiles_w)) // ntiles_w
            w_idx = idx - img_idx * (ntiles_h * ntiles_w) - h_idx * ntiles_w

            bg_camera_pos[i, :] = camera_pos_pool[img_idx, :]
            bg_render_t[i] = render_t_pool[img_idx]
            bg_tile_start[i, :] = np.array([h_idx, w_idx]) * tile_size

            imgfile = 'tile_%05d_%d_%d.png' % (img_idx, h_idx, w_idx)
            count = i + new_idx.shape[0]

            html_str += """
    <div class="responsive">
    <div class="gallery">
    <a target="_blank">
    <img src="{imgfile}">
    </a>
    <div class="desc">{count}, {imgfile}</div>
    </div>
    </div>
    """.format(**locals())

        html_str += """
    <div class="clearfix"></div>
    </body>
    </html>
"""
    
        new_camera_pos = np.concatenate((new_camera_pos, bg_camera_pos), 0)
        new_render_t = np.concatenate((new_render_t, bg_render_t), 0)
        new_tile_start = np.concatenate((new_tile_start, bg_tile_start), 0)
    
    np.save(os.path.join(orig_dataset, '%s.npy' % mode), new_camera_pos)
    np.save(os.path.join(orig_dataset, '%s_time.npy' % mode), new_render_t)
    np.save(os.path.join(orig_dataset, '%s_start.npy' % mode), new_tile_start)
    np.save(os.path.join(orig_dataset, '%s_orig_idx.npy' % mode), new_idx)
    
    open(os.path.join(orig_dataset, '%s.html' % (mode)), 'w').write(html_str)
    
if __name__ == '__main__':
    main()