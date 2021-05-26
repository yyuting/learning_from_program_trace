import sys
import os
import numpy
import numpy as np

nimages = 800
ntiles_h = 2
ntiles_w = 3

nbudget_fraction = 0.25
tile_size = 320

pyramid_file_prefix = 'tiles_var_pyramid'
saliency_file_prefix = 'tiles_saliency'

pyramid_scales = [6, 3, 0]
max_sample_rate = 4

# using saliency, pyramid scale 4, 2, 0 on mean response, rescale first 25% nonzero score to the range of 0 - 1, then sample probability in x
# each partition samples 300 tiles

def main():
    img_dir = sys.argv[1]
    cam_dir = sys.argv[2]
    
    mode = sys.argv[3]
    
    pyramid_file = os.path.join(img_dir, pyramid_file_prefix + '_0_2.npy' )
    if not os.path.exists(pyramid_file):
        pyramid_file = os.path.join(img_dir, pyramid_file_prefix + '.npy')
        
    pyramid_score = np.load(pyramid_file)
    
    saliency_file = os.path.join(img_dir, saliency_file_prefix + '.npy')
    saliency_score = np.load(saliency_file)
    
    all_chosen_indices = np.zeros(pyramid_score.shape[0]).astype('bool')
    invalid_indices = np.ones(pyramid_score.shape[0]).astype('bool')
    indices_vals = []
    
    min_score_pct = 0
    allow_random = True
    
    nbudget = int(nbudget_fraction * saliency_score.shape[0])
    

    scores = [saliency_score, pyramid_score[:, 8], pyramid_score[:, 4], pyramid_score[:, 0]]
    min_score_pct = 75
    allow_random = False
        
    if allow_random:
        sample_partition = nbudget // (len(scores) + 1)
    else:
        sample_partition = nbudget // len(scores)
        
    for current_score in scores:
        if not np.allclose(current_score, saliency_score):
            # record which indices are valid for this certain metric before drawing random samples
            invalid_indices[current_score > 0] = False
    
    # draw random partition first to include fewer less interesting tiles
    if allow_random:
        sample_prob = np.ones(pyramid_score.shape[0])
        sample_prob[invalid_indices] = 0
        sample_prob /= np.sum(sample_prob)

        sampled_ind = np.random.choice(np.arange(pyramid_score.shape[0]), sample_partition, replace=False, p=sample_prob)
        all_chosen_indices[sampled_ind] = True
        indices_vals = np.concatenate((indices_vals, sampled_ind)).astype('i')
            
        
    for current_score in scores:
        
        # mask out indices that are already chosen, so we can sample without replacement
        current_score[all_chosen_indices] = 0
        
        max_score = np.max(current_score)
        min_score = np.min(current_score[current_score > 0])
        
        nonzero_score = current_score[current_score > 0]
        min_score = np.percentile(nonzero_score, min_score_pct)
        
        # scale score to the range of 0 to 1
        scaled_score = (current_score - min_score) / (max_score - min_score)
        
        sample_prob = scaled_score
        
        if False:
            # linearly map patches with nonzero score to the sample rate of max_sample_rate to 1
            if max_score == min_score:
                sample_prob = np.ones(current_score.shape[0])
            else:
                sample_prob = (current_score - min_score) / (max_score - min_score) * max_sample_rate + 1
            
        
        
        # give 0 prob to 0 score
        sample_prob[current_score < min_score] = 0

        sample_prob[invalid_indices] = 0
        
        sample_prob /= np.sum(sample_prob)
        
        # make sure sample budget is smaller than or equal to the number of nonzero entries in sample_prob
        current_sample_budget = min(sample_partition, np.sum(sample_prob > 0))
        
        sampled_ind = np.random.choice(np.arange(current_score.shape[0]), current_sample_budget, replace=False, p=sample_prob)
        all_chosen_indices[sampled_ind] = True
        indices_vals = np.concatenate((indices_vals, sampled_ind)).astype('i')
        
    # fill the rest of the nbuget by randomly sample from untouched patches if they at least have a valid score in one of the pyramid scale
    random_sample_budget = nbudget - np.sum(all_chosen_indices)
    sample_prob = np.ones(pyramid_score.shape[0])
    sample_prob[invalid_indices] = 0
    sample_prob[all_chosen_indices] = 0
    sample_prob /= np.sum(sample_prob)
    
    sampled_ind = np.random.choice(np.arange(pyramid_score.shape[0]), random_sample_budget, replace=False, p=sample_prob)
    all_chosen_indices[sampled_ind] = True
    indices_vals = np.concatenate((indices_vals, sampled_ind)).astype('i')
    
    assert np.sum(all_chosen_indices) == nbudget
    
    #indices_vals = np.arange(all_chosen_indices.shape[0])[all_chosen_indices]
        
    numpy.save(os.path.join(img_dir, 'pyramid_sampled_idx_mode.npy'), indices_vals)
    
    raw_camera_pos = numpy.load(os.path.join(cam_dir, '%s.npy' % mode))
    raw_render_t = numpy.load(os.path.join(cam_dir, '%s_time.npy' % mode))
    
    camera_pos = numpy.empty([nbudget, 6])
    render_t = numpy.empty(nbudget)
    tile_start = numpy.empty([nbudget, 2])
    
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
    
    for i in range(indices_vals.shape[0]):
        idx = indices_vals[i]
        img_idx = idx // (ntiles_h * ntiles_w)
        h_idx = (idx - img_idx * (ntiles_h * ntiles_w)) // ntiles_w
        w_idx = idx - img_idx * (ntiles_h * ntiles_w) - h_idx * ntiles_w
        imgfile = 'tile_%05d_%d_%d.png' % (img_idx, h_idx, w_idx)
        
        camera_pos[i, :] = raw_camera_pos[img_idx, :]
        render_t[i] = raw_render_t[img_idx]
        tile_start[i, :] = np.array([h_idx, w_idx]) * tile_size
        
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
        
    html_str += """
<div class="clearfix"></div>
</body>
</html>
"""
    
    open(os.path.join(img_dir, 'pyramid_sampled_%s.html' % mode), 'w').write(html_str)
    numpy.save(os.path.join(img_dir, 'pyramid_%s.npy' % mode), camera_pos)
    numpy.save(os.path.join(img_dir, 'pyramid_%s_time.npy' % (mode)), render_t)
    numpy.save(os.path.join(img_dir, 'pyramid_%s_start.npy' % (mode)), tile_start)
    
if __name__ == '__main__':
    main()
    