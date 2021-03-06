"""
Modified from
https://www.shadertoy.com/view/MdlfDl
https://p5js.org/examples/hello-p5-flocking.html
"""

from render_util import *
from render_single import *

N_BOIDS = 40
MIN_DIST = 10
MIN_DIST2 = MIN_DIST ** 2
MIN_NEIGHBOR = 15
MIN_NEIGHBOR2 = MIN_NEIGHBOR ** 2
MAX_V = 25
MIN_V = 2
MAX_F = 0.5

fps = 120.

res_scale = 1.5

def boids(my_id, p_x, p_y, v_x, v_y, other_boids):
    
    acc = np.zeros(2)
    
    pc = np.zeros(2)
    avg_v = np.zeros(2)
    fc_cohesion = np.zeros(2)
    fc_repell = np.zeros(2)
    count_cohesion = 0.0
    count_repell = 0.0
    for i in range(N_BOIDS):
        not_me = (i != my_id)
        other_x = expand_1D(other_boids[i, 0])
        other_y = expand_1D(other_boids[i, 1])
        other_vx = expand_1D(other_boids[i, 2])
        other_vy = expand_1D(other_boids[i, 3])
        d2 = (p_x - other_x) ** 2 + (p_y - other_y) ** 2
        
        is_friend = (d2 < MIN_NEIGHBOR2)
        update_pc = is_friend * not_me
        
        pc = pc + select(update_pc, 1, 0) * np.array([other_x, other_y])
        avg_v = avg_v + select(update_pc, 1, 0) * np.array([other_vx, other_vy])
        count_cohesion = count_cohesion + select(update_pc, 1, 0)
        
        is_close = (d2 < 4 * MIN_DIST2)
        update_repell = is_close * not_me
        d = np.array([p_x - other_x, p_y - other_y])
        #fc_repell = fc_repell + d * select(update_repell, 1 / d2, 0)
        d_norm = max(sqrt(d2), 1e-4)
        fc_repell = fc_repell + 3 * d * select(update_repell, 1 / d_norm * (1 / d_norm - 1 / (2 * MIN_DIST)), 0)
        count_repell = count_repell + select(update_repell, 1, 0)
        
    update_cohesion = count_cohesion > 0.5
    
    count_cohesion = select(update_cohesion, count_cohesion, 1)
    pc = pc / count_cohesion
    fc_cohesion = pc - np.array([p_x, p_y])

    fc_cohesion_norm = length(fc_cohesion, 2)
    fc_cohesion = fc_cohesion / fc_cohesion_norm * MAX_V - np.array([v_x, v_y])
    update_cohesion = update_cohesion * (fc_cohesion_norm > 0)
    
    #fc_cohesion = normalize(fc_cohesion) * MAX_V - np.array([v_x, v_y])
    fc_cohesion_norm = length(fc_cohesion, 2)
    fc_cohesion = fc_cohesion * select(fc_cohesion_norm > MAX_F, MAX_F / fc_cohesion_norm, 1.0)
    fc_cohesion_x = select(update_cohesion, fc_cohesion[0], 0.0)
    fc_cohesion_y = select(update_cohesion, fc_cohesion[1], 0.0)
    fc_cohesion = np.array([fc_cohesion_x, fc_cohesion_y])
    
    orig_avg_v_norm = length(avg_v, 2)
    avg_v = avg_v / select(orig_avg_v_norm > 0, orig_avg_v_norm, 1.0)
    avg_v = avg_v * MAX_V - np.array([v_x, v_y])
    avg_v_norm = length(avg_v, 2)
    fc_alignment = avg_v * select(avg_v_norm > MAX_F, MAX_F / avg_v_norm, 1.0)
    fc_alignment_x = select(update_cohesion, fc_alignment[0], 0.0)
    fc_alignment_y = select(update_cohesion, fc_alignment[1], 0.0)
    fc_alignment = np.array([fc_alignment_x, fc_alignment_y])
    
    update_repell = count_repell > 0.5

    fc_repell_norm = length(fc_repell, 2)
    fc_repell = fc_repell / fc_repell_norm * MAX_V - np.array([v_x, v_y])
    update_repell = update_repell * (fc_repell_norm > 0)
    #fc_repell = normalize(fc_repell) * MAX_V - np.array([v_x, v_y])
    
    fc_repell_norm = length(fc_repell, 2)
    fc_repell = fc_repell * select(fc_repell_norm > MAX_F, MAX_F / fc_repell_norm, 1.0)
    fc_repell_x = select(update_repell, fc_repell[0], 0.0)
    fc_repell_y = select(update_repell, fc_repell[1], 0.0)
    fc_repell = np.array([fc_repell_x, fc_repell_y])
    
    fc_border = [0.0, 0.0]
    
    dist_left = p_x + 64.0 * res_scale
    hit_left = dist_left < MIN_DIST
    fc_border[0] = fc_border[0] + select(hit_left, (dist_left - MIN_DIST) ** 2., 0.0)
    
    dist_bottom = p_y + 64.0
    hit_bottom = dist_bottom < MIN_DIST
    fc_border[1] = fc_border[1] + select(hit_bottom, (dist_bottom - MIN_DIST) ** 2., 0.0)
    
    dist_right = 64.0 * res_scale - p_x
    hit_right = dist_right < MIN_DIST
    fc_border[0] = fc_border[0] - select(hit_right, (dist_right - MIN_DIST) ** 2., 0.0)
    
    dist_top = 64.0 - p_y
    hit_top = dist_top < MIN_DIST
    fc_border[1] = fc_border[1] - select(hit_top, (dist_top - MIN_DIST) ** 2, 0.0)
    
    fc_border = np.array(fc_border)
    old_fc_border_norm = length(fc_border, 2)
    fc_border = fc_border / old_fc_border_norm * MAX_V - np.array([v_x, v_y])
    new_fc_border_norm = length(fc_border, 2)
    fc_border = fc_border * select(new_fc_border_norm > MAX_F, MAX_F / new_fc_border_norm, 1.0)
    fc_border_x = select(old_fc_border_norm > 0, fc_border[0], 0.0)
    fc_border_y = select(old_fc_border_norm > 0, fc_border[1], 0.0)
    fc_border = np.array([fc_border_x, fc_border_y])
    
    fc_noise = rand2([my_id, count_cohesion]) * 2.0 - 1.0
    # supposed to be a small purturbation, so no need to reach MAX_F
    fc_noise = fc_noise * MAX_F / 3.0
        
    #acc = fc_cohesion + fc_repell * 2.0 + fc_alignment + fc_border * 10.0 + fc_noise
    # disable noise for now
    acc = fc_cohesion + fc_repell * 2.0 + fc_alignment + fc_border * 10.0
    acc_norm = length(acc, 2)
    acc = acc * select(acc_norm > MAX_F, MAX_F / acc_norm, 1.0)
    
    new_v = np.array([v_x, v_y]) + acc
    new_v_norm = length(new_v, 2)
    new_v = new_v * select(new_v_norm > MAX_V, MAX_V / new_v_norm, 1.0)
    
    new_pos = np.array([p_x, p_y]) + new_v / fps
    
    #new_pos[0] = new_pos[0] - select(new_pos[0] > 64. * res_scale, 128. * res_scale, 0.0)
    #new_pos[1] = new_pos[1] - select(new_pos[1] > 64., 128., 0.0)
    #new_pos[0] = new_pos[0] + select(new_pos[0] < -64. * res_scale, 128. * res_scale, 0.0)
    #new_pos[1] = new_pos[1] + select(new_pos[1] < -64., 128., 0.0)
        
    return output_color([new_pos[0], new_pos[1], new_v[0], new_v[1]])

shaders = [boids]
is_color = False


    
def main():
    
    if len(sys.argv) < 3:
        print('Usage: python render_[shader].py mode base_dir')
        raise
        
    mode = sys.argv[1]
    base_dir = sys.argv[2]
    
    camera_dir = os.path.join(base_dir, 'datasets/datas_boids')
    preprocess_dir = os.path.join(base_dir, 'preprocess/boids')
        
    if mode == 'generate_dataset':
        
        for mode in ['train', 'test', 'validate']:
            
            if mode == 'train':
                nframes = 1790000
            elif mode == 'validate':
                nframes = 85000
            else:
                nframes = 10000
        
            texture_maps = np.zeros([N_BOIDS, 4])
            texture_maps[:, :2] = 0.1 + 0.9 * np.random.rand(N_BOIDS, 2)
            texture_maps[:, :2] *= 128
            texture_maps[:, :2] -= 64
            texture_maps[:, 0] *= res_scale
            texture_maps[:, 2:] = np.random.rand(N_BOIDS, 2) * 0.5 - 0.25
            
            outdir = get_shader_dirname(os.path.join(preprocess_dir, mode), shaders[0], 'none', 'boids')

            render_single(os.path.join(preprocess_dir, mode), 'render_boids', 'boids', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_kw={'compute_f': False, 'ground_truth_samples': 1, 'zero_samples': True, 'gname': '%s_ground' % mode, 'temporal_texture_buffer': True, 'texture_maps': texture_maps, 'use_texture_maps': True, 'n_boids': N_BOIDS, 'log_getitem': False})
            
            os.rename(os.path.join(outdir, '%s_ground.npy' % mode),
                      os.path.join(camera_dir, '%s_ground.npy' %mode))
        
    return
    

if __name__ == '__main__':
    main()