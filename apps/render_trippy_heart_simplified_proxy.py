
"""
modified from shadertoy
https://www.shadertoy.com/view/MltBWM
"""

from render_util import *
from render_single import *

def length(vec):
    ans = 0.0
    for i in range(vec.shape[0]):
        ans = ans + vec[i] ** 2.0
    return ans ** 0.5

time_scale = 1.0

def trippy_heart_simplified_proxy(position, tex_coords, normal, light_dir, viewer_dir, use_triangle_wave, time=None):
    """
    Shader arguments:
     - position:   3 vector of x, y, z position on surface
     - tex_coords: 2 vector for s, t texture coords
     - normal:     3 vector of unit normal vector on surface
     - light_dir:  3 vector of unit direction towards light source
     - viewer_dir: 3 vector of unit direction towards camera
     - use_triangle_wave: if True then triangle_wave() should be used instead of fract() for shaders that perform tiling.
    """
    #new_p = (tex_coords / 10.0 + 1.0) * 10.0
    #new_p0 = (new_p[0] % 4.0 - 1.0) * 0.6
    #new_p1 = (new_p[1] % 4.0 - 1.0) * 0.6

    scale = 20.0

    new_p = tex_coords / scale
    #new_p[0] = new_p[0] / 2.0
    new_p = new_p + 2.0
    new_p0 = (new_p[0] % 4.0 - 2.0) * 0.6
    new_p1 = (new_p[1] % 4.0 - 2.0) * 0.6
    #new_p0 = (new_p[0] - 2.0) * 0.6
    #new_p1 = (new_p[1] - 2.0) * 0.6

    new_py = -0.1 - new_p1 * 1.2 + abs(new_p0) * (1.0 - abs(new_p0))
    #new_p[0] = new_p0
    #new_p[1] = new_py
    r = (new_p0 * new_p0 + new_py * new_py) ** 0.5
    d = 0.5
    mX = (sin(time * (0.5 * time_scale)) + 1.0) * 0.2
    mY = (cos(time * (0.5 * time_scale)) + 1.0) * 0.1

    hcol0 = (tex_coords[0] / scale + 1.0)
    hcol1 = (tex_coords[1] / scale + 1.0)
    hcol2 = mX

    for i in loop_generator(50):
        dot_hcol = hcol0 * hcol0 + hcol1 * hcol1 + hcol2 * hcol2
        hcol0 = 1.3 * abs(abs(hcol0) / dot_hcol - 1.0)
        new_hcol2 = 0.999 * abs(abs(hcol1) / dot_hcol - 1.0)
        hcol1 = 0.7 * abs(abs(hcol2) / dot_hcol - mY)
        hcol2 = new_hcol2

    dot_hcol = hcol0 * hcol0 + hcol1 * hcol1 + hcol2 * hcol2
    bcol0 = 1.3 * abs(abs(hcol0) / dot_hcol - 1.0) * 0.5
    bcol2 = 0.999 * abs(abs(hcol1) / dot_hcol - 1.0)
    bcol1 = 0.7 * abs(abs(hcol2) / dot_hcol - mY)

    hcol0 = hcol0 * 2.0

    col = mix(numpy.array([bcol0, bcol1, bcol2]), numpy.array([hcol0, hcol1, hcol2]), smoothstep(-0.15, 0.15, (d - r)))
    #col = numpy.array([smoothstep(-0.15, 0.15, d-r), smoothstep(-0.15, 0.15, d-r), smoothstep(-0.15, 0.15, d-r)])
    ans = output_color(col)
    for expr in normal.tolist():
        expr.log_intermediates_subset_rank = 1
    return ans

shaders = [trippy_heart_simplified_proxy]                 # This list should be declared so a calling script can programmatically locate the shaders
is_color = True                    # Set to True if the shader is 3 channel (color) and False if shader is greyscale
fov = 'small'

def main():
    dir = '/n/fs/shaderml/1x_1sample_trippy_heart_tile_rotation/raw_data'
    
    camera_pos = numpy.load('/n/fs/shaderml/datas_trippy_temporal/test_middle.npy')[:1]
    render_t = numpy.load('/n/fs/shaderml/datas_trippy_temporal/test_time.npy')[10:11] + 29 / 30
    nframes = 1
    render_single('out', 'render_trippy_heart_simplified_proxy', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'trippy_temporal_input', 'efficient_trace': True, 'chron_order': True, 'collect_loop_and_features': True, 'subsample_loops': 4})
    return
    
    if True:
        
        camera_pos = numpy.load(os.path.join(dir, 'train_pl.npy'))
        render_t = numpy.load(os.path.join(dir, 'train_time_pl.npy'))
        nframes = render_t.shape[0]
        render_single(os.path.join(dir, 'train'), 'render_trippy_heart_simplified_proxy', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 120), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small', 'efficient_trace': True, 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'SELECT_FEATURE_THRE': 400})
        #nframes = 10
        #camera_pos = camera_pos[:10, :]
        #render_t = render_t[:10]
        #render_single(os.path.join(dir, 'train'), 'render_trippy_heart_simplified_proxy', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small', 'efficient_trace': True, 'chron_order': True, 'collect_loop_and_features': True, 'subsample_loops': 4})
        #render_single(os.path.join(dir, 'train'), 'render_trippy_heart_simplified_proxy', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (40, 60), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small', 'efficient_trace': True, 'chron_order': True, 'collect_loop_and_features': True, 'subsample_loops': 4})
        #render_single(os.path.join(dir, 'train'), 'render_trippy_heart_simplified_proxy', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (40, 60), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small', 'efficient_trace': True, 'chron_order': True})
        return
   
    if False:
        #for mode in ['test_far']:
        for mode in ['train']:
            if mode == 'train':
                nframes = 800
                z_min = 40 / 1.5
                z_max = 165 * 1.5
            elif mode == 'test_close':
                nframes = 5
                z_max = 40 / 1.5
                z_min = 10
            elif mode == 'test_far':
                nframes = 5
                z_min = 165 * 1.5
                z_max = 400
            elif mode == 'test_middle':
                nframes = 20
                z_min = 40
                z_max = 165
            else:
                raise
            camera_pos = numpy.array([[0.0, 0.0, 50.0, numpy.pi, 0.0, 0.0]] * nframes)
            for i in range(nframes):
                camera_pos[i, 2] = numpy.random.uniform(z_min, z_max)
            numpy.save(os.path.join(dir, mode + '_pl.npy'), camera_pos)
            t_min = 0.0
            t_max = 4.0 * numpy.pi
            render_t = numpy.random.uniform(t_min, t_max, size=nframes)
            numpy.save(os.path.join(dir, mode + '_time_pl.npy'), render_t)
    return

    #for mode in ['test_far']:
    #for mode in ['train', 'test_close', 'test_far', 'test_middle']:
    #for mode in ['test_close', 'test_far', 'test_middle']:
    for mode in ['train']:
        if not os.path.isdir(os.path.join(dir, mode)):
            os.mkdir(os.path.join(dir, mode))
        camera_pos = numpy.load(os.path.join(dir, mode + '.npy'))
        render_t = numpy.load(os.path.join(dir, mode + '_time.npy'))
        nframes = render_t.shape[0]
        print(nframes)
        render_single('out', 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_small', 'is_tf': True})
        #render_single(os.path.join(dir, mode), 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (672, 896), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_ground', 'is_tf': True})
        #render_single(os.path.join(dir, mode), 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 120), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'small', 'efficient_trace': True})
    return
    for mode in ['train', 'test_close', 'test_far', 'test_middle']:
    #for mode in ['test_close', 'test_far', 'test_middle']:
        camera_pos = numpy.load(os.path.join(dir, mode) + '.npy').tolist()
        nframes = len(camera_pos)
        render_single('out', 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_small', 'is_tf': True})

if __name__ == '__main__':
    main()
