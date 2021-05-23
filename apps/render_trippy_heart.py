
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

def trippy_heart(position, tex_coords, normal, light_dir, viewer_dir, use_triangle_wave, time=None):
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

    for i in loop_generator(100):
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

shaders = [trippy_heart]                 # This list should be declared so a calling script can programmatically locate the shaders
is_color = True                    # Set to True if the shader is 3 channel (color) and False if shader is greyscale
fov = 'small'

def main():
    dir = '/localtmp/yuting/1x_1sample_trippy_heart'
    #render_t = numpy.linspace(0, 1, 30)
    #nframes = 30
    #camera_pos = numpy.array([[0.0, 0.0, 43.0, numpy.pi, 0.0, 0.0]] * nframes)
    #camera_pos_velocity = numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * nframes)
    #render_single(os.path.join('out', 'fixed_path'), 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'aliasing', 'is_tf': True, 'camera_pos_velocity': camera_pos_velocity, 't_sigma': 0, 'zero_samples': True})
    #render_single(os.path.join('out', 'fixed_path'), 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'noisy', 'is_tf': True, 'camera_pos_velocity': camera_pos_velocity, 't_sigma': 1/60.0})
    #render_single(os.path.join('out', 'fixed_path'), 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'ground_4x', 'is_tf': True, 'camera_pos_velocity': camera_pos_velocity, 't_sigma': 1/60.0})
    #render_single(os.path.join('out', 'fixed_path'), 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'ground_no_motion_blur_4x', 'is_tf': True})
    #return

    if False:
        for mode in ['test_far']:
            if mode == 'train':
                nframes = 200
                z_min = 40
                z_max = 165
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
            numpy.save(os.path.join(dir, mode + '.npy'), camera_pos)
            t_min = 0.0
            t_max = 4.0 * numpy.pi
            render_t = numpy.random.uniform(t_min, t_max, size=nframes)
            numpy.save(os.path.join(dir, mode + '_time.npy'), render_t)

    #for mode in ['test_far']:
    #for mode in ['train', 'test_close', 'test_far', 'test_middle']:
    for mode in ['train']:
        if not os.path.isdir(os.path.join(dir, mode)):
            os.mkdir(os.path.join(dir, mode))
        camera_pos = numpy.load(os.path.join(dir, mode + '.npy'))
        render_t = numpy.load(os.path.join(dir, mode + '_time.npy'))
        nframes = render_t.shape[0]
        print(nframes)
        #render_single(os.path.join(dir, mode), 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_small', 'is_tf': True})
        #render_single(os.path.join(dir, mode), 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_ground', 'is_tf': True})
        render_single(os.path.join(dir, mode), 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 120), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'small', 'efficient_trace': True, 'every_nth': 36, 'every_nth_stratified': True})
    return
    for mode in ['train', 'test_close', 'test_far', 'test_middle']:
    #for mode in ['test_close', 'test_far', 'test_middle']:
        camera_pos = numpy.load(os.path.join(dir, mode) + '.npy').tolist()
        nframes = len(camera_pos)
        render_single('out', 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_small', 'is_tf': True})

if __name__ == '__main__':
    main()
