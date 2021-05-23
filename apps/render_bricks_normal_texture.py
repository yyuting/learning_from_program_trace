
"""
A color (3 channel) shader which renders staggered brick

Re-implementation of shader described in Dorn et al. paper
"""

from render_util import *
from render_single import render_single

approx_mode = 'gaussian'
#approx_mode = 'regular'
#approx_mode = 'mc'

texture_xdim = 4096
texture_ydim = 4096

def bricks_normal_texture(position, tex_coords, normal, light_dir, viewer_dir, use_triangle_wave, time):
    """
    Shader arguments:
     - position:   3 vector of x, y, z position on surface
     - tex_coords: 2 vector for s, t texture coords
     - normal:     3 vector of unit normal vector on surface
     - light_dir:  3 vector of unit direction towards light source
     - viewer_dir: 3 vector of unit direction towards camera
     - use_triangle_wave: if True then triangle_wave() should be used instead of fract() for shaders that perform tiling.
    """

    #Sources
    #https://www.shadertoy.com/view/Mll3z4
    #Dorn et al. 2015
    
    # normal map texture
    # https://www.deviantart.com/zabacar/art/Seamless-Brick-Rock-Wall-Normal-Map-545390463

    #####
    # Parameter Settings
    #####
    
    

    #set discrete brick color, vs. just applying mortar grid over noise distribution
    #discrete is more similar to original Dorn implementation.
    #NOTE: currently having trouble setting Var to be a boolean, using 1.0 -> True
    discrete_bricks = Var('discrete_bricks', 0.0)

    #lighting
    specular_pow = 0.2
    specular_scale = 0.2

    #brick dimensions
    brick_height = 5.0
    brick_width = 15.0
    mortar_width = 1.0
    
    mortar_offset_x = -0.28
    mortar_offset_y = -0.28

    #colors
    brick_color_light = vec_color('brick_color_light', [0.98, 0.42, 0.24]) #http://rgb.to/keyword/4261/1/brick
    brick_color_dark = vec_color('brick_color_dark', [0.26, 0.06, 0.05]) #Dorn et al. 2015
    mortar_color =  vec_color('mortar_color', [0.7, 0.7, 0.7]) #Dorn et al. 2015

    #gabor noise parameters
    K = 1.0
    a = 0.05
    F_0 = 0.05
    omega_0 = 0.0
    impulses = 64.0
    period = 256
    
    # making the over simplified assumption that bricks is always on a plane
    # so that we don't need to compute the transformation from local to world coordinate
    assert normal.shape[0] == 3
    for i in range(3):
        if i < 2:
            assert normal[i].initial_value.value == 0.0
        else:
            assert normal[i].initial_value.value == 1.0
            
    if False:
        normal = [bilinear_texture_map(0, sign(tex_coords[1]) * tex_coords[1] * texture_ydim / (12 * brick_height), sign(tex_coords[0]) * tex_coords[0] * texture_xdim / (4 * brick_width), texture_xdim, texture_ydim),
                  bilinear_texture_map(1, sign(tex_coords[1]) * tex_coords[1] * texture_ydim / (12 * brick_height), sign(tex_coords[0]) * tex_coords[0] * texture_xdim / (4 * brick_width), texture_xdim, texture_ydim),
                  bilinear_texture_map(2, sign(tex_coords[1]) * tex_coords[1] * texture_ydim / (12 * brick_height), sign(tex_coords[0]) * tex_coords[0] * texture_xdim / (4 * brick_width), texture_xdim, texture_ydim)]
    
    normal = [bilinear_texture_map(0, tex_coords[1] * texture_ydim / (12 * brick_height), tex_coords[0] * texture_xdim / (4 * brick_width), texture_xdim, texture_ydim),
              bilinear_texture_map(1, tex_coords[1] * texture_ydim / (12 * brick_height), tex_coords[0] * texture_xdim / (4 * brick_width), texture_xdim, texture_ydim),
              bilinear_texture_map(2, tex_coords[1] * texture_ydim / (12 * brick_height), tex_coords[0] * texture_xdim / (4 * brick_width), texture_xdim, texture_ydim)]
    normal = np.array(normal)
    
    #normal[0] *= 4
    #normal[1] *= 4
        
    normal -= 0.5
    normal *= 2
    
    normal_norm = (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) ** 0.5
    normal /= normal_norm
    
    

    x = tex_coords[0] + mortar_offset_x
    y = tex_coords[1] + mortar_offset_y

    #used to identify mortar location
    x_mod = x % brick_width
    x_abs = x_mod * sign(x_mod)
    y_mod = y % brick_height
    y_abs = y_mod * sign(y_mod)


    #alternate staggered brick rows
    double_brick_height = brick_height * 2.0
    row_indicator = 0.5 + 0.5 * sign(brick_height - y % double_brick_height)
        
    staggered_x_abs = (x_abs + (brick_width / 2.0)) % brick_width

    #optional use of discretized bricks to set noise parameter.
    indicator = Var('indicator', 1.0 == row_indicator)
    x_discrete = select(indicator, x - x_abs, x - staggered_x_abs)
    y_discrete = y - y_abs
    
    darkness = 0.7 * rand([x_discrete, y_discrete]) + 0.3

    x_noise = x #select(1.0 == discrete_bricks, x_discrete, x)
    y_noise = y #select(1.0 == discrete_bricks, y_discrete, y)

    noise = Var('noise', 0.5 + 0.5 * simplex_noise(x_noise, y_noise))


    #compute pixel color
    brick_color = brick_color_light * noise + brick_color_dark * (1.0 - noise)
    brick_color = brick_color * darkness
    #horizontal mortar lines
    
    cond_y = abs(y_abs - brick_height) >= (mortar_width / 2.0)
    vertical_brick_boundary = Var('vertical_brick_boundary', select(indicator, x_abs, staggered_x_abs))
    cond_x = abs(vertical_brick_boundary - brick_width) >= (mortar_width / 2.0)
    
    cond = cond_y * cond_x
    brick_color0 = select(cond, brick_color[0], mortar_color[0])
    brick_color1 = select(cond, brick_color[1], mortar_color[1])
    brick_color2 = select(cond, brick_color[2], mortar_color[2])
    
    if False:
        brick_color0 = select(cond_y, mortar_color[0], brick_color[0])
        brick_color1 = select(cond_y, mortar_color[1], brick_color[1])
        brick_color2 = select(cond_y, mortar_color[2], brick_color[2])



        #vertical mortar lines (staggered)

        #brick_color = select(mortar_width >= vertical_brick_boundary, mortar_color, brick_color)
        brick_color0 = select(mortar_width >= vertical_brick_boundary, mortar_color[0], brick_color0)
        brick_color1 = select(mortar_width >= vertical_brick_boundary, mortar_color[1], brick_color1)
        brick_color2 = select(mortar_width >= vertical_brick_boundary, mortar_color[2], brick_color2)
    
    brick_color = numpy.array([brick_color0, brick_color1, brick_color2])
    
    #old_log_intermediates_subset_rank = compiler.log_intermediates_subset_rank
    #compiler.log_intermediates_subset_rank = 1
    #lighting computations
    LN = Var('LN', max(dot(light_dir, normal), 0.0))    # Diffuse term
    R = vec('R', 2.0 * LN * normal - light_dir)
    specular_intensity = Var('specular_intensity', (LN > 0) * max(dot(R, viewer_dir), 0.0) ** specular_pow) * specular_scale
    
    
    
    diffuse = brick_color * LN

    #get output color
    ans = output_color(diffuse + specular_intensity)


    # color rank changed in geometry_wrapper, depth recorded in demo.py
    for expr in normal.tolist() + [specular_intensity] + diffuse.tolist():
        expr.log_intermediates_subset_rank = 1

    return ans

shaders = [bricks_normal_texture]                 # This list should be declared so a calling script can programmatically locate the shaders
log_intermediates = False
is_color = True
normal_map = True

def main():
    
    camdir = '/n/fs/shaderml/datas_bricks_dsf'
    
    dir = '/n/fs/shaderml/1x_1sample_bricks'
    
    texture_maps = np.load('bricks_normal_map3.npy')
    
    
    if True:
    
        camera_pos = np.load(os.path.join(camdir, 'test_middle.npy'))
        render_t = np.load(os.path.join(camdir, 'test_time.npy'))[10:]
        nframes = camera_pos.shape[0]

        render_single(os.path.join(dir, 'train'), 'render_bricks_normal_texture', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': True, 'gname': 'train_small', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'texture_maps': texture_maps, 'use_texture_maps': True})

        return



        camera_pos = np.load(os.path.join(camdir, 'train.npy'))
        render_t = np.load(os.path.join(camdir, 'train_time.npy'))
        train_start = np.load(os.path.join(camdir, 'train_start.npy'))

        nframes = camera_pos.shape[0]

        render_single(os.path.join(dir, 'train'), 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (320, 320), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_noisy_expanded', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'tile_only': True, 'tile_start': train_start, 'collect_feature_mean_only': True, 'feature_normalize_dir': camdir, 'reference_dir': os.path.join(camdir, 'train_img')})

        return

        #camera_pos = numpy.load('/n/fs/shaderml/datas_bricks_staggered_tiles/train.npy')

        camera_pos = numpy.load(os.path.join(dir, 'train.npy'))
        nframes = camera_pos.shape[0]
        render_t = numpy.zeros(nframes)
        nframes = render_t.shape[0]

        render_single(os.path.join(dir, 'train'), 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_noisy_expanded', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'expand_boundary': 160})
        return
    
    # generate and store random camera data
    if True:
        #for mode in ['train', 'test_close', 'test_far', 'test_middle']:
        for mode in ['validate']:
            if mode == 'train':
                nframes = 800
                z_min = 7
                z_max = 90
            elif mode == 'validate':
                nframes = 80
                z_min = 7
                z_max = 90
            elif mode == 'test_close':
                nframes = 5
                z_min = 3
                z_max = 7
            elif mode == 'test_far':
                nframes = 5
                z_min = 90
                z_max = 180
            else:
                nframes = 20
                z_min = 7
                z_max = 90
            camera_pos = [None] * nframes
            render_single(os.path.join(dir, mode), 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (40, 60), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'z_min': z_min, 'z_max': z_max, 'camera_pos': camera_pos})
            numpy.save(os.path.join(dir, mode), camera_pos)
            
            render_t = np.zeros(nframes)
            numpy.save(os.path.join(dir, mode) + '_time.npy', render_t)
            
            render_single(os.path.join(dir, mode), 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'gname': '%s_noisy_expanded' % mode, 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'expand_boundary': 160})
        return
    
    if False:
        camera_pos = numpy.load(os.path.join(dir, 'test_far.npy'))
        render_t = numpy.zeros(5)
        nframes = camera_pos.shape[0]
        render_single(dir, 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'gname': 'train_small', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True})
        return
    
    if False:
        camera_pos = numpy.load(os.path.join(dir, 'train.npy'))

        camera_pos_periodic_range = numpy.array([20, 20, 0, 2 * np.pi, 2 * np.pi, 2 * np.pi])
        random_half_idx = numpy.random.choice(camera_pos.shape[0], size=camera_pos.shape[0] // 2, replace=False)
        camera_pos -= camera_pos_periodic_range
        camera_pos[random_half_idx] += 2 * camera_pos_periodic_range

        render_t = numpy.zeros(800)
        #render_t = numpy.load('/n/fs/shaderml/datas_bricks_staggered_tiles/train_time.npy')
        #tile_start = numpy.load('/n/fs/shaderml/datas_bricks_staggered_tiles/train_start.npy')
        nframes = camera_pos.shape[0]
        render_single(dir, 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 120), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'gname': 'train_small', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True})
        return
    
    if False:
        camera_pos = numpy.load('/n/fs/shaderml/datas_bricks_dsf/train.npy')
        nframes = camera_pos.shape[0]
        render_t = numpy.zeros(nframes)
        tile_start = numpy.load('/n/fs/shaderml/datas_bricks_dsf/train_start.npy')

        nsamples = 100
        sampled_camera_pos = np.tile(camera_pos, (nsamples, 1))
        camera_sigma = np.array([0.3, 0.3, 1, 0.1, 0.1, 0.1])
        tile_start = np.tile(tile_start, (nsamples, 1))
        for n in range(1, nsamples):
            sampled_camera_pos[nframes*n:nframes*(n+1), :] += np.random.randn(nframes, 6) * camera_sigma
        numpy.save(os.path.join('/n/fs/shaderml/1x_1sample_bricks', 'camera_sampled_train_v2.npy'), sampled_camera_pos)
        numpy.save(os.path.join('/n/fs/shaderml/1x_1sample_bricks', 'camera_sampled_tile_start.npy'), tile_start)
    
    if False:
        camera_pos = numpy.load(os.path.join(dir, 'train.npy'))
        nframes = camera_pos.shape[0]
        nsamples = 100
        sampled_camera_pos = np.tile(camera_pos, (nsamples, 1))
        camera_sigma = np.array([0.3, 0.3, 1, 0.1, 0.1, 0.1])
        for n in range(1, nsamples):
            sampled_camera_pos[nframes*n:nframes*(n+1), :] += np.random.randn(nframes, 6) * camera_sigma
        numpy.save(os.path.join('/n/fs/shaderml/1x_1sample_bricks', 'camera_sampled_full_res.npy'), sampled_camera_pos)
        
    if False:
        camera_pos = numpy.load('/n/fs/shaderml/1x_1sample_bricks_loss_proxy/validate.npy')
        nframes = camera_pos.shape[0]
        nsamples = 10
        sampled_camera_pos = np.tile(camera_pos, (nsamples, 1))
        camera_sigma = np.array([0.3, 0.3, 1, 0.1, 0.1, 0.1])
        for n in range(1, nsamples):
            sampled_camera_pos[nframes*n:nframes*(n+1), :] += np.random.randn(nframes, 6) * camera_sigma
        numpy.save(os.path.join('/n/fs/shaderml/1x_1sample_bricks_loss_proxy', 'camera_sampled_full_res.npy'), sampled_camera_pos)
    
    camera_pos0 = numpy.load('/n/fs/shaderml/datas_bricks_dsf/test_close.npy')
    camera_pos1 = numpy.load('/n/fs/shaderml/datas_bricks_dsf/test_far.npy')
    camera_pos2 = numpy.load('/n/fs/shaderml/datas_bricks_dsf/test_middle.npy')
    camera_pos = numpy.concatenate((camera_pos0, camera_pos1, camera_pos2), 0)
    nframes = camera_pos.shape[0]
    nsamples = 10
    sampled_camera_pos = np.tile(camera_pos, (nsamples, 1))
    camera_sigma = np.array([0.3, 0.3, 1, 0.1, 0.1, 0.1])
    for n in range(1, nsamples):
        sampled_camera_pos[nframes*n:nframes*(n+1), :] += np.random.randn(nframes, 6) * camera_sigma
    numpy.save(os.path.join('/n/fs/shaderml/datas_bricks_dsf', 'camera_sampled_test_full_res.npy'), sampled_camera_pos)
    
    return
    #render_single(dir, 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (320, 320), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 100, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'gname': 'train_smooth_camera_ground', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'camera_sigma': '0.05,0.05,0.1,0.005,0.005,0.005', 'tile_only': True, 'tile_start': tile_start, 'batch_size': 5})
    camera_pos = camera_pos[396:401]
    nframes = camera_pos.shape[0]
    render_t = render_t[396:401]
    tile_start = tile_start[396:401]
    #render_single(dir, 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (320, 320), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 100, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'gname': 'train_smooth_camera_ground', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'camera_sigma': '0,0,0,0.01,0.01,0.01', 'tile_only': True, 'tile_start': tile_start, 'batch_size': 5})
    render_single(dir, 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (320, 320), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'gname': 'train_small', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'tile_only': True, 'tile_start': tile_start})
    return

    camera_pos = numpy.array([[0, 0, 100, 0, np.pi, 0]] * 30)
    for i in range(30):
        camera_pos[i, 2] = 10 + 5 * i
    nframes = camera_pos.shape[0]
    render_t = numpy.zeros(nframes)
    render_single('out/top_view', 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small'})
    #render_single(os.path.join(dir, 'train'), 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size=(40, 60), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small', 'collect_loop_and_features': True})
    #render_single('out/higher_res', 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 640), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': True, 'gname': 'train_small', 'tile_only': True, 'tile_start': tile_start, 'rescale_tile_start': 0.5})
    #for mode in ['test_close', 'test_middle', 'test_far']:
    #    camera_pos = numpy.load(os.path.join(dir, mode + '.npy'))
    #    render_t = numpy.zeros(camera_pos.shape[0])
    #    nframes = camera_pos.shape[0]
    #    render_single('out', 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': mode+'_ground'})
    return
    

    if True:
        for mode in ['train', 'test_close', 'test_far', 'test_middle']:
        #for mode in ['test_close', 'test_far', 'test_middle']:
        #for mode in ['test_close']:
        #for mode in ['train']:
            camera_pos = numpy.load(os.path.join(dir, mode) + '.npy').tolist()
            nframes = len(camera_pos)
            render_single(os.path.join('out', mode), 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_small', 'is_tf': True})
            #render_single(os.path.join('out', mode), 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size=(640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_ground2', 'is_tf': False})

    #camera_pos = [None] * 25
    #render_single('out', 'render_bricks', 'sphere', 'none', sys.argv[1:], nframes=len(camera_pos), log_intermediates=False, render_size=(640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'x_min': -1000.0, 'x_max': 1000.0, 'y_min': -1000.0, 'y_max': 1000.0, 'z_min': -1000.0, 'z_max': 1000.0, 'z_log_range': False, 'camera_pos': camera_pos, 'count_edge': True})
    #numpy.save('camera_pos_sphere.npy', numpy.array(camera_pos))
        #render_single('out', 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=1, time_error=time_error, log_intermediates=True)
#    render_plane_shaders(shaders, default_render_size, use_triangle_wave=False, log_intermediates=log_intermediates, is_color=is_color, normal_map='ripples')

if __name__ == '__main__':
    main()
