
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
    
    if len(sys.argv) < 3:
        print('Usage: python render_[shader].py base_mode base_dir')
        raise
        
    base_mode = sys.argv[1]
    base_dir = sys.argv[2]
    
    camera_dir = os.path.join(base_dir, 'datasets/datas_bricks_with_bg')
    preprocess_dir = os.path.join(base_dir, 'preprocess/bricks')
    
    if not os.path.exists(camera_dir):
        os.makedirs(camera_dir, exist_ok=True)
    
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir, exist_ok=True)
    
    texture_maps = os.path.join(base_dir, 'datasets', 'bricks_texture.npy')
    
    if base_mode == 'collect_raw':
        
        camera_pos = numpy.load(os.path.join(camera_dir, 'train.npy'))
        render_t = numpy.load(os.path.join(camera_dir, 'train_time.npy'))
        nframes = render_t.shape[0]
        
        train_start = numpy.load(os.path.join(camera_dir, 'train_start.npy'))
        render_single(os.path.join(preprocess_dir, 'train'), 'render_bricks_normal_texture', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 80), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': 'train_small', 'tile_only': True, 'tile_start': train_start, 'texture_maps': texture_maps, 'use_texture_maps': True, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
        
    elif base_mode == 'generate_dataset':
        for mode in ['train', 'test_close', 'test_far', 'test_middle', 'validate']:
            camera_pos = numpy.load(os.path.join(camera_dir, mode + '.npy'))            
            nframes = camera_pos.shape[0]
            
            if mode in ['train', 'validate']:
                tile_start = numpy.load(os.path.join(camera_dir, mode + '_start.npy'))[:nframes]
                render_size = (320, 320)
                tile_only = True
                render_t = numpy.load(os.path.join(camera_dir, mode + '_time.npy'))
            else:
                tile_start = None
                render_size = (640, 960)
                tile_only = False
                render_t_pool = numpy.load(os.path.join(camera_dir, 'test_time.npy'))
                if mode == 'test_close':
                    render_t = render_t_pool[:5]
                elif mode == 'test_far':
                    render_t = render_t_pool[5:10]
                else:
                    render_t = render_t_pool[10:]
                    
            render_t = render_t[:nframes]
            
                    
            outdir = get_shader_dirname(os.path.join(preprocess_dir, mode), shaders[0], 'none', 'plane')
                
            render_single(os.path.join(preprocess_dir, mode), 'render_bricks_normal_texture', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = render_size, render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_ground' % mode, 'tile_only': tile_only, 'tile_start': tile_start, 'texture_maps': texture_maps, 'use_texture_maps': True, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
            
            if mode in ['train', 'validate']:
                target_dir = os.path.join(camera_dir, mode + '_img')
            else:
                target_dir = os.path.join(camera_dir, 'test_img')
                
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
                
            
            for file in os.listdir(outdir):
                if file.startswith('%s_ground' % mode) and file.endswith('.png'):
                    os.rename(os.path.join(outdir, file),
                              os.path.join(target_dir, file))
                    
    elif base_mode == 'sample_camera_pos':
        
        test_render_t = None
        t_range = 1
        
        for mode in ['train', 'test_close', 'test_far', 'test_middle', 'validate']:
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
                
            if mode in ['train', 'validate']:
                expand_boundary = 160
                render_t = np.random.rand(nframes) * t_range
                numpy.save(os.path.join(preprocess_dir, mode + '_time.npy'), render_t)
            else:
                expand_boundary = 0
                if test_render_t is None:
                    test_render_t = np.random.rand(30) * t_range
                    np.save(os.path.join(preprocess_dir, 'test_time.npy'), render_t)
                
                if mode == 'test_close':
                    render_t = test_render_t[:5]
                elif mode == 'test_far':
                    render_t = test_render_t[5:10]
                else:
                    render_t = test_render_t[10:]
                
            camera_pos = [None] * nframes
            render_single(os.path.join(preprocess_dir, mode), 'render_bricks_normal_texture', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'z_min': z_min, 'z_max': z_max, 'camera_pos': camera_pos, 'gname': '%s_noisy' % mode, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'expand_boundary': expand_boundary, 'texture_maps': texture_maps, 'use_texture_maps': True})
            
            numpy.save(os.path.join(preprocess_dir, mode + '.npy'), camera_pos)
    return

if __name__ == '__main__':
    main()
