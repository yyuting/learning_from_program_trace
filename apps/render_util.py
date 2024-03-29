
"""
Utility functions for rendering.
"""

import math
import pprint
import sys; sys.path += ['../compiler']
import time
import hashlib
import os, os.path
import shutil
import copy
import multiprocessing
from compiler import *
import compiler
import numpy.random
import numpy
import numpy as np
import numpy.linalg
import skimage
import skimage.io
import skimage.feature
from analyze_loop_statistic import *

from tempfile import NamedTemporaryFile

import subprocess

def system_parallel(cmdL, nproc=None, verbose=True):
    """
    Run a list of commands (each a string) via the shell using GNU parallel with nproc processes, return all outputs in a single str instance.
    """
    if nproc is None:
        nproc = multiprocessing.cpu_count()
    sh_filename = '_run_parallel_' + hashlib.md5('\n'.join(cmdL).encode('utf-8')).hexdigest()
    with open(sh_filename, 'wt') as f:
        f.write('\n'.join(cmdL))
    out = subprocess.check_output('parallel -j%d %s--keep-order < %s' % (nproc, '--verbose ' if verbose else '', sh_filename), shell=True)
    out = out.decode('utf-8')
    if verbose:
        print('-'*80)
        print('system_parallel output:')
        print('-'*80)
        print(out)
    os.remove(sh_filename)
    return out

default_render_size = (320, 480)
default_is_color = True
use_quadric_wrapper = False
use_scale_out_normal = True

apply_geometry_optimization = True

geometry_transfer = False

log_prefix = '_log_'

def get_filenames(id):
    return id + 'camera_pos.npy', id + 'render_t.npy', id + 'render_index.npy', id + COMPILER_PROBLEM_PY, id + 'camera_pos_velocity.npy', id + 'tile_start.npy', id + 'texture_maps.npy'

def unique_id():
    return hashlib.md5(str(time.time()).encode('utf-8')).hexdigest()

def vec(prefix, point):
    """
    Declare several Var() instances and return them as a numpy array.

     - prefix: string prefix for the variable names.
     - point:  array-like object storing the point (e.g. containing floats or Exprs), up to 4D
    """
    ans = numpy.zeros(len(point), dtype='object')
    for i in range(len(point)):
        ans[i] = Var(prefix + '_' + 'xyzw'[i], point[i])
    return ans

def set_channels(v):
    assert len(v) in [3, 4]
    v[0].channel = 'r'
    v[1].channel = 'g'
    v[2].channel = 'b'
    if len(v) == 4:
        v[3].channel = 'w'

def vec_color(prefix, point):
    assert len(point) == 3
    ans = vec(prefix, point)
    set_channels(ans)
    return ans


def vec_long(prefix, point):
    """
    Declare several Var() instances and return them as a numpy array.

     - prefix: string prefix for the variable names.
     - point:  array-like object storing the point (e.g. containing floats or Exprs), up to 4D
    """
    ans = numpy.zeros(len(point), dtype='object')
    for i in range(len(point)):
        ans[i] = Var(prefix + '_' + str(i), point[i])
    return ans

def bilinear_texture_map(ind, xx, yy, xdim, ydim, uv=False):
    
    if not uv:
        xx_scaled = xx % xdim
        yy_scaled = yy % ydim
    else:
        xx_scaled = xx * xdim
        yy_scaled = yy * ydim

    xx_down = floor(xx_scaled)
    xx_up = xx_down + 1
    yy_down = floor(yy_scaled)
    yy_up = yy_down + 1

    rx = xx_up - xx_scaled
    ry = yy_up - yy_scaled

    if xdim > 1:
        xx_up_mod = xx_up % xdim
    else:
        xx_up_mod = xx_up
    if ydim > 1:
        yy_up_mod = yy_up % ydim
    else:
        yy_up_mod = yy_up
    
    if isinstance(ind, int):
        inds = [ind]
        return_scalar = True
    else:
        assert isinstance(ind, (list, numpy.ndarray))
        inds = ind
        return_scalar = False
        
    ans = []
    
    for i in inds:
        val_dd = TextureMapScaled(i, xx_down, yy_down)
        val_du = TextureMapScaled(i, xx_down, yy_up_mod)
        val_ud = TextureMapScaled(i, xx_up_mod, yy_down)
        val_uu = TextureMapScaled(i, xx_up_mod, yy_up_mod)

        ans.append(rx * ry * val_dd + rx * (1.0 - ry) * val_du + (1.0 - rx) * ry * val_ud + (1.0 - rx) * (1.0 - ry) * val_uu)
    
    if return_scalar:
        ans = ans[0]
    else:
        ans = np.array(ans)
    return ans
bilinear_texture_map = def_generator(bilinear_texture_map)

def texcube_functor(texture_xdim, texture_ydim):
    def texcube(texture_id, p, n):
        x = bilinear_texture_map(texture_id, p[1] * texture_xdim, p[2] * texture_ydim, texture_xdim, texture_ydim)
        y = bilinear_texture_map(texture_id, p[2] * texture_xdim, p[0] * texture_ydim, texture_xdim, texture_ydim)
        z = bilinear_texture_map(texture_id, p[0] * texture_xdim, p[1] * texture_ydim, texture_xdim, texture_ydim)
        return x * abs(n[0]) + y * abs(n[1]) + z * abs(n[2])
    return def_generator(texcube)

def normalize(point):
    """
    Normalize a given numpy array of constants or Exprs, returning a new array with the resulting normalized Exprs.
    """
    prefix = unique_id()
    if isinstance(point[0], Var):
        prefix = point[0].name
        if prefix.endswith('_x'):
            prefix = prefix[:-2]
        prefix = prefix + '_normalized'
    var_norm = Var(prefix + '_norm', sqrt(sum(x**2 for x in point)))
    return vec(prefix, point / var_norm)

def normalize_const(v):
    """
    Normalize a numpy array of floats or doubles.
    """
    return v / numpy.linalg.norm(v)

def smoothstep(edge0, edge1, x):
    """
    re-implementation of opengl smoothstep
    https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/smoothstep.xhtml
    genType t;  /* Or genDType t; */
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
    """
    t0 = (x - edge0) / (edge1 - edge0)
    t = max(min(t0, 1.0), 0.0)
    return t * t * (3.0 - 2.0 * t)
smoothstep = def_generator(smoothstep)

def mix(x, y, a):
    """
    re-implementation of opengl mix
    https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/mix.xhtml
    The return value is computed as x×(1−a)+y×a
    """
    return x * (1.0 - a) + y * a
mix = def_generator(mix)

def rotation_y(p, a):
    c = cos(a)
    s = sin(a)
    new_p = [c * p[0] + s * p[2], -s * p[0] + c * p[2], p[1]]
    return new_p
rotation_y = def_generator(rotation_y)

def normal_functor(base_func, h, ndims, extra_term=[], extra_pos=[]):
    
    if len(extra_term) > 0:
        assert len(extra_term) == len(extra_pos)
        insert_order = np.argsort(extra_pos)
        
        extra_term = np.array(extra_term)[insert_order]
        extra_pos = np.array(extra_pos)[insert_order]
    
    def func(pos):
        vec = []
        base_offset = np.zeros(ndims)
        vec_norm = 0.0
        for i in range(ndims):
            base_offset[:] = 0
            base_offset[i] = h
            val_pos = base_func(pos + base_offset)
            val_neg = base_func(pos - base_offset)
            val_dif = val_pos - val_neg
            vec_norm = vec_norm + val_dif ** 2
            vec.append(val_dif)
            
        for i in range(len(extra_term)):
            current_pos = extra_pos[i]
            vec = vec[:current_pos] + [extra_term[i]] + vec[current_pos:]
            vec_norm = vec_norm + extra_term[i] ** 2
        
        ans = np.array(vec) / (vec_norm ** 0.5)
        return ans
    return def_generator(func)
            

def clip_0_1(val):
    return max(min(val, 1.0), 0.0)
clip_0_1 = def_generator(clip_0_1)

def dot(u, v):
    """
    Take dot product between two arrays.
    """
    return sum(u[i]*v[i] for i in range(len(u)))
dot = def_generator(dot)

def length(vec, norm):
    return sum(vec[i] ** norm for i in range(len(vec))) ** (1 / norm)
length = def_generator(length)

def rand(n):
    return fract(sin(n[0] * 12.9898 + n[1] * 4.1414) * 415.92653)
rand = def_generator(rand)

def rand2(n):
    return np.array([fract(sin(n[0] * 12.9898) * 415.92653), fract(sin(n[1] * 4.1414) * 415.92653)])
rand2 = def_generator(rand2)

def hash(n):
    return fract(cos(n) * 415.92653)
hash = def_generator(hash)

def hash2(n):
    return np.array([fract(sin(n) * 11.1451239123), fract(sin(n+1) * 34.349430423)])
hash2 = def_generator(hash2)

def hash3(n):
    return np.array([fract(sin(n) * 84.54531253), fract(sin(n+1) * 42.145259123), fract(sin(n+2) * 23.349041223)])
hash3 = def_generator(hash3)

def pseudo_noise_2d(p):
    """
    noise_2d and rand method from
    https://www.shadertoy.com/view/XsG3Dc
    """
    ip = numpy.array([floor(p[0]), floor(p[1])])
    u = numpy.array([fract(p[0]), fract(p[1])])
    u = u * u * (3.0 - 2.0 * u)
    res = mix(
              mix(rand(ip), rand([ip[0] + 1.0, ip[1]]), u[0]),
              mix(rand([ip[0], ip[1] + 1.0]), rand(ip + 1.0), u[0]),
              u[1])
    return res
pseudo_noise_2d = def_generator(pseudo_noise_2d)

def pseudo_noise_3d(x):
    p0 = floor(x[0])
    p1 = floor(x[1])
    p2 = floor(x[2])
    f0 = smoothstep(0.0, 1.0, fract(x[0]))
    f1 = smoothstep(0.0, 1.0, fract(x[1]))
    f2 = smoothstep(0.0, 1.0, fract(x[2]))
    n = p0 + p1 * 57.0 + 113.0 * p2

    return mix(mix(mix(hash(n +   0.0), hash(n +   1.0), f0),
                   mix(hash(n +  57.0), hash(n +  58.0), f0), f1),
               mix(mix(hash(n + 113.0), hash(n + 114.0), f0),
                   mix(hash(n + 170.0), hash(n + 171.0), f0), f1), f2)

pseudo_noise_3d = def_generator(pseudo_noise_3d)

def fbm_2d_functor(noise_2d, octaves):
    def fbm_2d(p):
        f = 0.5 * noise_2d(p)
        
        if octaves == 1:
            return f
        
        new_p0 = 1.6 * p[0] + 1.2 * p[1]
        new_p1 = -1.2 * p[0] + 1.6 * p[1]
        p = [new_p0, new_p1]
        f = f + 0.25 * noise_2d(p)
        
        if octaves == 2:
            return f
        
        new_p0 = 1.6 * p[0] + 1.2 * p[1]
        new_p1 = -1.2 * p[0] + 1.6 * p[1]
        p = [new_p0, new_p1]
        f = f + 0.1666 * noise_2d(p)
        
        if octaves == 3:
            return f
        
        new_p0 = 1.6 * p[0] + 1.2 * p[1]
        new_p1 = -1.2 * p[0] + 1.6 * p[1]
        p = [new_p0, new_p1]
        f = f + 0.0834 * noise_2d(p)
        
        if octaves == 4:
            return f
        raise
    return def_generator(fbm_2d)

def fbm_3d_functor(noise_3d, octaves):
    def fbm_3d(p):
        f = 0.5 * noise_3d(p)
        
        if octaves == 1:
            return f
        
        new_p0 = 1.1 * (-1.6 * p[1] - 1.2 * p[2])
        new_p1 = 1.1 * (1.6 * p[0] + 0.72 * p[1] - 0.96 * p[2])
        new_p2 = 1.1 * (1.2 * p[0] - 0.96 * p[1] + 1.28 * p[2])
        p[0] = new_p0
        p[1] = new_p1
        p[2] = new_p2
        f = f + 0.25 * noise_3d(p)
        
        if octaves == 2:
            return f
        
        new_p0 = 1.2 * (-1.6 * p[1] - 1.2 * p[2])
        new_p1 = 1.2 * (1.6 * p[0] + 0.72 * p[1] - 0.96 * p[2])
        new_p2 = 1.2 * (1.2 * p[0] - 0.96 * p[1] + 1.28 * p[2])
        p = [new_p0, new_p1, new_p2]
        f = f + 0.1666 * noise_3d(p)
        
        if octaves == 3:
            return f
        
        new_p0 = (-1.6 * p[1] - 1.2 * p[2])
        new_p1 = (1.6 * p[0] + 0.72 * p[1] - 0.96 * p[2])
        new_p2 = (1.2 * p[0] - 0.96 * p[1] + 1.28 * p[2])
        p = [new_p0, new_p1, new_p2]
        f = f + 0.0834 * noise_3d(p)
        
        if octaves == 4:
            return f
        
        raise
        
    return def_generator(fbm_3d)

def det2x2(A):
    """
    Calculate determinant of a 2x2 matrix
    """
    return A[0,0]*A[1,1] - A[0,1]*A[1,0]

def det3x3(A):
    """
    Calculate determinant of a 3x3 matrix
    """
    return A[0, 0] * A[1, 1] * A[2, 2] + A[0, 2] * A[1, 0] * A[2, 1] + \
           A[0, 1] * A[1, 2] * A[2, 0] - A[0, 2] * A[1, 1] * A[2, 0] - \
           A[0, 1] * A[1, 0] * A[2, 2] - A[0, 0] * A[1, 2] * A[2, 1]

def transpose_3x3(A):
    """
    transpose a 3x3 matrix
    """
    B = A[:]


def inv3x3(A):
    """
    Inverse of a 3x3 matrix
    """
    a00 = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    a01 = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
    a02 = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    a10 = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
    a11 = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    a12 = A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]
    a20 = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    a21 = A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]
    a22 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    det = det3x3(A)
    return numpy.array([[a00, a01, a02],
                        [a10, a11, a12],
                        [a20, a21, a22]]) / det

def matrix_vec_mul3(A, x):
    """
    Matrix multiplication with vector x
    Matrix is 3x3, x is 1x3
    """
    y0 = A[0, 0] * x[0] + A[0, 1] * x[1] + A[0, 2] * x[2]
    y1 = A[1, 0] * x[0] + A[1, 1] * x[1] + A[1, 2] * x[2]
    y2 = A[2, 0] * x[0] + A[2, 1] * x[1] + A[2, 2] * x[2]
    return numpy.array([y0, y1, y2])

def cross(x, y):
    """
    cross product of 2 length 3 vectors
    """
    z0 = x[1] * y[2] - x[2] * y[1]
    z1 = x[2] * y[0] - x[0] * y[2]
    z2 = x[0] * y[1] - x[1] * y[0]
    return numpy.array([z0, z1, z2])

def list_or_scalar_to_str(render_t):
    if hasattr(render_t, '__len__'):
        return ','.join(str(x) for x in render_t)
    else:
        return str(render_t)

def intersect_plane(X, window_size, camera_angular_velocity=1.0, camera_translation_velocity=50.0, camera_path=1):
    """
    Intersect unknown array X with infinite plane, returning (intersect_pos, viewer_dir).

    Here X[0] is window x coord, X[1] is window y coord, X[2] is time,
    window_size is (height, width) for window size, and the plane goes
    through the origin and has normal (0, 0, 1).
    """
    (height, width) = window_size
    t = Var('t', X[2])
    ray_dir = vec('ray_dir', [X[0]-width/2, X[1]+1, width/2])
    theta = Var('theta', t*camera_angular_velocity)

    if camera_path == 1:
        ray_dir_p = -vec('ray_dir_p', [ray_dir[0]*cos(theta)-ray_dir[2]*sin(theta),
                                       ray_dir[0]*sin(theta)+ray_dir[2]*cos(theta),
                                       ray_dir[1]])
        ray_origin = numpy.array([0.0, 0.0, 50.0])
    elif camera_path == 2:
        rotate_angle = -theta
        ray_dir_rotate = vec('ray_dir_rotate', [ray_dir[0]*cos(rotate_angle)-ray_dir[2]*sin(rotate_angle),
                                                ray_dir[0]*sin(rotate_angle)+ray_dir[2]*cos(rotate_angle),
                                                ray_dir[1]])
        ray_dir_p = vec('ray_dir_p', [ray_dir_rotate[0],
                                      ray_dir_rotate[1],
                                      ray_dir_rotate[2]])
        camera_radius = 300
        ray_origin = numpy.array([camera_radius * cos(theta),
                                  camera_radius * sin(theta),
                                  50.0+t*camera_translation_velocity])
    elif camera_path == 3:
        ray_dir_p = -vec('ray_dir_p', [ray_dir[0],
                                       ray_dir[2],
                                       ray_dir[1]])
        ray_origin = numpy.array([0.0, 0.0, 50.0])
    else:
        raise ValueError

    N = numpy.array([0.0, 0.0, 1.0])
    t_ray = Var('t_ray', -(N.dot(ray_origin))/N.dot(ray_dir_p))
    intersect_pos = vec('intersect_pos', ray_origin + t_ray * ray_dir_p)
    return (intersect_pos, -normalize(ray_dir_p))

def get_shader_dirname(base_dir, objective_functor, normal_map, geometry, render_prefix=False):
    if normal_map == '':
        normal_map = 'none'
    if isinstance(objective_functor, str):
        if render_prefix:
            objective_functor = objective_functor[len('render_'):]
        objective_name = objective_functor
    else:
        objective_name = objective_functor.__name__
    outdir = os.path.join(base_dir, objective_name + '_' + geometry + '_normal_' + normal_map)
    return outdir

def render_shader(objective_functor, window_size, trace=False, outdir=None, ground_truth_samples=None, use_triangle_wave=False, is_color=default_is_color, log_intermediates=None, normal_map='', nproc=None, geometry='', nframes=None, start_t=0.0, end_t=1.0, base_dir='out', base_ind=0, check_kw={}, get_objective=False, use_objective=None, verbose=True, camera_path=1, nfeatures=22, denoise=False, denoise_h=None, msaa_samples=None, compute_f=True, random_camera=False, z_min=10.0, z_max=60.0, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0, r_min=200, r_max=1000, camera_pos=None, gname='ground', code_only=False, top_view_only=False, zero_samples=False, z_log_range=True, count_edge=False, parallel_gpu=0, upright_pos=False, log_intermediates_level=0, log_intermediates_subset_level=1, extra_suffix='', render_t=None, log_t_ray=False, collect_loop_statistic=False, collect_loop_and_features=False, fov=None, sanity_check_loop_statistic=False, sanity_code=None, camera_pos_velocity=None, t_sigma=1/60.0, first_last_only=False, last_only=False, log_manual_features=False, subsample_loops=None, last_n=None, first_n=None, first_n_no_last=None, mean_var_only=False, every_nth=None, every_nth_stratified=False, stratified_random_file=None, one_hop_parent=False, tile_only=False, tile_start=None, batch_size=1, rescale_tile_start=None, render_sigma=0.5, texture_maps=None, use_texture_maps=False, partial_trace=1, chron_order=False, temporal_texture_buffer=False, automate_loop_statistic=False, def_loop_log_last=False, automate_raymarching_def=False, store_temporal_texture=True, log_only_return_def_raymarching=False, save_downsample_scale=1, SELECT_FEATURE_THRE=200, log_getitem=True, n_boids=None, camera_sigma=None, expand_boundary=0, allowed_cos=0, robust_simplification=False, collect_feature_mean_only=False, feature_normalize_dir='', reference_dir='', feature_start_ind=-1, feature_end_ind=-1):
    """
    Low-level routine for rendering a shader.

    If ground_truth_samples is positive, use that many samples MSAA for estimating ground truth: if 0 or negative, skip evaluation of ground truth.
    If nproc is given, use that many processes in parallel.
    If normal_map is given, it is a string description of a kind of normal mapping.
    If get_objective is True, then just return the objective function.
    If use_objective is not None, then use the given objective in place of generating one.
    """
    
    our_id = util.machine_process_id()
    camera_pos_file, render_t_file, render_index_file, compiler_file, camera_pos_velocity_file, tile_start_file, texture_file = get_filenames(our_id)
    
    
    
    
    
    if parallel_gpu > 0:
        assert log_intermediates == False and ground_truth_samples > 1

    if ground_truth_samples is None:
        ground_truth_samples = 1000
    #print('render_shader, ground_truth_samples:', ground_truth_samples)
    if log_intermediates is None:
        log_intermediates = False
    if window_size is not None:
        (height, width) = window_size
    else:
        (height, width) = default_render_size
    if outdir is None:
        outdir = get_shader_dirname(base_dir, objective_functor, normal_map, geometry)
        outdir = os.path.join(outdir, extra_suffix)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if verbose:
        print('Rendering shader to %s' % outdir)

    T0 = time.time()
    if use_objective:
        f = use_objective

    if not use_objective:

        X = ArgumentArray(ndims=nfeatures)
        

        f2 = objective_functor(X, use_triangle_wave)
        if get_objective:
            return f2
        if not use_objective:
            f = f2



    
    
    # automatically select a subsample number for loop, based on runtime ratio between logged and not logged program
    if automate_loop_statistic and log_intermediates:
        # if not log_intermediates, doesn't matter whether a subsample number is selected or not
        AUTOMATE_LOOP_CACHE = '__loop_%d.cache' % SELECT_FEATURE_THRE
        SELECT_SUBSAMPLE_THRE = 1.3
        #SELECT_FEATURE_THRE = 200
        MAX_SUBSAMPLE = 10
        MAX_EXAMPLE = 50
        old_feature_size = np.inf
        feature_size = np.inf
        
        def run_cmd(cmd):
            out = subprocess.check_output(cmd, shell=True)
            out = out.decode('utf-8')
            return out
        
        if not os.path.exists(AUTOMATE_LOOP_CACHE):
            file = open(AUTOMATE_LOOP_CACHE, 'w')
            file.write('')
            file.close()
        
        shadername = objective_functor.__name__
        if shadername.endswith('simplified'):
            shadername += '_proxy'
                
        try:
            
            lines  = run_cmd( "grep %s %s" % (shadername, AUTOMATE_LOOP_CACHE ))
        except:
            lines = ''
        
        subsample_count = None
        if len(lines) > 0:
            lines = lines.split('\n')
            for line in lines:
                read_vals = line.split()
                if len(read_vals) == 2:
                    if read_vals[0] == shadername:
                        subsample_count = int(read_vals[1])
                        break
            
        if subsample_count is None:
            #if camera_pos is not None:
            #    numpy.save(camera_pos_file, numpy.array(camera_pos))
            #    camera_args = ' --camera_pos_file %s ' % os.path.abspath(camera_pos_file)
            #else:
            #    camera_args = ''
            
            #if tile_start is not None:
            #    numpy.save(tile_start_file, tile_start)
                
            
            cwd = os.getcwd()
            os.chdir('../../../FastImageProcessing/CAN24_AN')
            
            test_dir = 'test_' + our_id

            cmd_base = 'python demo_refactored.py --name %s --add_initial_layers --initial_layer_channels 48 --add_final_layers --final_layer_channels 48 --conv_channel_multiplier 0 --conv_channel_no 48 --dilation_remove_layer --no_batch_norm --shader_name %s --geometry %s --data_from_gpu --identity_initialize --no_identity_output_layer --efficient_trace --collect_loop_statistic --lpips_loss --lpips_loss_scale 0.04 --fov small --tile_only --tiled_h 320 --tiled_w 320 --use_batch --batch_size 6 --render_sigma 0.3 --feature_size_only --no_dataroot --automate_raymarching_def ' % (test_dir, shadername, geometry)
            
            if shadername.endswith('simplified_proxy'):
                cmd_base += ' --patch_gan_loss --gan_loss_scale 0.05 --epoch 800 --discrim_train_steps 8 '
                
            if temporal_texture_buffer:
                cmd_base += ' --temporal_texture_buffer '
                
            if def_loop_log_last:
                cmd_base += ' --def_loop_log_last '
                
            if chron_order:
                cmd_base += ' --chron_order '
                
            if use_texture_maps:
                _, _, _, _, _, _, texture_file = get_filenames(our_id)
                
                if texture_maps is not None:
                    if isinstance(texture_maps, np.ndarray):
                        numpy.save(texture_file, texture_maps)
                    elif isinstance(texture_maps, str):
                        assert os.path.exists(texture_maps)
                        os.symlink(texture_maps, texture_file)
                        
                cmd_base += ' --texture_maps %s ' % texture_file
            
            log_intermediates_args = '--no_additional_features --ignore_last_n_scale 7 --include_noise_feature '
            
            subsample_cand = 1
            while True:
                try:
                    os.remove(os.path.join(test_dir, 'compiler_problem.py'))
                    os.remove(os.path.join(test_dir, 'option.txt'))
                except:
                    pass
                
                cmd = cmd_base + log_intermediates_args + '--every_nth %d' % subsample_cand
                old_feature_size = feature_size
                try:
                    out = subprocess.check_output(cmd, shell=True)
                    out = out.decode('utf-8')
                    print(out)
                    #render_times = parse_output_float(out, 'rough time estimate', multiple=True)[10:]
                    feature_size = parse_output_float(out, 'feature size', multiple=False)
                except:
                    feature_size = np.inf
                #print(subsample_cand, 'min / mean runtime:', np.min(render_times), np.mean(render_times))
                print(subsample_cand, 'feature size:', feature_size)
                #subsample_cand_time[i] = np.median(render_times)
                
                #if subsample_cand_time[i] < baseline_time * SELECT_SUBSAMPLE_THRE:
                if feature_size <= SELECT_FEATURE_THRE:
                    if abs(feature_size - SELECT_FEATURE_THRE) >= abs(old_feature_size - SELECT_FEATURE_THRE):
                        subsample_cand -= 1
                        feature_size = old_feature_size
                    subsample_count = subsample_cand
                    os.chdir(cwd)
                    file = open( AUTOMATE_LOOP_CACHE, 'a' )
                    file.write('%s %d\n' % (shadername, subsample_cand))
                    file.close()
                    break
                subsample_cand += 1
                    
        #subsample_loops = subsample_count
        every_nth = subsample_count
    
    
    
    print("log_intermediates_level", log_intermediates_level)
    print("log_intermediates_subset_level", log_intermediates_subset_level)
    c = CompilerParams(trace=trace, log_intermediates=log_intermediates, log_intermediates_level=log_intermediates_level, log_intermediates_subset_level=log_intermediates_subset_level, collect_loop_statistic=collect_loop_statistic or collect_loop_and_features, simplify_compute_graph=(not sanity_check_loop_statistic), first_last_only=first_last_only, last_only=last_only, subsample_loops=subsample_loops, last_n=last_n, first_n=first_n, first_n_no_last=first_n_no_last, mean_var_only=mean_var_only, every_nth=every_nth, every_nth_stratified=every_nth_stratified, stratified_random_file=stratified_random_file, one_hop_parent=one_hop_parent, chron_order=chron_order, automate_loop_statistic=automate_loop_statistic or automate_raymarching_def, def_loop_log_last=def_loop_log_last, log_only_return_def_raymarching=log_only_return_def_raymarching, log_getitem=log_getitem, robust_simplification=robust_simplification)
    if is_color:
        c.constructor_code = [OUTPUT_ARRAY + '.resize(3);']
    #    c.tf_global_code = [OUTPUT_ARRAY + '_len = 3']
    #else:
    #    c.tf_global_code = [OUTPUT_ARRAY + '_len = 0']
    c0 = copy.deepcopy(c)

    if nframes is None:
        nframes = 20
    if render_t is None:
        if nframes > 1:
            render_t = numpy.linspace(start_t, end_t, nframes)
        else:
            render_t = numpy.array([start_t])
    #render_t = numpy.array([0.0])
    #render_t = numpy.linspace(0.0, 1.0, 2)
    render_sigma_x = render_sigma_y = render_sigma
    render_sigma_t = 0.0
    g_samples = 1

    extra_args = '--render %d,%d --render_sigma %f,%f,%f --g_samples %d --outdir %s --min_time %f --max_time %f '%(width,height, render_sigma_x, render_sigma_y, render_sigma_t, g_samples, os.path.abspath(outdir), start_t, end_t)


    extra_args += ' --shader_only 1 --geometry_name %s --nfeatures %d --camera_path %d'%(geometry, nfeatures, camera_path)
    check_kw['ndims'] = 3

    extra_args = extra_args + ' --geometry_name ' + geometry + ' '
    if zero_samples:
        extra_args = extra_args + ' --zero_samples '
    if fov is not None:
        extra_args = extra_args + ' --fov ' + fov
    if expand_boundary > 0:
        extra_args = extra_args + ' --expand_boundary %d ' % expand_boundary
    if collect_loop_statistic is True:
        extra_args = extra_args + ' --collect_loop_statistic'
    if collect_loop_and_features is True:
        extra_args = extra_args + ' --collect_loop_statistic_and_features'
    if camera_pos_velocity is not None:
        extra_args = extra_args + ' --t_sigma %f ' % t_sigma
    if log_manual_features:
        extra_args = extra_args + ' --log_manual_features '
        
    extra_args = extra_args + ' --efficient_trace '
    
    if partial_trace < 1:
        extra_args = extra_args + ' --partial_trace %f ' % partial_trace
    if not store_temporal_texture:
        extra_args = extra_args + ' --no_store_temporal_texture '
    else:
        extra_args = extra_args + ' --store_temporal_texture '
    if save_downsample_scale > 1:
        extra_args = extra_args + ' --save_downsample_scale %d ' % save_downsample_scale
    if camera_sigma is not None:
        extra_args = extra_args + ' --camera_sigma ' + camera_sigma

    if collect_feature_mean_only:
        extra_args = extra_args + ' --collect_feature_mean_only --feature_normalize_dir %s --reference_dir %s --feature_start_ind %d --feature_end_ind %d ' % (feature_normalize_dir, reference_dir, feature_start_ind, feature_end_ind)


    if log_t_ray:
        extra_args = extra_args + ' --log_t_ray'

    if tile_only:
        extra_args = extra_args + ' --tile_only'

    if use_texture_maps:
        extra_args = extra_args + ' --use_texture_maps'
        
    if temporal_texture_buffer:
        extra_args = extra_args + ' --temporal_texture_buffer'

    if batch_size > 1:
        extra_args = extra_args + ' --batch_size %d' % batch_size

    if rescale_tile_start is not None:
        extra_args = extra_args + ' --rescale_tile_start %f' % rescale_tile_start
        
    if n_boids is not None:
        extra_args = extra_args + ' --n_boids %d' % n_boids

    #default_out_dir = '../csolver/'
    #def do_copy(i, basename, out_prefix=''):
    #    for ext in ['.png', '.npy']:
    #        shutil.copyfile(os.path.join(default_out_dir, 'render_' + basename+'%05d'%i + ext), os.path.join(outdir, (out_prefix if out_prefix != '' else basename)+'%05d'%i + ext))
    #print('check_kw:', check_kw)

    check_kw = dict(check_kw)
    check_kw.setdefault('print_command', True)
    if sanity_check_loop_statistic and sanity_code is not None:
        check_kw['sanity_code'] = sanity_code
    if 'extra_args' in check_kw:
        extra_args += check_kw['extra_args']
        del check_kw['extra_args']
        
    
        
    
    check(f, c, do_run=False, extra_args=extra_args, code_only=code_only, **check_kw)

    if code_only:
        target_file = COMPILER_PROBLEM_PY
        
        shutil.copyfile(compiler_file, os.path.join(base_dir, target_file))
        os.remove(compiler_file)
        return

    cmdL = []

    reci_z_min = 1.0 / z_min
    reci_z_max = 1.0 / z_max

    our_id = util.machine_process_id()
    camera_pos_file, render_t_file, render_index_file, compiler_file, camera_pos_velocity_file, tile_start_file, texture_file = get_filenames(our_id)
    render_index = numpy.array([i for i in range(render_t.shape[0])]) + base_ind
    render_index_file = our_id + 'render_index.npy'
    if not geometry.startswith('boids'):
        # 2D shader case
        if camera_pos is None:
            camera_pos = [None] * nframes
        backup_camera_pos = []
        for (i, t) in enumerate(render_t):
            if camera_pos[i] is None:
                if count_edge:
                    if len(backup_camera_pos) == 0:
                        ntries = 500
                        numpy.save(render_index_file, numpy.array(range(ntries)))
                        success_ind = -1
                        max_edge_count = 0
                        while len(backup_camera_pos) == 0:
                            camera_x = numpy.random.rand(ntries) * (x_max - x_min) + x_min
                            camera_y = numpy.random.rand(ntries) * (y_max - y_min) + y_min
                            if z_log_range:
                                reci_camera_z = numpy.random.rand(ntries) * (reci_z_max - reci_z_min) + reci_z_min
                                camera_z = 1.0 / reci_camera_z
                            else:
                                camera_z = numpy.random.rand(ntries) * (z_max - z_min) + z_min
                            if top_view_only:
                                rotation1 = numpy.pi * numpy.ones(ntries)
                                rotation2 = numpy.zeros(ntries)
                            else:
                                rotation1 = numpy.random.rand(ntries) * 2.0 * numpy.pi
                                rotation2 = numpy.random.rand(ntries) * 2.0 * numpy.pi
                            rotation3 = numpy.random.rand(ntries) * 2.0 * numpy.pi
                            camera_tuple = numpy.transpose([camera_x, camera_y, camera_z, rotation1, rotation2, rotation3])
                            numpy.save(camera_pos_file, camera_tuple)
                            numpy.save(render_t_file, numpy.array([t] * ntries))
                            check(f, c, do_compile=False, extra_args=extra_args + ' --gname ' + gname + ('' if msaa_samples is None else (' --samples %d' % msaa_samples)), **check_kw)
                            for k in range(ntries):
                                img = skimage.io.imread(os.path.join(outdir, gname + '%05d.png'%k))
                                img_gray = skimage.color.rgb2gray(img)
                                if numpy.sum(img_gray == 1.0) > 0:
                                    continue
                                img_edge = skimage.feature.canny(img_gray)
                                edge_count = numpy.sum(img_edge)
                                if edge_count > 10000:
                                    backup_camera_pos.append(camera_tuple[k, :])
                    camera_pos[i] = backup_camera_pos[0]
                    backup_camera_pos = backup_camera_pos[1:]
                    print('-------------------------------')
                    print("finished", i)
                    #camera_pos[i] = camera_tuple
                elif top_view_only:
                    #print('generating random top view position')
                #elif camera_pos[i] is None and top_view_only:
                    camera_x = numpy.random.rand() * (x_max - x_min) + x_min
                    camera_y = numpy.random.rand() * (y_max - y_min) + y_min
                    reci_camera_z = numpy.random.rand() * (reci_z_max - reci_z_min) + reci_z_min
                    camera_z = 1.0 / reci_camera_z
                    rotation1 = numpy.pi
                    rotation2 = 0.0
                    rotation3 = numpy.random.rand() * 2.0 * numpy.pi
                    camera_pos[i] = numpy.array([camera_x, camera_y, camera_z, rotation1, rotation2, rotation3])
                elif upright_pos:
                    print('generating upright camera position')
                    camera_x = numpy.random.rand() * (x_max - x_min) + x_min
                    camera_y = numpy.random.rand() * (y_max - y_min) + y_min
                    reci_camera_z = numpy.random.rand() * (reci_z_max - reci_z_min) + reci_z_min
                    camera_z = 1.0 / reci_camera_z
                    rotation1 = numpy.pi / 2.0
                    rotation2 = numpy.pi
                    rotation3 = numpy.random.rand() * 2.0 * numpy.pi
                    camera_pos[i] = numpy.array([camera_x, camera_y, camera_z, rotation1, rotation2, rotation3])
                else:
                    #print('generating random camera position for', geometry)
                    camera_x = numpy.random.rand() * (x_max - x_min) + x_min
                    camera_y = numpy.random.rand() * (y_max - y_min) + y_min
                    if geometry == 'plane':
                        reci_camera_z = numpy.random.rand() * (reci_z_max - reci_z_min) + reci_z_min
                        camera_z = 1.0 / reci_camera_z
                    else:
                        camera_z = numpy.random.rand() * (z_max - z_min) + z_min
                    rotation1 = numpy.random.rand() * 2.0 * numpy.pi
                    rotation2 = numpy.random.rand() * 2.0 * numpy.pi
                    rotation3 = numpy.random.rand() * 2.0 * numpy.pi
                    while True:
                        if geometry == 'plane':
                            if fov == 'small':
                                if numpy.cos(rotation1) * numpy.cos(rotation2) < -allowed_cos:
                                    break
                            else:
                                assert (fov == 'regular') or (fov is None)
                                assert allowed_cos == 0
                                if numpy.cos(rotation2) * (numpy.sin(rotation1) * height / 2.0 + numpy.cos(rotation1) * width / 2.0) < 0:
                                    break

                            rotation1 = numpy.random.rand() * 2.0 * numpy.pi
                            rotation2 = numpy.random.rand() * 2.0 * numpy.pi
                        elif geometry == 'hyperboloid1' or geometry == 'sphere' or geometry == 'paraboloid':
                            ray_p = -numpy.array([camera_x, camera_y, camera_z])
                            ray_p /= numpy.linalg.norm(ray_p)
                            sin1 = numpy.sin(rotation1)
                            cos1 = numpy.cos(rotation1)
                            sin2 = numpy.sin(rotation2)
                            cos2 = numpy.cos(rotation2)
                            sin3 = numpy.sin(rotation3)
                            cos3 = numpy.cos(rotation3)
                            rot_A = numpy.array([[cos2*cos3, (-cos1 * sin3 + sin1 * sin2 * cos3), (sin1 * sin3 + cos1 * sin2 * cos3)],
                                                 [cos2 * sin3, (cos1 * cos3 + sin1 * sin2 * sin3), (-sin1 * cos3 + cos1 * sin2 * sin3)],
                                                 [-sin2, sin1 * cos2, cos1 * cos2]])
                            orig_ray_dir_p = numpy.matmul(numpy.transpose(rot_A), ray_p)
                            scale = width / 2.0 / orig_ray_dir_p[2]
                            orig_ray_dir_p *= scale
                            if scale > 0.0 and orig_ray_dir_p[0] >= -width / 2.0 and orig_ray_dir_p[0] <= width / 2.0 and orig_ray_dir_p[1] >= 1.0 and orig_ray_dir_p[1] <= height + 1.0:
                                if geometry == 'sphere':
                                    radius = numpy.linalg.norm(numpy.array([camera_x, camera_y, camera_z]))
                                    if radius >= r_min and radius <= r_max:
                                        break
                                else:
                                    break
                                break
                            else:
                                camera_x = numpy.random.rand() * (x_max - x_min) + x_min
                                camera_y = numpy.random.rand() * (y_max - y_min) + y_min
                                camera_z = numpy.random.rand() * (z_max - z_min) + z_min
                                rotation1 = numpy.random.rand() * 2.0 * numpy.pi
                                rotation2 = numpy.random.rand() * 2.0 * numpy.pi
                                rotation3 = numpy.random.rand() * 2.0 * numpy.pi
                        else:
                            break

                    camera_pos[i] = numpy.array([camera_x, camera_y, camera_z, rotation1, rotation2, rotation3])

        numpy.save(camera_pos_file, numpy.array(camera_pos))
    if not geometry == 'boids':
        numpy.save(render_t_file, render_t)
    numpy.save(render_index_file, render_index)
    if tile_start is not None:
        numpy.save(tile_start_file, tile_start)
    if texture_maps is not None:
        if isinstance(texture_maps, np.ndarray):
            numpy.save(texture_file, texture_maps)
        elif isinstance(texture_maps, str):
            assert os.path.exists(texture_maps)
            os.symlink(texture_maps, texture_file)
    if camera_pos_velocity is not None:
        numpy.save(camera_pos_velocity_file, camera_pos_velocity)

    c.verbose = 1

    if ground_truth_samples > 0 and not check_kw.get('time_error', False):
        c = c0
        c.log_intermediates = False
        if parallel_gpu < 1:
            check(f, c, do_compile=False, extra_args=extra_args + ' --samples ' + str(ground_truth_samples) + ' --gname ' + gname + ' --is_gt 1', **check_kw)
        else:
            each_n = int(numpy.ceil(float(nframes) / float(parallel_gpu)))
            cmdL = []
            for i in range(parallel_gpu):
                camera_pos_each = camera_pos[i*each_n:(i+1)*each_n]
                render_t_each = render_t[i*each_n:(i+1)*each_n]
                render_index_each = render_index[i*each_n:(i+1)*each_n]
                if tile_start is not None:
                    tile_start_each = tile_start[i*each_n:(i+1)*each_n]

                camera_pos_file_new, render_t_file_new, render_index_file_new, compiler_file_new, camera_pos_velocity_file_new, tile_start_file_new, texture_file_new = get_filenames(our_id + gname + str(i))

                numpy.save(camera_pos_file_new, numpy.array(camera_pos_each))
                numpy.save(render_t_file_new, numpy.array(render_t_each))
                numpy.save(render_index_file_new, numpy.array(render_index_each))

                if tile_start is not None:
                    numpy.save(tile_start_file_new, numpy.array(tile_start_each))

                if texture_maps is not None:
                    if isinstance(texture_maps, np.ndarray):
                        numpy.save(texture_file_new, texture_maps)
                    elif isinstance(texture_maps, str):
                        assert os.path.exists(texture_maps)
                        os.symlink(texture_maps, texture_file_new)

                if camera_pos_velocity is not None:
                    camera_pos_velocity_each = camera_pos_velocity[i*each_n:(i+1)*each_n]
                    numpy.save(camera_pos_velocity_file_new, camera_pos_velocity_each)

                shutil.copyfile(compiler_file, compiler_file_new)
                cmdL.append(check(f, c, do_compile=False, get_command=True, our_id=our_id+gname+str(i), extra_args=extra_args + ' --samples ' + str(ground_truth_samples) + ' --gname ' + gname + ' --is_gt 1', **check_kw))
            open(our_id+'.txt', 'w').write('\n'.join(cmdL))
            print("write commands to " + our_id + '.txt')

    if not geometry.startswith('boids'):
        os.remove(camera_pos_file)
        os.remove(render_t_file)
    os.remove(render_index_file)
    shutil.copyfile(compiler_file, os.path.join(base_dir, 'compiler_problem.py'))
    os.remove(compiler_file)
    if tile_start is not None:
        os.remove(tile_start_file)
    if texture_maps is not None:
        if isinstance(texture_maps, np.ndarray):
            os.remove(texture_file)
        else:
            os.unlink(texture_file)
    if camera_pos_velocity is not None:
        os.remove(camera_pos_velocity_file)
    return {}

    

def render_any_shader(objective_functor, window_size, *args, **kw):
    """
    Render any shader allowing multiple camera paths.
    For multiple camera paths, compile each one, and append the index
    """
    ans = []
    geometry = kw['geometry']
    render_func = globals()['geometry_wrapper']

    num_cameras = kw.get('num_cameras', 1)
    nframes = kw['nframes']
    for i in range(num_cameras):
        #kw['base_ind'] = i * nframes
        kw['camera_path'] = kw.get('specific_camera_path', i+1)
        ans.append(render_func(objective_functor, window_size, *args, **kw))
    return ans

def normal_mapping(time, tex_coords, viewer_dir, unit_normal, tangent_t, tangent_b, *args, **kw):
    """
    dummy
    """
    normal_map = kw.get('normal_map', 'none')
    unit_new_normal = unit_normal

    return (unit_new_normal, tex_coords)

def render_plane_shader(objective_functor, window_size, *args, **kw):
    """
    Higher level function for rendering a shader on an infinite plane.

    Same arguments as render_shader, but objective_functor(X) receives X[0] and X[1] being the texture coordinates on a plane.
    """
    def objective(X, *args):
        camera_path = kw.get('camera_path', 1)
        (intersect_pos, viewer_dir) = intersect_plane(X, window_size, camera_path=camera_path)
        viewer_dir = vec('viewer_dir', viewer_dir)
        normal = numpy.array([0.0, 0.0, 1.0])
        
        tangent_t = numpy.array([1.0, 0.0, 0.0])
        tangent_b = numpy.array([0.0, 1.0, 0.0])
        (normal, intersect_pos) = normal_mapping(X[2], intersect_pos, viewer_dir, normal, tangent_t, tangent_b, *args, **kw)

        light_dir = vec('light_dir', normalize_const(numpy.array([0.3, 0.8, 1.0])))
        #intersect_pos[2] = z
        #light_dir = vec('light_dir', viewer_dir) #vec('light_dir', viewer_dir) #numpy.array([0.0, 0.0, 1.0])
        return objective_functor(intersect_pos, intersect_pos[:2], normal, light_dir, viewer_dir, *args, time=X[2])

    objective.__name__ = objective_functor.__name__
    kw2 = clear_extra_keywords(kw)
    return render_shader(objective, window_size, *args, **kw2)

def clear_extra_keywords(kw):
    kw2 = kw.copy()
    #kw2.pop('camera_path', 0)
    kw2.pop('num_cameras', 0)
    kw2.pop('need_time', 0)
    kw2.pop('specific_camera_path', 0)
    return kw2

def geometry_wrapper(objective_functor, window_size, *args, **kw):
    def objective(X, *args):
        if kw['geometry'] == 'none':
            time = X[0]
            ray_dir_p = numpy.array([X[1], X[2], X[3]])
            ray_origin = numpy.array([X[4], X[5], X[6]])
            return objective_functor(ray_dir_p, ray_origin, time)
        if kw['geometry'] == 'texture':
            time = X[0]
            uv = numpy.array([X[1], X[2]])
            mouse = numpy.array([X[3], X[4], X[5], X[6], X[7], X[8]])
            return objective_functor(uv, time, mouse)
        if kw['geometry'] == 'texture_approximate_10f':
            time = X[0]
            uv = numpy.array([X[1], X[2]])
            mouse = []
            for i in range(33):
                mouse.append(X[i+3])
            mouse = numpy.array(mouse)
            return objective_functor(uv, time, mouse)
        if kw['geometry'] == 'boids':
            return objective_functor(X[0], X[1], X[2], X[3], X[4], X[5])
        elif kw['geometry'] == 'boids_coarse':
            return objective_functor(X[0], X[1], X[2], X[3], X[4], X[5], X[6])
        
        intersect_pos = numpy.array([X[1], X[2], X[3]])
        tex_coords = numpy.array([X[8], X[9]])
        normal = numpy.array([X[10], X[11], X[12]])
        light_dir = numpy.array([X[19], X[20], X[21]])
        viewer_dir = numpy.array([X[4], X[5], X[6]])
        time = X[0]
        is_intersect = Var(log_prefix + 'is_intersect', X[7])
        tangent_t = numpy.array([X[13], X[14], X[15]])
        tangent_b = numpy.array([X[16], X[17], X[18]])

        if apply_geometry_optimization:
            if kw['geometry'] == 'plane':
                tex_coords = vec('tex_coords', numpy.array([intersect_pos[0], intersect_pos[1]]))
                normal = vec('normal', numpy.array([0.0, 0.0, 1.0]))
                light_dir = vec('light_dir', normalize_const(numpy.array([0.3, 0.8, 1.0])))
                tangent_t = vec('tangent_t', numpy.array([1.0, 0.0, 0.0]))
                tangent_b = vec('tangent_b', numpy.array([0.0, 1.0, 0.0]))
            elif kw['geometry'] in ['sphere', 'hyperboloid1', 'paraboloid']:
                tex_coords = vec('tex_coords', numpy.array([X[8], X[9]]))
                normal = vec('normal', numpy.array([X[10], X[11], X[12]]))
                light_dir = vec('light_dir', normalize_const(numpy.array([0.3, -1.0, 0.3])))
                tangent_t = vec('tangent_t', numpy.array([X[13], X[14], X[15]]))
                tangent_b = vec('tangent_b', numpy.array([X[16], X[17], X[18]]))

        (normal, tex_coords) = normal_mapping(time, tex_coords, viewer_dir, normal, tangent_t, tangent_b, *args, **kw)

        for expr in normal.tolist():
            expr.log_intermediates_subset_rank = 1
        ans = objective_functor(intersect_pos, tex_coords, normal, light_dir, viewer_dir, *args, time=time)
        return wrap_is_intersect(ans, is_intersect, kw['is_color'])
    objective.__name__ = objective_functor.__name__
    if apply_geometry_optimization:
        if kw['geometry'] == 'plane':
            kw['nfeatures'] = 8
        elif kw['geometry'] in ['sphere', 'hyperboloid1', 'paraboloid']:
            kw['nfeatures'] = 19
    kw2 = clear_extra_keywords(kw)
    return render_shader(objective, window_size, *args, **kw2)

def wrap_is_intersect(f, is_intersect, is_color):
    compare = is_intersect >= 0.0
    if is_color:
        if isinstance(f, Compound) and len(f.children) == 3 and all(isinstance(child, Assign) for child in f.children):
            ans = [child.children[1] for child in f.children]
        out_intensity_R = select_nosmooth(compare, ans[0], 0.0)
        out_intensity_G = select_nosmooth(compare, ans[1], 0.0)
        out_intensity_B = select_nosmooth(compare, ans[2], 0.0)
        out_intensity_R.log_intermediates_subset_rank = 1
        out_intensity_G.log_intermediates_subset_rank = 1
        out_intensity_B.log_intermediates_subset_rank = 1

        out_intensity = numpy.array([out_intensity_R, out_intensity_G, out_intensity_B])
        ans = output_color(out_intensity)
    else:
        ans = select_nosmooth(compare, f, 0.0)
    return ans

def multiple_shaders(base_func):
    def f(objective_functor_L, *args, **kw):
        ans = []
        for objective_functor in objective_functor_L:
            ans.append(base_func(objective_functor, *args, **kw))
        return ans
    return f

def output_color(c):
    """
    Given a array-like object with 3 Exprs, output an RGB color for a shader. Has the side effect of setting channels of c.
    """
    set_channels(c)
    return Compound([Assign(GetItem(ArgumentArray(OUTPUT_ARRAY), i), c[i]) for i in range(len(c))])

def intersect_keys(d, target_keys):
    return dict([(key, d.get(key, 0.0)) for key in target_keys])


def calculate_gradient_contribution(offsets, gis):
    t = 0.5 - offsets[:, 0] ** 2.0 - offsets[:, 1] ** 2.0 - offsets[:, 2] ** 2.0
    mapped_gis = map_gradients(gis)
    dot_products = offsets[:, 0] * mapped_gis[:, 0] + \
                   offsets[:, 1] * mapped_gis[:, 1] + \
                   offsets[:, 2] * mapped_gis[:, 2]

    ans = [None] * len(t)
    for i in range(len(t)):
        ans[i] = (t[i] >= 0) * (t[i] ** 4.0) * dot_products[i]
    return numpy.array(ans)

def map_gradients(gis):
    # gis: 10
    # return: 10x3
    ans = [None] * len(gis)
    for i in range(len(gis)):
        ans[i] = [lookup_table_np_grad0(gis[i]),
                  lookup_table_np_grad1(gis[i]),
                  lookup_table_np_grad2(gis[i])]
    return numpy.array(ans)

vertex_options = np.array([
    [1, 0, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0, 1],
    [0, 0, 1, 0, 1, 1],
    [0, 1, 0, 0, 1, 1],
    [0, 1, 0, 1, 1, 0]
], dtype=np.float32)

# Dimesions are: x0 >= y0, y0 >= z0, x0 >= z0
np_vertex_table = np.array([
    [[vertex_options[3], vertex_options[3]],
     [vertex_options[4], vertex_options[5]]],
    [[vertex_options[2], vertex_options[1]],
     [vertex_options[2], vertex_options[0]]]
], dtype=np.float32)

def get_simplex_vertices(offsets):
    # ans: 10x2x3
    vertices_all = [None] * len(offsets)

    def binary2index(binary, i):
        if i == 0:
            return 1.0 - binary
        elif i == 1:
            return binary
        else:
            raise

    for ind in range(len(offsets)):
        vertex_table_x_index = offsets[ind, 0] >= offsets[ind, 1]
        vertex_table_y_index = offsets[ind, 1] >= offsets[ind, 2]
        vertex_table_z_index = offsets[ind, 0] >= offsets[ind, 2]

        vertices = []
        for n in range(6):
            ans = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        ans = ans + binary2index(vertex_table_x_index, i) * binary2index(vertex_table_y_index, j) * binary2index(vertex_table_z_index, k) * np_vertex_table[i, j, k, n]
            vertices.append(ans)
        vertices_all[ind] = [[vertices[0], vertices[1], vertices[2]],
                           [vertices[3], vertices[4], vertices[5]]]

    return numpy.array(vertices_all)

render_shaders = multiple_shaders(render_shader)
render_any_shaders = multiple_shaders(render_any_shader)
render_plane_shaders = multiple_shaders(render_plane_shader)

FIX_NAMES = [
'tex_coords_x',
'tex_coords_y',
'tex_coords_z',
'normal_x',
'normal_y',
'normal_z',
'light_dir_x',
'light_dir_y',
'light_dir_z',
'tangent_t_x',
'tangent_t_y',
'tangent_t_z',
'tangent_b_x',
'tangent_b_y',
'tangent_b_z']
