from render_util import *
from render_single import *
import numpy
import math
import shutil

"""
modified from
https://www.shadertoy.com/view/MdXGW2
"""

BUMPFACTOR = 0.2
EPSILON = 0.1
BUMPDISTANCE = 200

CAMERASPEED = 15.

BUILDINGSPACING = 20.
MAXBUILDINGINSET = 12.

GALLERYHEIGHT = 10.5
GALLERYINSET = 2.5

texture_xdim = 1024
texture_ydim = 1024

def noise_3d_texture(x):
    z = x[2] * 64.0
    uv1_x = x[0] + 0.317 * floor(z)
    uv1_y = x[1] + 0.123 * floor(z)
    uv2_x = uv1_x + 0.317
    uv2_y = uv1_y + 0.123
    return mix(bilinear_texture_map(0, uv1_x * 4 * texture_xdim, uv1_y * 4 * texture_ydim, texture_xdim, texture_ydim), bilinear_texture_map(0, uv2_x * 4 * texture_xdim, uv2_y * 4 * texture_ydim, texture_xdim, texture_ydim), fract(z)) - 0.5

fbm = fbm_3d_functor(noise_3d_texture, 2)

def sdBox(p, b):
    dx = abs(p[0]) - b[0]
    dy = abs(p[1]) - b[1]
    dz = abs(p[2]) - b[2]
    ans0 = min(max(dx, max(dy, dz)), 0.0)
    new_d = [max(dx, 0.0), max(dy, 0.0), max(dz, 0.0)]
    return ans0 + dot(new_d, new_d) ** 0.5

def sdSphere(p, s):
    return length(p, 2) - s

def udBox(p, b):
    return length([max(abs(p[0]) - b[0], 0.0), max(abs(p[1]) - b[1], 0.0), max(abs(p[2]) - b[2], 0.0)], 2)

def sdCylinderXY(p, h):
    return length([p[0], p[1]], 2) - h[0]

def sdCylinderXZ(p, h):
    return max(length([p[0], p[2]], 2) - h[0], abs(p[1]) - h[1])

def sdTriPrism(p, h):
    return max(abs(p[2]) - h[1], max(abs(p[0]) * 0.866025 + p[1] * 0.5, -p[1]) - h[0] * 0.5)

def opS(d1, d2):
    return max(-d2, d1)

def opU(d1, d2):
    return min(d2, d1)

def opU_2d(d1, d2):
    cond = d1[0] < d2[0]
    return [select(cond, d1[0], d2[0]), select(cond, d1[1], d2[1])]

def opI(d1, d2):
    return max(d1, d2)

def getXoffset(z):
    return 20 * sin(z * 0.02)

def getBuildingInfo(pos):
    res_x = floor(pos[2] / BUILDINGSPACING + 0.5)
    res_y = res_x * BUILDINGSPACING
    res_x = res_x * sign(pos[0] + getXoffset(pos[2]))
    return [res_x, res_y]

def getBuildingParams(buildingindex):
    h = hash3(buildingindex)
    return [20 + 4.5 * floor(h[0] * 7), h[1] * MAXBUILDINGINSET, (sign_up(0.5 - h[2]) + 1) / 2, (sign_up(0.25 - abs(h[2] - 0.4)) + 1) / 2]

def baseBuilding(pos, h):
    tpos = [pos[2], pos[1], pos[0]]
    res = opS(udBox(tpos, [8.75, h, 8.75]),
              opS(opU(sdBox([((tpos[0] + 1.75) % 3.5) - 1.75, ((tpos[1] + 4.5) % 9) - 2.5, tpos[2] - 5], [1, 2, 4]),
                      sdCylinderXY([((tpos[0] + 1.75) % 3.5) - 1.75, ((tpos[1] + 4.5) % 9) - 4.5, tpos[2] - 5], [1, 4])),
                  udBox([tpos[0], tpos[1] - h, tpos[2]], [9, 1, 9])))
    
    res = opU(res,
              opI(udBox(tpos, [8.75, h, 8.75]),
                  opU(udBox([((tpos[0] + 1.75) % 3.5) - 1.75, tpos[1], tpos[2] - 8.45], [0.05, h, 0.05]),
                      udBox([tpos[0], ((tpos[1] + 0.425) % 1.75) - 0.875, tpos[2] - 8.45], [10, 0.05, 0.05]))))
    return res

def baseGallery(pos):
    tpos = [pos[2], pos[1], pos[0]]
    res = opU(opS(udBox([tpos[0], tpos[1], tpos[2]-GALLERYINSET], [8.75, GALLERYHEIGHT, 0.125]),
                  opU(sdBox([((tpos[0] + 1.75) % 3.5) - 1.75, tpos[1] - 5, tpos[2] - 5], [1.6, 3, 10]),
                      sdCylinderXY([((tpos[0] + 1.75) % 3.5) - 1.75, tpos[1] - 8, tpos[2] - 5], [1.6, 10]))),
              sdTriPrism([tpos[2] + 3.4, -44.4 + 3.9 * tpos[1], tpos[0]], [7.5, 8.7]))
    return res

def baseBalcony(pos, h):
    res = opI(udBox(pos, [9, h, 9]),
              sdBox([pos[0], ((pos[1] + 4.5) % 9) - 7.5, pos[2] - 5], [40, 0.5, 40]))
    return res

def baseBridge(pos):
    new_pos = [pos[0] * 0.38, pos[1], pos[2]]
    res = opS(opU(sdBox(new_pos, [4, 2, 2.5]),
                  sdTriPrism([new_pos[0], -8 + 3 * new_pos[1], new_pos[2]], [4.5, 2.5])),
              sdCylinderXY([new_pos[0], new_pos[1] + 1.5, new_pos[2]], [3.8, 3]))
    return res

def mapSimpleTerrain(p):
    new_p_x = p[0] + getXoffset(p[2])
    new_p_x = -abs(new_p_x)
    res = [udBox([new_p_x + 30, p[1] - 1, p[2]], [20, 100.25, 99999]), 1]
    
    zcenter = ((p[2] + 60) % 120) - 70

    res = opU_2d(res, [baseBridge([new_p_x, p[1], zcenter]), 8])
    return min(res[0], p[1] + 10)

def mapTerrain(p):
    buildingInfo = getBuildingInfo(p)
    buildingParams = getBuildingParams(buildingInfo[0])
    pos = [p[0], p[1], p[2]]
    pos[0] = pos[0] + getXoffset(pos[2])
    pos[0] = -abs(pos[0])
    
    res = [udBox([pos[0] + 30, pos[1], pos[2]], [20, 0.25, 99999]), 1]
    
    z = buildingInfo[1]
    zcenter = ((pos[2] + 10) % 20) - 10
    
    
    
    res = opU_2d(res, [baseBridge([pos[0], pos[1], ((pos[2] + 60) % 120) - 70]), 8])
    res = opU_2d(res, [sdSphere([pos[0] + 11.5, pos[1] - 6, zcenter], 0.5), 3])
    
    
    res = opU_2d(res, [sdSphere([pos[0] + 11.5, pos[1] - 5.4, zcenter + 0.6], 0.35), 3])
    res = opU_2d(res, [sdSphere([pos[0] + 11.5, pos[1] - 5.4, zcenter - 0.6], 0.35), 3])
    
    
    
    res = opU_2d(res, [sdCylinderXZ([pos[0] + 11.5, pos[1], zcenter], [0.1, 6]), 4])
    
    
    
    pos = [pos[0] + 28.75 + buildingParams[1], pos[1] + 2.5, pos[2]]
    res = opU_2d(res, [baseBuilding([pos[0], pos[1], zcenter], buildingParams[0] + 2.5), 2])
    #return [min(res[0], 11 - zcenter), res[1]]
    
    pos[0] = pos[0] - 8.75 - GALLERYINSET
    
    res = mix(res, opU_2d(res, [baseGallery([pos[0], pos[1], zcenter]), 5]), buildingParams[2])
    
    return [min(res[0], 11 - zcenter), res[1]]

def waterHeightMap(pos, time):
    posm_x = 0.02 * (pos[0] * 0.8 - pos[1] * 0.6) + 0.001 * time
    posm_y = 0.02 * (pos[0] * 0.6 + pos[1] * 0.8)
    f = fbm([posm_x * 1.9, posm_y * 1.9, time * 0.01])
    height = 0.5 + 0.1 * f
    height = height + 0.05 * sin(posm_x * 6.0 + 10.0 * f)
    return height

def intersectPlane(ro, rd, height):
    d = select(rd[1] != 0, -(ro[1] - height) / rd[1], -100)
    d = min(100000.0, d)
    d = select(d > 0, d, -100)
    # if d > 0, it's true, otherwise, it's false
    return d

def intersectSphere(ro, rd, sph):
    ds = ro - sph[:3]
    bs = dot(rd, ds)
    cs = dot(ds, ds) - sph[3] * sph[3]
    ts = bs * bs - cs
    
    new_ts = -bs - select(ts > 0, sqrt(ts), -1)
    
    return_normal = ((ro + ts * rd) - sph[:3]) / sph[3]
    return_normal_norm = length(return_normal, 2)
    
    return_cond = (ts > 0) * (new_ts > 0)
    normal_x = select(return_cond, return_normal[0] / return_normal_norm, -10)
    normal_y = select(return_cond, return_normal[1] / return_normal_norm, -10)
    normal_z = select(return_cond, return_normal[2] / return_normal_norm, -10)
    # if normal_x >= -1, then it's true, otherwise, it's false
    return [normal_x, normal_y, normal_z]

def intersect(ro, rd):
    maxd = 1500
    precis = 0.01
    h = ConstExpr(precis * 2)
    t = ConstExpr(0.0)
    d = ConstExpr(0.0)
    m = ConstExpr(1.0)
    update_cond = ConstExpr(True)
    
    for i in loop_generator(100, is_raymarching=True):
        update_cond = update_cond * (abs(h) >= precis) * (t <= maxd)
        t = select(update_cond, t + h, t)
        mt = mapTerrain(ro + rd * t)
        h = mt[0] * 0.9
        m = mt[1]
        
    m = select(t > maxd, -1, m)
    return [t, h, m]

def intersectSimple(ro, rd):
    maxd = 10000
    precis = 0.01
    h = ConstExpr(precis * 2)
    t = ConstExpr(0.0)
    update_cond = ConstExpr(True)
    
    for i in loop_generator(20, is_raymarching=True):
        update_cond = update_cond * (abs(h) >= precis) * (t <= maxd)
        t = select(update_cond, t + h, t)
        h = mapSimpleTerrain(ro + rd * t)
    return t

def getSkyColor(rd):
    lig = np.array([-2.5, 1.7, 2.5]) / np.linalg.norm(np.array([-2.5, 1.7, 2.5]))
    
    bgcol = 1.1 * np.array([0.15, 0.15, 0.4]) - rd[1] * 0.4
    bgcol = bgcol * 0.3
    moon = max(min(dot(rd, lig), 1.0), 0.0)
    bgcol = bgcol + np.array([2, 1.5, 0.8]) * 0.015 * (moon ** 32)
    # col = bgcol when SHOW_MOON_AND_CLOUDS is false
    return bgcol

texcube = texcube_functor(texture_xdim, texture_ydim)

def venice_simplified_proxy_20_100(ray_dir_p, ray_origin, time):
    ro = ray_origin
    rd = ray_dir_p
    distSimple = intersectSimple(ro, rd)
        
    dist = intersectPlane(ro, rd, 0)
    
    reflection = (dist > 0.0) * (dist < distSimple)
    
    ro = ro + rd * select(reflection, dist, 0.0)
    depth = mapTerrain(ro)[0]
    totaldist = select(reflection, dist, 0.0)
    bumpfactor = BUMPFACTOR * (1.0 - smoothstep(0.0, BUMPDISTANCE, dist))
    
    normal = normal_functor(lambda x: -bumpfactor * waterHeightMap(x, time) / (2. * EPSILON), EPSILON, 2, extra_term=[1.], extra_pos=[1])(ro[[0, 2]])
    normal_x = select(reflection, normal[0], 0.0)
    normal_y = select(reflection, normal[1], 0.0)
    normal_z = select(reflection, normal[2], 0.0)
    normal = np.array([normal_x, normal_y, normal_z])
    
    
    rd = rd - select(reflection, 2 * dot(normal, rd), 0.0) * normal
        
    tmat = intersect(ro, rd)

    
    totaldist = totaldist + tmat[0]
        
    bgcol = getSkyColor(rd)
    col = [bgcol[0], bgcol[1], bgcol[2]]
    pos = ro + tmat[0] * rd
    
    
    hit_building = (tmat[2] > -0.5) * (totaldist < 500)
    
    
    buildingInfo = getBuildingInfo(pos)
    
    
    buildingParams = getBuildingParams(buildingInfo[0])
    
    
    z = buildingInfo[1]
    lp = np.array([11.5 * sign(buildingInfo[0]) - getXoffset(z), 6., z])
    lig = lp - pos
    lig_norm = length(lig, 2)
    lig = lig / lig_norm
    
    nor = normal_functor(lambda x: mapTerrain(x)[0], 0.1, 3)(pos)
    #nor = calcNormal(pos)
    
    
    matpos = pos * 0.3
    
    hit_gallery = abs(tmat[2] - 5) < 0.5
    factor = select(hit_gallery, 0.2, 0.4)
    mate_x = texcube(1, matpos, nor) * factor
    mate_y = texcube(2, matpos, nor) * factor
    mate_z = texcube(3, matpos, nor) * factor
    
    #mate_x = select(hit_gallery, texcube(1, matpos, nor) * 0.2, texcube(1, matpos, nor) * 0.4)
    #mate_y = select(hit_gallery, texcube(2, matpos, nor) * 0.2, texcube(2, matpos, nor) * 0.4)
    #mate_z = select(hit_gallery, texcube(3, matpos, nor) * 0.2, texcube(3, matpos, nor) * 0.4)
    
    
    origmate_x = mate_x
    origmate_y = mate_y
    origmate_z = mate_z
    
    
    
    hit_light = abs(tmat[2] - 3) < 0.5
    mate_x = select(hit_light, 160*1.3, mate_x)
    mate_y = select(hit_light, 160*1.1, mate_y)
    mate_z = select(hit_light, 160*0.4, mate_z)
    
    
    
    hit_building2 = abs(tmat[2] - 2) < 0.5
    
    
    mate_x = mate_x * select(hit_building2, min(max(4 * bilinear_texture_map(4, buildingInfo[0] * 1.4231153121 * texture_xdim, buildingInfo[0] * 1.4231153121 * texture_ydim, texture_xdim, texture_ydim), 0.0), 1.0), 1.0)
    mate_y = mate_y * select(hit_building2, min(max(4 * bilinear_texture_map(5, buildingInfo[0] * 1.4231153121 * texture_xdim, buildingInfo[0] * 1.4231153121 * texture_ydim, texture_xdim, texture_ydim), 0.0), 1.0), 1.0)
    mate_z = mate_z * select(hit_building2, min(max(4 * bilinear_texture_map(6, buildingInfo[0] * 1.4231153121 * texture_xdim, buildingInfo[0] * 1.4231153121 * texture_ydim, texture_xdim, texture_ydim), 0.0), 1.0), 1.0)
    
    
    
    #occ = calcAO(pos, nor)
    occ = 1.0
    amb = max(min(0.5 + 0.5 * nor[1], 1.0), 0.0)
    dif = max(dot(nor, lig), 0.0)
    
    
    
    hit_gallery_top = hit_gallery * (pos[1] > GALLERYHEIGHT - 2.6)
    dif = select(hit_gallery_top, abs(dot(nor, lig)), dif)
    mate_x = select(hit_gallery_top, 0.3, mate_x)
    mate_y = select(hit_gallery_top, 0.0, mate_y)
    mate_z = select(hit_gallery_top, 0.0, mate_z)
    mate = np.array([mate_x, mate_y, mate_z])
    
    dif = dif / dot(lp - pos, lp - pos)
    
    
    bac_vec = np.array([-lig[0], 0.0, -lig[2]])
    bac_vec_norm = length(bac_vec, 2)
    bac_vec = bac_vec / bac_vec_norm
    bac = max(0.2 + 0.8 * dot(nor, bac_vec), 0.0)
    
    above_gallery = (abs(buildingParams[2] - 1) < 0.5) * (pos[1] > GALLERYHEIGHT)
    lcol_x = select(above_gallery, 2.9, 1.3)
    lcol_y = select(above_gallery, 1.65, 0.6)
    lcol_z = select(above_gallery, 0.65, 0.4)
    lcol = np.array([lcol_x, lcol_y, lcol_z])
    
    brdf = 60.0 * dif * lcol + 0.1 * amb * np.array([0.1, 0.15, 0.3]) + 0.1 * bac * np.array([0.09, 0.03, 0.01])
    col_building = mate * brdf * occ
    col[0] = select(hit_building, col_building[0], col[0])
    col[1] = select(hit_building, col_building[1], col[1])
    col[2] = select(hit_building, col_building[2], col[2])
    
    isLeft = sign(buildingInfo[0])
    inRoom = ((pos[0] + getXoffset(pos[2])) * isLeft > buildingParams[1] + 20.25) * (abs(pos[2] - buildingInfo[1]) < 8.5) * (pos[1] < buildingParams[0] - 0.5) * hit_building
    
    roomcoord_x = floor((pos[2] - buildingInfo[1] + 5) / 3.5) * 3.5 + floor((buildingInfo[1] + 5) / 10) * 10
    roomcoord_y = floor(pos[1] / 9) * 9
    
    room_light_on = inRoom * (noise_3d_texture([roomcoord_x * 1.15321*isLeft, roomcoord_y * 1.15321*isLeft, time * 0.0005]) > -0.1)
    rlc = np.array([(buildingParams[1] + 3 + 20.25) * isLeft - getXoffset(roomcoord_x - 5), roomcoord_y + 5.5, roomcoord_x - 5])
    ld = rlc - pos
    ld_norm = length(ld, 2)
    ld_normalized = ld / ld_norm
    
    dif = max(dot(nor, ld_normalized), 0.0) / dot(ld, ld)
    col[0] = col[0] + select(room_light_on, origmate_x * (dif * 120) * bilinear_texture_map(4, roomcoord_x * 0.1231 * texture_xdim, roomcoord_y * 0.1231 * texture_ydim, texture_xdim, texture_ydim), 0.0)
    col[1] = col[1] + select(room_light_on, origmate_y * (dif * 120) * bilinear_texture_map(5, roomcoord_x * 0.1231 * texture_xdim, roomcoord_y * 0.1231 * texture_ydim, texture_xdim, texture_ydim), 0.0)
    col[2] = col[2] + select(room_light_on, origmate_z * (dif * 120) * bilinear_texture_map(6, roomcoord_x * 0.1231 * texture_xdim, roomcoord_y * 0.1231 * texture_ydim, texture_xdim, texture_ydim), 0.0)
    
    
    
    basez = floor(pos[2] / 2) * 2 - 2
    for i in loop_generator(3, is_raymarching=True):
        buildingInfo = getBuildingInfo([pos[0], pos[1], basez])
        update_loop = ((abs(basez) - buildingInfo[1]) <= 8.75) * hit_building
        
        buildingParams_y = select(update_loop, getBuildingParams(buildingInfo[0])[1], buildingParams[1])
        rlc = np.array([(buildingParams_y - 1 + 20.25) * isLeft - getXoffset(basez), 7.7 - 1.5 * abs(sin(basez * 0.3)), basez])
        ld = rlc - pos
        ld_norm = length(ld, 2)
        ld_normalized = ld / ld_norm
        dif = max(dot(nor, ld_normalized), 0.0) / dot(ld, ld)
        col[0] = col[0] + select(update_loop, mate[0] * dif * 6 * bilinear_texture_map(4, basez * time * 0.0001 * 0.1231 * texture_xdim, basez * time * 0.0001 * 0.1231 * texture_ydim, texture_xdim, texture_ydim), 0.0)
        col[1] = col[1] + select(update_loop, mate[1] * dif * 6 * bilinear_texture_map(5, basez * time * 0.0001 * 0.1231 * texture_xdim, basez * time * 0.0001 * 0.1231 * texture_ydim, texture_xdim, texture_ydim), 0.0)
        col[2] = col[2] + select(update_loop, mate[2] * dif * 6 * bilinear_texture_map(6, basez * time * 0.0001 * 0.1231 * texture_xdim, basez * time * 0.0001 * 0.1231 * texture_ydim, texture_xdim, texture_ydim), 0.0)
        basez = basez + 2
        
        
    col[0] = col[0] * select(reflection * hit_building, 0.9 * 0.8 * (0.5 + min(max(depth * 2, 0.0), 0.5)), 1.0)
    col[1] = col[1] * select(reflection * hit_building, 0.9 * 0.9 * (0.5 + min(max(depth * 2, 0.0), 0.5)), 1.0)
    col[2] = col[2] * select(reflection * hit_building, 0.9 * 1.0 * (0.5 + min(max(depth * 2, 0.0), 0.5)), 1.0)
    
    
        
    # the exponetial modeling on dist may cause numerical instability on tf, so didn't include

    # didn't decrease brightness at the edge of image
    col[0] = (min(max(col[0], 0.0), 1.0) ** 0.45) * 1.03
    col[1] = (min(max(col[1], 0.0), 1.0) ** 0.45) * 1.02
    col[2] = (min(max(col[2], 0.0), 1.0) ** 0.45) 
    
    for expr in col + nor.tolist() + [totaldist]:
        expr.log_intermediates_subset_rank = 1
    
    return output_color(col)

shaders = [venice_simplified_proxy_20_100]
is_color = True
fov = 'small'

def path_np(time):
    z = time * CAMERASPEED
    return [-20 * np.sin(z * 0.02) + 5 * np.cos(time * 0.1), 1.25 * np.ones(z.shape), z]

def main():
    
    if len(sys.argv) < 3:
        print('Usage: python render_[shader].py mode base_dir')
        raise
        
    mode = sys.argv[1]
    base_dir = sys.argv[2]
    
    camera_dir = os.path.join(base_dir, 'datasets/datas_venice_simplified_20_100_new_extrapolation')
    preprocess_dir = os.path.join(base_dir, 'preprocess/venice')
    
    if not os.path.exists(camera_dir):
        os.makedirs(camera_dir, exist_ok=True)
    
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir, exist_ok=True)
    
    texture_maps = os.path.join(base_dir, 'datasets/venice_texture.npy')
        
    if mode == 'collect_raw':
        
        camera_pos = numpy.load(os.path.join(camera_dir, 'train.npy'))
        render_t = numpy.load(os.path.join(camera_dir, 'train_time.npy'))
        nframes = render_t.shape[0]
        
        train_start = numpy.load(os.path.join(camera_dir, 'train_start.npy'))
        render_single(os.path.join(preprocess_dir, 'train'), 'render_venice_simplified_proxy_20_100', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 80), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': 'train_small', 'tile_only': True, 'tile_start': train_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'texture_maps': texture_maps, 'use_texture_maps': True})
        
    elif mode == 'generate_dataset':
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
                    
            outdir = get_shader_dirname(os.path.join(preprocess_dir, mode), shaders[0], 'none', 'none')
                
            render_single(os.path.join(preprocess_dir, mode), 'render_venice_simplified_proxy_20_100', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = render_size, render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_ground' % mode, 'tile_only': tile_only, 'tile_start': tile_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'texture_maps': texture_maps, 'use_texture_maps': True})
            
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
        
    return
        
if __name__ == '__main__':
    main()
        