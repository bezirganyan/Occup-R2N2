import os, json
import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand, choice
from pytorch3d.io import load_obj, save_obj

import binvox_rw #install it from here: https://github.com/dimatura/binvox-rw-py

obj_1_code = '03001627'
obj_2_code = '02933112'

path_to_json = ['/media/vcucu/Local Disk/ShapeNetCore.v2/' + obj_1_code + '/', #chair
                '/media/vcucu/Local Disk/ShapeNetCore.v2/' + obj_2_code + '/'] #cabinet

d = [{}, {}]
arr = [[], []]

for i in range(2):
    for filename in os.listdir(path_to_json[i]):
        for files in os.listdir(os.path.join(path_to_json[i], filename, 'models')):
            if files.endswith('.json'):
                json_file = open(os.path.join(path_to_json[i], filename, 'models', files))
                for line in json_file.readlines():
                    json_dict = json.loads(line)
                    x_min = np.asarray(json_dict['min'])
                    x_max = np.asarray(json_dict['max'])
                    l2 = np.linalg.norm(x_max - x_min, 2)
                    d[i][filename]=l2
                    arr[i].append(l2)

good_meshes=[[], []]
m = [np.median(arr[0]), np.median(arr[1])]
for i in range(2):
    for j in list(d[i].keys()):
        if d[i][j] < 1.5*m[i] and d[i][j] > 0.75*m[i]:
            good_meshes[i].append(j)


def object_near_object(min1, max1, min2, max2):
    return [min2[0] - max1[0] - rand(1)[0] * (max1[0] - min1[0]) / 5.0, min2[1] - min1[1], max2[2] - max1[2]]

def object_on_object(min1, max1, min2, max2):
    return [min2[0] - min1[0] + rand(1)[0] * (max2[0] - min2[0] - (max1[0] - min1[0])),
           max2[1] - min1[1], min2[2] - min1[2] + rand(1)[0] * (max2[2] - min2[2] - (max1[2] - min1[2]))]

#One object placed randomly near another object at same height

n = 10000 #number of generated meshes
multi_lst = []
min_max = []
name_obj = 'chair_near_cabinet_'

for i in range(n):
    f1 = choice(good_meshes[0])
    obj_filename1 = os.path.join(path_to_json[0], f1, "models/model_normalized.obj")
    f2 = choice(good_meshes[1])
    obj_filename2 = os.path.join(path_to_json[1], f2, "models/model_normalized.obj")

    verts1, faces1, _ = load_obj(obj_filename1)
    verts2, faces2, _ = load_obj(obj_filename2)

    scale1 = d[0][f1]
    scale2 = d[1][f2]

    verts1 *= scale1
    verts2 *= scale2

    min1 = verts1.min(0).values
    max1 = verts1.max(0).values
    min2 = verts2.min(0).values
    max2 = verts2.max(0).values

    dx = object_near_object(min1, max1, min2, max2)
    verts1 += torch.Tensor(np.array(dx))
    min1 += torch.Tensor(np.array(dx))
    max1 += torch.Tensor(np.array(dx))

    temp = faces2.verts_idx
    temp += verts1.size()[0]

    verts = torch.cat([verts1,verts2],0)
    faces = torch.cat([faces1.verts_idx,temp],0)

    final_obj = os.path.join('./generated/', obj_1_code, name_obj+str(i), 'model_multi.obj')
    os.makedirs(os.path.join('./generated/', obj_1_code, name_obj+str(i)))
    save_obj(final_obj, verts, faces)

    multi_lst.append(name_obj+str(i))
    min_max.append([min1, max1, min2, max2])

with open(os.path.join('./generated/', obj_1_code, 'obj_multi.lts'), 'w') as f:
    for item in multi_lst:
        f.write("%s\n" % item)

for i in range(n):
    with open(os.path.join('./generated/', obj_1_code, name_obj+str(i), 'model_multi.binvox'), 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    min1, max1, min2, max2 = min_max[i]
    obj_arr = (model.data+0)
    scale = 32.0 / max(max2-min1)
    max1 -= min1
    max1 *= scale
    max1 = np.ceil(max1)
    max2 -= min1
    max2 *= scale
    max2 = np.ceil(max2)
    min2 -= min1
    min2 *= scale
    min2 = np.ceil(min2)
    min1 -= min1

    max1_x = min(32,int(max1[0].item()))
    if obj_arr[max1_x-1,:,:].sum() - obj_arr[max1_x-2,:,:].sum() >= obj_arr[max1_x-2,:,:].sum() - obj_arr[max1_x-3,:,:].sum():
        max1_x -= 1
    for x in range(max1_x):
        for y in range(min(32,int(max1[1].item()))):
            for z in range(min(32,int(max1[2].item()))):
                if obj_arr[x,y,z] == 1:
                    obj_arr[x,y,z] = 2

    minimum1, maximum1, minimum2, maximum2 = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    for x in range(32):
        for y in range(32):
            for z in range(32):
                if obj_arr[x,y,z] == 2:
                    if x > maximum2[0]: maximum2[0] = x
                    if y > maximum2[1]: maximum2[1] = y
                    if z > maximum2[2]: maximum2[2] = z
                if obj_arr[x,y,z] == 1:
                    if minimum1[0] == 0: minimum1[0] = x
                    if minimum1[1] == 0: minimum1[1] = y
                    if minimum1[2] == 0: minimum1[2] = z
                    if x > maximum1[0]: maximum1[0] = x
                    if y > maximum1[1]: maximum1[1] = y
                    if z > maximum1[2]: maximum1[2] = z

    center1 = np.floor((maximum1 + minimum1)/2)
    center2 = np.floor((maximum2 + minimum2)/2)

    centers = [center1, center2]

    #centers = np.zeros((32,32,32))
    #centers[int(center1[0])][int(center1[1])][int(center1[2])] = 1
    #centers[int(center2[0])][int(center2[1])][int(center2[2])] = 2

    #centers = np.array([center2, center1]).tolist()
    np.save(os.path.join('./generated/', obj_1_code, name_obj+str(i), 'model_multi.npy'), obj_arr)
    np.save(os.path.join('./generated/', obj_1_code, name_obj+str(i), 'centers.npy'), centers)
    #with open(os.path.join('./generated/', obj_1_code, name_obj+str(i), 'centers.lts'), 'w') as f:
    #    for item in centers:
    #        f.write("%s\n" % item)
