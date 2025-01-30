import numpy as np
import torch
import torch.nn as nn
import trimesh
import open3d as o3d
import math

from hair_renderer import *

class RenderDepth(nn.Module):
    def __init__(self, hair_path, camera_path, width=256, deg=0):
        super(RenderDepth, self).__init__()
        self.body_path = './data/half_body.obj'
        self.hair_path = hair_path

        self.body_vertices, self.body_faces, self.num_bodyV = self.load_body()
        #self.hair_vertices, self.hair_faces, self.num_hairV = self.load_hair()
        self.hair_vertices, self.hair_faces, self.num_hairV, self.hair_lines = self.load_hair_raw()
        self.faces = self.combine_faces().unsqueeze(0)
        
        self.camera = self.get_camera(camera_path)
        self.width = width
        self.world_to_view_point = np.eye(4)
        self.set_world_view_point(deg)

        self.normalizer = nn.functional.normalize

    def load_body(self):
        body_mesh = trimesh.load(self.body_path)
        num_bodyV = body_mesh.vertices.shape[0]
        body_vertices = torch.from_numpy(body_mesh.vertices.astype(np.float32)).cuda()
        body_faces = torch.from_numpy(body_mesh.faces.astype(np.int32)).cuda()
        return body_vertices, body_faces, num_bodyV

    def load_hair(self):
        hair_mesh = o3d.io.read_line_set(self.hair_path)
        hair_vertices = np.array(hair_mesh.points)
        num_hairV = hair_vertices.shape[0]
        hair_lines = np.array(hair_mesh.lines)

        hair_faces = np.zeros((hair_lines.shape[0],3))
        hair_faces[:,:2] = hair_lines
        hair_faces[:,2] = hair_lines[:,1]#+1

        hair_vertices = torch.from_numpy(hair_vertices.astype(np.float32)).cuda()
        hair_faces = torch.from_numpy(hair_faces.astype(np.int32)).cuda()

        return hair_vertices, hair_faces, num_hairV
    
    def load_hair_raw(self):
        hair_mesh = o3d.io.read_line_set(self.hair_path)
        hair_vertices = np.array(hair_mesh.points)
        num_hairV = hair_vertices.shape[0]
        hair_lines = np.array(hair_mesh.lines)
        num_hairL = hair_lines.shape[0]

        hair_faces = np.zeros((hair_lines.shape[0]*2,3))
        hair_faces[:num_hairL, :2] = hair_lines
        hair_faces[:num_hairL, 2] = hair_lines[:,1] + num_hairV

        hair_faces[num_hairL:, :2] = hair_lines + num_hairV
        hair_faces[num_hairL:, 2] = hair_lines[:,0]

        # hair_faces[num_hairL:, 0] = hair_lines[:,1] + num_hairV
        # hair_faces[num_hairL:, 1] = hair_lines[:,0] + num_hairV
        # hair_faces[num_hairL:, 2] = hair_lines[:,0]

        hair_vertices = torch.from_numpy(hair_vertices.astype(np.float32)).cuda()
        hair_lines = torch.from_numpy(hair_lines).cuda().long()
        hair_faces = torch.from_numpy(hair_faces.astype(np.int32)).cuda()

        return hair_vertices, hair_faces, num_hairV*2, hair_lines

    def save_strands(self, outputpath):
        vertices = self.hair_vertices
        lines = self.hair_lines

        vertices = vertices.detach().cpu().numpy()
        lines = lines.detach().cpu().numpy()

        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.asarray(vertices)), lines=o3d.utility.Vector2iVector(lines))
        o3d.io.write_line_set(outputpath, line_set)

    # def process_hair_vertices(self, hair_vertices, dis=2e-3):
    #     hair_vertices_dis = hair_vertices + dis
    #     return torch.cat((hair_vertices, hair_vertices_dis), 0)
    def process_hair_vertices(self, hair_vertices, dis=2e-3):
        hair_vertices_dis = hair_vertices + dis
        hair_vertices_x = hair_vertices[:,0]
        hair_vertices_dis[:,2] = hair_vertices[:,2] # recover z
        hair_vertices_dis_x = torch.where(hair_vertices_x<0,
                                    hair_vertices_dis[:,0]-2*dis,
                                    hair_vertices_dis[:,0]) # move another direction when x<0
        hair_vertices_dis = torch.cat([hair_vertices_dis_x.unsqueeze(1),hair_vertices_dis[:,1:]],dim=1)
        return torch.cat((hair_vertices, hair_vertices_dis), 1)

    def combine_faces(self):
        return torch.cat((self.hair_faces, self.body_faces + self.num_hairV), 0)

    def get_camera(self, param_path, loadSize=1024):
        # loading calibration data
        param = np.load(param_path, allow_pickle=True)
        # pixel unit / world unit
        ortho_ratio = param.item().get('ortho_ratio')
        # world unit / model unit
        scale = param.item().get('scale')
        # camera center world coordinate
        center = param.item().get('center')
        # model rotation
        R = param.item().get('R')

        #print(param)

        translate = -np.matmul(R, center).reshape(3, 1)
        extrinsic = np.concatenate([R, translate], axis=1)
        extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
        # Match camera space to image pixel space
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = scale / ortho_ratio
        scale_intrinsic[1, 1] = -scale / ortho_ratio
        scale_intrinsic[2, 2] = scale / ortho_ratio
        # Match image pixel space to image uv space
        uv_intrinsic = np.identity(4)
        uv_intrinsic[0, 0] = 1.0 / float(loadSize // 2)
        uv_intrinsic[1, 1] = 1.0 / float(loadSize // 2)
        uv_intrinsic[2, 2] = 1.0 / float(loadSize // 2)
        # Transform under image pixel space
        trans_intrinsic = np.identity(4)

        intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic).astype(np.float32)).unsqueeze(0).cuda()

        return calib

    def world2uv(self, verts):
        world_to_view_point = torch.from_numpy(self.world_to_view_point.astype(np.float32)).unsqueeze(0).cuda()
        verts = verts.unsqueeze(0)

        # world_mat = world_to_view_point[:,:3,:3]
        # world_trans = world_to_view_point[:,:3,3:4]
        
        #verts = torch.baddbmm(world_trans,world_mat,verts.transpose(1,2)).transpose(1,2)

        mat = self.camera[:,:3,:3]
        trans = self.camera[:,:3,3:4]
        #[-1,1]
        verts = torch.baddbmm(trans,mat,verts.transpose(1,2)).transpose(1,2)

        return verts

    def euler_to_rot_mat(self, r_x, r_y, r_z):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(r_x), -math.sin(r_x)],
                        [0, math.sin(r_x), math.cos(r_x)]
                        ])

        R_y = np.array([[math.cos(r_y), 0, math.sin(r_y)],
                        [0, 1, 0],
                        [-math.sin(r_y), 0, math.cos(r_y)]
                        ])

        R_z = np.array([[math.cos(r_z), -math.sin(r_z), 0],
                        [math.sin(r_z), math.cos(r_z), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))

        return R

    def set_world_view_point(self,deg):
        '''setting from world to view homo rotation matrix
        deg: degree to rotation your map
        '''
        rz = deg / 180. * np.pi
        
        self.world_to_view_point[:3, :3] = self.euler_to_rot_mat(0, rz, 0)
    
    def set_texture_mask(self):
        hair_texture = np.array([1.0,1.0,1.0]*self.num_hairV).reshape(self.num_hairV,3)
        body_texture = np.array([0.0,0.0,0.0]*self.num_bodyV).reshape(self.num_bodyV,3)
        
        hair_texture = torch.from_numpy(hair_texture.astype(np.float32)).unsqueeze(0).cuda()
        body_texture = torch.from_numpy(body_texture.astype(np.float32)).unsqueeze(0).cuda() 

        return torch.cat((hair_texture,body_texture), 1)

    def set_texture(self, verts):
        '''attribute you want to set when you render,
        we only need 2d orientation information here
        '''
        verts[:,0] = -verts[:,0]
        #compute orien
        verts = (verts+1)*self.width/2
        hair_orien = np.zeros((int(self.num_hairV/2),3))
        hair_orien = torch.from_numpy(hair_orien.astype(np.float32)).cuda()

        cur_orien = (verts[self.hair_lines[:,1]] - verts[self.hair_lines[:,0]])[:,:2]
        cur_orien = self.normalizer(cur_orien, dim=1)
        
        # orien_len = torch.norm(cur_orien.data,p=2,dim=1)
        # orien_len_ones = torch.ones_like(orien_len).float().cuda()
        # orien_len = torch.where(orien_len>0.001, orien_len, orien_len_ones)
        # cur_orien = (cur_orien.T/orien_len).T

        # if torch.any(torch.isnan(cur_orien)):
        #     print('divide zeros when compute orientation map!')

        ambi_y = cur_orien[:,1]<0
        cur_orien[ambi_y,:] = -cur_orien[ambi_y,:]

        cur_orien[:,0] = cur_orien[:,0]/ 2.0 + 0.5
        
        orien3 = torch.Tensor([1.0]*cur_orien.shape[0]).unsqueeze(1).cuda().float()
        cur_orien = torch.cat((cur_orien, orien3), 1)
           
        hair_orien[self.hair_lines[:,1]] = cur_orien

        #BGR 2 RGB
        hair_orien = hair_orien[:,[2,1,0]]
        body_texture = np.array([0.5,0.0,0.0]*self.num_bodyV).reshape(self.num_bodyV,3)#[0.0,0.0,0.0]
        body_texture = torch.from_numpy(body_texture.astype(np.float32)).cuda() 

        return torch.cat((hair_orien, hair_orien, body_texture), 0).unsqueeze(0)

    def set_texture_depth(self, verts):
        texture = torch.zeros((verts.shape[0],3))
        texture[:,0] = (verts[:,2] +1.0)/2.0
        # texture[:,0] = verts[:,2]
        texture[:,1] = texture[:,0] 
        texture[:,2] = texture[:,0] 

        return texture.float().cuda().unsqueeze(0)

    def set_texture_new(self, verts):
        '''attribute you want to set when you render,
        we only need 2d orientation information here
        '''
        ids = torch.arange(0,verts.shape[0]).unsqueeze(1)
        full_lines = torch.cat((ids,ids),1).long().cuda()
        full_lines[self.hair_lines[:,0]] = self.hair_lines

        #compute orien
        verts = (verts+1)*self.width/2
        hair_orien = np.zeros((int(self.num_hairV/2),3))
        hair_orien = torch.from_numpy(hair_orien.astype(np.float32)).cuda()

        hair_orien = (verts[full_lines[:,1]] - verts[full_lines[:,0]])[:,:2]
        hair_orien = self.normalizer(hair_orien, dim=1)

        hair_orien[:,0] = hair_orien[:,0]/ 2.0 + 0.5
        
        orien3 = torch.Tensor([1.0]*hair_orien.shape[0]).unsqueeze(1).cuda().float()
        hair_orien = torch.cat((hair_orien, orien3), 1)

        #BGR 2 RGB
        hair_orien = hair_orien[:,[2,1,0]]
        body_texture = np.array([0.5,0.0,0.0]*self.num_bodyV).reshape(self.num_bodyV,3)#[0.5,0.0,0.0]
        body_texture = torch.from_numpy(body_texture.astype(np.float32)).cuda() 

        return torch.cat((hair_orien, hair_orien, body_texture), 0).unsqueeze(0)

    # def forward(self):
    #     hair_vertices = self.world2uv(self.hair_vertices)
    #     body_vertices = self.world2uv(self.body_vertices)

    #     vertices = torch.cat((hair_vertices,body_vertices), 1)

    #     textures = self.set_texture_mask()

    #     return Mesh(vertices, self.faces, textures, 1, 'vertex')

    def forward(self):
        # hair_vertices = self.process_hair_vertices(self.hair_vertices, dis=1e-3)
        # hair_vertices = self.world2uv(hair_vertices)
        hair_vertices = self.world2uv(self.hair_vertices)
        hair_vertices = self.process_hair_vertices(hair_vertices, dis=1e-3)
        body_vertices = self.world2uv(self.body_vertices)

        vertices = torch.cat((hair_vertices,body_vertices), 1)

        #textures = self.set_texture_mask()
        
        textures = self.set_texture_depth(vertices.squeeze())

        #mirror
        vertices[:,:,2] = -vertices[:,:,2]
        #vertices[:,:,1] = -vertices[:,:,1]

        # mesh = trimesh.Trimesh(vertices.detach().cpu().numpy()[0],self.faces.detach().cpu().numpy()[0],process=False)
        # with open('./results/save.obj', "w") as fout:
        #     mesh.export(fout, file_type="obj") 

        return Mesh(vertices, self.faces, textures, 1, 'vertex')