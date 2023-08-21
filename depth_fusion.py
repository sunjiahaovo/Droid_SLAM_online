import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import taichi as ti
import taichi.math as tm
ti.init(arch=ti.gpu, device_memory_GB=10.0)



@ti.data_oriented
class DepthFusion:
    def __init__(self, dataset='replica', downsample=1):
        images = np.load(f'reconstructions/{dataset}/images.npy')[::downsample]
        # normalize to [0,1] for o3d visualization
        # images = images.transpose(0,2,3,1) / 255.0
        images_up = images.transpose(0,2,3,1) / 255.0
        images = images_up[:,3::8,3::8, :]
        self.num = images.shape[0]
        self.height, self.width = images.shape[1], images.shape[2]
        self.images = ti.Vector.field(3, dtype=ti.f32, shape=(self.num, self.height, self.width))
        self.images.from_numpy(images.astype(np.float32))
        
        # make intrinsics match the disp size
        intrinsics = np.load(f'reconstructions/{dataset}/intrinsics.npy')[::downsample]
        # 4 means fx fy cx cy
        self.intrinsics = ti.Vector.field(4, dtype=ti.f32, shape=self.num)
        self.intrinsics.from_numpy(intrinsics.astype(np.float32))
  
        quat_poses = np.load(f'reconstructions/{dataset}/poses.npy')[::downsample]
        # 7 means qx qy qz qw tx ty tz
        self.quat_poses = ti.Vector.field(7, dtype=ti.f32, shape=self.num)
        self.quat_poses.from_numpy(quat_poses.astype(np.float32))    
        
        disps = np.load(f'reconstructions/{dataset}/disps.npy')[::downsample]
        self.masks = disps > .5*disps.mean(axis=(1,2), keepdims=True)
        self.disps = ti.field(dtype=ti.f32, shape=(self.num, self.height, self.width))
        self.disps.from_numpy(disps.astype(np.float32))

        self.points = ti.Vector.field(3, dtype=ti.f32, shape=(self.num, self.height, self.width))
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=(self.num, self.height, self.width))
        self.counter = ti.field(dtype=ti.i32, shape=(self.num, self.height, self.width))
       

    def create_point_actor(self, points, clrs, mask):
        """ open3d point cloud from numpy array """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points[mask])
        point_cloud.colors = o3d.utility.Vector3dVector(clrs[mask])
        return point_cloud


    @ti.func
    def actSO3(self, q: ti.template(), X: ti.template(), Y: ti.template()):
        uv = ti.Vector.zero(ti.f32, 3)
        uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1])
        uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2])
        uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0])

        Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1])
        Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2])
        Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0])
    

    @ti.func
    def actSE3(self, t:ti.template(), q: ti.template(), X: ti.template(), Y: ti.template()):
        self.actSO3(q, X, Y)
        Y[3] = X[3]
        Y[0] += X[3] * t[0]
        Y[1] += X[3] * t[1]
        Y[2] += X[3] * t[2]
    
    @ti.func
    def relSE3(self, t_i:ti.template(), q_i:ti.template(), t_j:ti.template(), q_j:ti.template(), tij:ti.template(), qij:ti.template()):
        qij[0] = -q_j[3] * q_i[0] + q_j[0] * q_i[3] - q_j[1] * q_i[2] + q_j[2] * q_i[1]
        qij[1] = -q_j[3] * q_i[1] + q_j[1] * q_i[3] - q_j[2] * q_i[0] + q_j[0] * q_i[2]
        qij[2] = -q_j[3] * q_i[2] + q_j[2] * q_i[3] - q_j[0] * q_i[1] + q_j[1] * q_i[0]
        qij[3] =  q_j[3] * q_i[3] + q_j[0] * q_i[0] + q_j[1] * q_i[1] + q_j[2] * q_i[2]
  
        self.actSO3(qij, t_i, tij)
        tij[0] = t_j[0] - tij[0]
        tij[1] = t_j[1] - tij[1]
        tij[2] = t_j[2] - tij[2]
    

    @ti.func
    def invSE3(self, t:ti.template(), q: ti.template()):
        q_inv = ti.Vector([-q[0], -q[1], -q[2], q[3]], dt=ti.f32)
        t_inv = ti.Vector.zero(dt=ti.f32, n=3)
        self.actSO3(q_inv, t, t_inv)
        return -t_inv, q_inv


    @ti.kernel
    def depth_filter(self, threshold: ti.f32):
        ti.loop_config(block_dim=256)
        for I in ti.grouped(self.disps):
            idx = I.x
            pose_i = self.quat_poses[idx]
            intrinsic_i = self.intrinsics[idx]
            t2cam_i = ti.Vector([pose_i[0], pose_i[1], pose_i[2]], dt=ti.f32)
            q2cam_i = ti.Vector([pose_i[3], pose_i[4], pose_i[5], pose_i[6]], dt=ti.f32)

            u = I.z
            v = I.y
            ui = (float(u) - intrinsic_i[2]) / intrinsic_i[0]
            vi = (float(v) - intrinsic_i[3]) / intrinsic_i[1]
            disp = self.disps[idx, v, u]
            xi = ti.Vector([ui, vi, 1, disp], dt=ti.f32)
            for neighbor_id in range(6):
                jdx = idx - neighbor_id - 1 if neighbor_id < 3 else idx + neighbor_id - 2
                if jdx < 0 or jdx >= self.num:
                    continue
                pose_j = self.quat_poses[jdx]
                intrinsic_j = self.intrinsics[jdx]
                t2cam_j = ti.Vector([pose_j[0], pose_j[1], pose_j[2]], dt=ti.f32)
                q2cam_j = ti.Vector([pose_j[3], pose_j[4], pose_j[5], pose_j[6]], dt=ti.f32)
                t_ij = ti.Vector.zero(dt=ti.f32, n=3)
                q_ij = ti.Vector.zero(dt=ti.f32, n=4)
                self.relSE3(t2cam_i, q2cam_i, t2cam_j, q2cam_j, t_ij, q_ij)
                xj = ti.Vector.zero(dt=ti.f32, n=4)
                self.actSE3(t_ij, q_ij, xi, xj)
                uj = xj[0] / xj[2] * intrinsic_j[0] + intrinsic_j[2]
                vj = xj[1] / xj[2] * intrinsic_j[1] + intrinsic_j[3]
                dj = xj[3] / xj[2]

                u0 = ti.cast(ti.floor(uj), ti.i32)
                v0 = ti.cast(ti.floor(vj), ti.i32)

                if u0 < 0 or u0 >= self.images.shape[2] or v0 < 0 or v0 >= self.images.shape[1]:
                    continue
                wx = ti.ceil(uj) - uj
                wy = ti.ceil(vj) - vj
                d00 = self.disps[jdx, v0, u0]
                d01 = self.disps[jdx, v0, u0 + 1]
                d10 = self.disps[jdx, v0 + 1, u0]
                d11 = self.disps[jdx, v0 + 1, u0 + 1]

                if abs(1 / d00 - 1 / dj) < threshold :
                    ti.atomic_add(self.counter[idx, v, u], 1)  
                elif abs(1 / d01 - 1 / dj) < threshold :
                    ti.atomic_add(self.counter[idx, v, u], 1)
                elif abs(1 / d10 - 1 / dj) < threshold :
                    ti.atomic_add(self.counter[idx, v, u], 1)
                elif abs(1 / d11 - 1 / dj) < threshold :
                    ti.atomic_add(self.counter[idx, v, u], 1) 





    @ti.kernel
    def iproj(self):
        
        ti.loop_config(block_dim=256)
        for I in ti.grouped(self.disps):
            idx = I.x
            pose = self.quat_poses[idx]
            intrinsic = self.intrinsics[idx]
            t2cam = ti.Vector([pose[0], pose[1], pose[2]], dt=ti.f32)
            q2cam = ti.Vector([pose[3], pose[4], pose[5], pose[6]], dt=ti.f32)
            t2base, q2base = self.invSE3(t2cam, q2cam)
         

            u = I.z
            v = I.y
            ui = (float(u) - intrinsic[2]) / intrinsic[0]
            vi = (float(v) - intrinsic[3]) / intrinsic[1]

            if u < 0 or u >= self.images.shape[2] or v < 0 or v >= self.images.shape[1]:     
                continue
            disp = self.disps[idx, v, u]
            point_cam = ti.Vector([ui, vi, 1, disp])
            point_base = ti.Vector.zero(dt=ti.f32, n=4)
            self.actSE3(t2base, q2base, point_cam, point_base)
            # BGR -> RGB
            self.colors[idx, v, u] = ti.Vector([self.images[idx, v, u].b, self.images[idx, v, u].g, self.images[idx, v, u].r], dt=ti.f32)
            self.points[idx, v, u] = point_base[:3] / point_base[3]

    def depth_fusion(self):

        self.iproj()    
        ### 目前感觉还有问题，需要再调试一下 ###
        # self.depth_filter(0.25)
        clrs = self.colors.to_numpy()
        pts = self.points.to_numpy()
        # cnts = self.counter.to_numpy()
        # self.masks = (cnts >= 2) & self.masks

        clrs = clrs.reshape(-1, 3)
        pts = pts.reshape(-1, 3)
       
        ### create Open3D visualization ###
        point_cloud = self.create_point_actor(pts.reshape(-1, 3), clrs.reshape(-1, 3), self.masks.reshape(-1))
        # o3d.io.write_point_cloud("filtered_point_cloud.ply", point_cloud)
        # o3d.io.write_point_cloud("nofiltered_point_cloud.ply", point_cloud)

        ### 体素滤波 ###
        point_cloud_voxel_filter = point_cloud.voxel_down_sample(voxel_size=0.05)
        ## 统计滤波 ###
        point_cloud_statistical_filter, ind = point_cloud_voxel_filter.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.05)
        ### 半径滤波 ###
        # point_cloud_radius_filter, ind = point_cloud.remove_radius_outlier(nb_points=100, radius=0.3)

        o3d.io.write_point_cloud("filtered_point_cloud.ply", point_cloud_statistical_filter)

        o3d.visualization.draw_geometries([point_cloud_statistical_filter], window_name='point_cloud', width=960, height=540)





if __name__ == '__main__':

    # tstamps = np.load('/home/perple/Public/DROID-SLAM/reconstructions/d435_sample/tstamps.npy')
    # print(tstamps.shape)
    # print(tstamps)
    fusion = DepthFusion(dataset='d435_sample_depth',downsample=4)
    # # fusion = DepthFusion(dataset='replica',downsample=4)

    fusion.depth_fusion()









    







