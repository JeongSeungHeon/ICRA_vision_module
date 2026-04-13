import numpy as np
import torch
import cv2
import open3d as o3d
try:
  import warp as wp
  wp.init()
except:
  wp = None


if wp is not None:
  @wp.kernel(enable_backward=False)
  def bilateral_filter_depth_kernel(depth:wp.array(dtype=float, ndim=2), out:wp.array(dtype=float, ndim=2), radius:int, zfar:float, sigmaD:float, sigmaR:float):
    h,w = wp.tid()
    H = depth.shape[0]
    W = depth.shape[1]
    if w>=W or h>=H:
      return
    out[h,w] = 0.0
    mean_depth = float(0.0)
    num_valid = int(0)
    for u in range(w-radius, w+radius+1):
      if u<0 or u>=W:
        continue
      for v in range(h-radius, h+radius+1):
        if v<0 or v>=H:
          continue
        cur_depth = depth[v,u]
        if cur_depth>=0.001 and cur_depth<zfar:
          num_valid += 1
          mean_depth += cur_depth
    if num_valid==0:
      return
    mean_depth /= float(num_valid)

    depthCenter = depth[h,w]
    sum_weight = float(0.0)
    sum = float(0.0)
    for u in range(w-radius, w+radius+1):
      if u<0 or u>=W:
        continue
      for v in range(h-radius, h+radius+1):
        if v<0 or v>=H:
          continue
        cur_depth = depth[v,u]
        if cur_depth>=0.001 and cur_depth<zfar and abs(cur_depth-mean_depth)<0.01:
          weight = wp.exp( -float((u-w)*(u-w) + (h-v)*(h-v)) / (2.0*sigmaD*sigmaD) - (depthCenter-cur_depth)*(depthCenter-cur_depth)/(2.0*sigmaR*sigmaR) )
          sum_weight += weight
          sum += weight*cur_depth
    if sum_weight>0 and num_valid>0:
      out[h,w] = sum/sum_weight

  def bilateral_filter_depth(depth, radius=2, zfar=100, sigmaD=2, sigmaR=100000, device='cuda'):
    if isinstance(depth, np.ndarray):
      depth_wp = wp.array(depth, dtype=float, device=device)
    else:
      depth_wp = wp.from_torch(depth)
    out_wp = wp.zeros(depth.shape, dtype=float, device=device)
    wp.launch(kernel=bilateral_filter_depth_kernel, device=device, dim=[depth.shape[0], depth.shape[1]], inputs=[depth_wp, out_wp, radius, zfar, sigmaD, sigmaR])
    depth_out = wp.to_torch(out_wp)

    if isinstance(depth, np.ndarray):
      depth_out = depth_out.data.cpu().numpy()
    return depth_out


  @wp.kernel(enable_backward=False)
  def erode_depth_kernel(depth:wp.array(dtype=float, ndim=2), out:wp.array(dtype=float, ndim=2), radius:int, depth_diff_thres:float, ratio_thres:float, zfar:float):
    h,w = wp.tid()
    H = depth.shape[0]
    W = depth.shape[1]
    if w>=W or h>=H:
      return
    d_ori = depth[h,w]
    if d_ori<0.001 or d_ori>=zfar:
      out[h,w] = 0.0
    bad_cnt = float(0)
    total = float(0)
    for u in range(w-radius, w+radius+1):
      if u<0 or u>=W:
        continue
      for v in range(h-radius, h+radius+1):
        if v<0 or v>=H:
          continue
        cur_depth = depth[v,u]
        total += 1.0
        if cur_depth<0.001 or cur_depth>=zfar or abs(cur_depth-d_ori)>depth_diff_thres:
          bad_cnt += 1.0
    if bad_cnt/total>ratio_thres:
      out[h,w] = 0.0
    else:
      out[h,w] = d_ori


  def erode_depth(depth, radius=2, depth_diff_thres=0.0005, ratio_thres=0.8, zfar=100, device='cuda'):
    depth_wp = wp.from_torch(torch.as_tensor(depth, dtype=torch.float, device=device))
    out_wp = wp.zeros(depth.shape, dtype=float, device=device)
    wp.launch(kernel=erode_depth_kernel, device=device, dim=[depth.shape[0], depth.shape[1]], inputs=[depth_wp, out_wp, radius, depth_diff_thres, ratio_thres, zfar],)
    depth_out = wp.to_torch(out_wp)

    if isinstance(depth, np.ndarray):
      depth_out = depth_out.data.cpu().numpy()
    return depth_out
  

def align_mesh_to_coordinate(mesh, scale=0.001):
    """
    mesh를 x,y,z 좌표계에 정렬시키는 함수

    Args:
        mesh (trimesh.Trimesh): 정렬할 mesh
        scale (float): mesh 크기 조정 스케일 (기본값: 0.1)

    Returns:
        trimesh.Trimesh: 정렬된 mesh
        o3d.geometry.TriangleMesh: 정렬된 Open3D mesh
        o3d.geometry.OrientedBoundingBox: 정렬된 mesh의 bounding box
    """
    # mesh 스케일 조정
    mesh.vertices *= scale

    # Open3D mesh로 변환
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))

    # OBB 계산
    obb = mesh_o3d.get_oriented_bounding_box()
    obb.color = (0, 1, 0)

    # mesh의 중심점 계산
    center = mesh_o3d.get_center()

    # mesh의 중심점을 기준으로 회전
    mesh_o3d.rotate(np.linalg.inv(obb.R), center=center)

    # 회전된 mesh의 vertices를 원본 mesh에도 적용
    # mesh.vertices = np.asarray(mesh_o3d.vertices)

    # 새로운 bounding box 계산
    new_obb = mesh_o3d.get_oriented_bounding_box()
    new_obb.color = (1, 0, 0)

    # return mesh, mesh_o3d, new_obb
    return  mesh

def reset_mesh(mesh=None):

    min_xyz = mesh.vertices.min(axis=0)
    max_xyz = mesh.vertices.max(axis=0)
    model_center = (min_xyz+max_xyz)/2
    # if mesh is not None:
    #     mesh_ori = mesh.copy()
    #     mesh = mesh.copy()
    #     mesh.vertices = mesh.vertices - model_center.reshape(1, 3)

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))

    mesh_o3d.compute_vertex_normals()

    return mesh_o3d