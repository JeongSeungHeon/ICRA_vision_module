import numpy as np
import time
import cv2
from realsense import *
from ultralytics import YOLOE
import open3d as o3d
import matplotlib.pyplot as plt
import icp
from utils import *
import trimesh
import copy


def reset_realsense():
    print("Finding RealSense devices...")
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices found.")
        return
        
    for dev in devices:
        print(f"Sending hardware reset to: {dev.get_info(rs.camera_info.name)}")
        dev.hardware_reset()
        
    print("Reset command sent. Wait 3-5 seconds for the camera to reboot before running your main script!")

def viz_seg_result(mask, color):
    # Make a copy of the original frame to draw on
    display_frame = color.copy()
    
    # 1. Resize the mask to match your camera resolution
    # YOLO often processes images at 640x640, so the raw mask might be the wrong size
    mask = cv2.resize(mask, (color.shape[1], color.shape[0]))
    
    # 2. Create an empty image the exact same size as your camera frame
    colored_overlay = np.zeros_like(color)
    
    # 3. Paint the mask onto the overlay (BGR format)
    # Where the mask is greater than 0.5 (active), paint it Green [0, 255, 0]
    colored_overlay[mask > 0.5] = [0, 255, 0] 
    
    # 4. Alpha Blending! 
    # This mixes the original frame with the colored overlay.
    # The '0.5' is the transparency (alpha). 1.0 is solid green, 0.1 is very faint.
    alpha = 0.5
    cv2.addWeighted(colored_overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

    return display_frame

def make_pcd(intrinsics, color, depth, mask, r2c):
    
    mask = cv2.resize(mask, (color.shape[1], color.shape[0]))
    mask = mask > 0.5
    h, w, _ = color.shape

    masked_depth = depth.copy()
    masked_depth[~mask] = 0.0

    u, v = np.meshgrid(np.arange(w),np.arange(h))
    u = u[mask]
    v = v[mask]
    z = depth[mask]
    # z = masked_depth

    x = (u-intrinsics[0,2]) * z / intrinsics[0,0]
    y = (v-intrinsics[1,2]) * z / intrinsics[1,1]

    points = np.dstack((x,y,z)).squeeze()

    colors = color[mask]

    colors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))

    aug = np.ones((len(points),1))
    pcd_cam = np.hstack((points, aug))
    pcd_robot = (np.linalg.inv(r2c) @ pcd_cam.T).T
    pcd_robot = pcd_robot[:,:3]

    return pcd_robot, colors

def run_icp(template, pcd):

    obj_mean = np.median(pcd, axis=0)

    template_mean = np.median(template, axis=0)

    # Create your 4x4 matrix, but only shift the Z axis!
    init_pose = np.eye(4)
    init_pose[:3,3] = obj_mean-template_mean

    total_time = 0

    # Run ICP
    start = time.time()
    T = icp.icp(template, pcd, init_pose, tolerance=0.000001)
    total_time += time.time() - start

    # Make C a homogeneous representation of B
    C = np.ones((len(template), 4))
    C[:,0:3] = np.copy(template)

    # Transform C
    C = np.dot(T, C.T).T

    np.save("transformed_pt", C)

    print("time:", total_time)

    # assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
    # assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)     # T and R should be inverses
    # assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma)        # T and t should be inverses


    return C

def match_bounding_box_scale(source_pcd, source_bbox, target_bbox):
    """
    Scales and moves a source point cloud to perfectly fit inside a target bounding box.
    Works for both Axis-Aligned (AABB) and Oriented (OBB) bounding boxes, 
    as long as they are already facing the same direction!
    """
    # Create a copy so we don't destroy the original
    scaled_pcd = copy.deepcopy(source_pcd)
    
    # 1. Get the physical sizes (extents) of both boxes
    extent_source = source_bbox.extent
    extent_target = target_bbox.extent

    # 2. Calculate the exact scale factor for X, Y, and Z
    s_x = extent_target[0] / extent_source[0]
    s_y = extent_target[1] / extent_source[1]
    s_z = extent_target[2] / extent_source[2]

    
    # print(f"Scaling by: X={s_x:.2f}, Y={s_y:.2f}, Z={s_z:.2f}")

    # 3. Create the 4x4 Non-Uniform Scaling Matrix
    # This matrix only scales; it does not rotate or translate.
    scale_matrix = np.array([
        [s_z, 0,   0,   0],
        [0,   s_y, 0,   0],
        [0,   0,   s_x, 0],
        [0,   0,   0,   1]
    ])

    # --- THE EXECUTION ---
    
    # Step A: Move source to the 0,0,0 origin (so it scales from its own center)
    scaled_pcd.translate(-source_bbox.center)
    
    # Step B: Apply the matrix to stretch the object
    scaled_pcd.transform(scale_matrix)
    
    # Step C: Move the newly scaled object to the target's location
    scaled_pcd.translate(target_bbox.center)
    
    # Optional: Generate the new bounding box to verify it matches!
    new_bbox = scaled_pcd.get_oriented_bounding_box()
    new_bbox.color = (1, 0, 0) # Make it red to confirm it matches the target
    
    return scaled_pcd, new_bbox

def filter_pcd(points, target_num=3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    print("Running DBSCAN clustering...")
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))

    # Safely extract only the valid cluster IDs (ignores -1 noise)
    valid_cluster_ids = np.unique(labels[labels >= 0])
    num_clusters = len(valid_cluster_ids)
    
    print(f"Found {num_clusters} valid clusters.")

    # 1. Check if we already have the target amount (or fewer)
    if num_clusters <= target_num:
        print(f"Already at {num_clusters} clusters. Removing noise and keeping everything else.")
        keep_ids = valid_cluster_ids
        
    else:
        # 2. Calculate mean X for every valid cluster
        # We loop over valid_cluster_ids just in case DBSCAN skips a number
        mean_x_values = np.array([np.mean(points[labels == i, 0]) for i in valid_cluster_ids])
        
        # 3. Sort by Mean X (Lowest to Highest)
        # argsort gives us the exact order of the cluster IDs
        sorted_indices = np.argsort(mean_x_values)
        
        # 4. Grab the "Winners"
        # Slicing with [-target_num:] grabs the last N items (the Highest X values).
        # By doing this, we naturally throw away all the lowest X values!
        keep_indices = sorted_indices[-target_num:]
        keep_ids = valid_cluster_ids[keep_indices]
        
        print(f"Keeping clusters {keep_ids} (Highest X). Removed {num_clusters - target_num} clusters.")

    # ==========================================
    # The Magic Filter: np.isin()
    # ==========================================
    # np.isin() instantly checks every single pixel's label. 
    # If the label is in our 'keep_ids' list, it returns True. Otherwise, False.
    mask = np.isin(labels, keep_ids)
    
    # Extract only the points that survived the mask
    final_objects = pcd.select_by_index(np.where(mask)[0])

    # ==========================================
    # Optional: Clean up the Colors!
    # ==========================================
    # Because we deleted clusters, our IDs might be messy (e.g., [2, 5, 8]).
    # We map them back to [0, 1, 2] so the colormap looks pretty.
    kept_labels = labels[mask]
    color_map = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}
    remapped_labels = np.array([color_map[lbl] for lbl in kept_labels])
    
    colors = plt.get_cmap("tab10")(remapped_labels / max(1, len(keep_ids) - 1))
    final_objects.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Return the cleaned NumPy array
    return np.asarray(final_objects.points)

def main():

    device_serial = get_devices()
        
    resolution = (640, 480)

    camera = RealSense(device_serial[0], resolution, resolution)

    camera.start()

    time.sleep(1)

    intrinsics = camera.get_intrinsics_matrix();
    cmtx = intrinsics[0]
    depth_scale = camera.get_depth_scale()

    model = YOLOE("./yoloe-26m-seg.pt")
    model.set_classes(["cup"])

    r2c = np.load('./assets/robot2camera.npy')

    vis = o3d.visualization.Visualizer()
    vis.create_window("Real-Time 3D Cup", width=800, height=600)
    
    # Create an empty point cloud container
    obj_pcd = o3d.geometry.PointCloud()
    
    is_first_frame = True


    mesh_path = '../cup_small.stl'
    mesh = trimesh.load(mesh_path)
    mesh = align_mesh_to_coordinate(mesh)
    mesh = reset_mesh(mesh)

    mesh_pcd = mesh.sample_points_uniformly(number_of_points=1000)
    mesh_pcd_clean, _ = mesh_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    obb_mesh = mesh_pcd_clean.get_oriented_bounding_box()
    obb_mesh.color = (0, 0, 1)
    mesh_pcd_clean.paint_uniform_color([0.0, 0.0, 0.0])
        
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.02, 
    origin=[0,0,0]
    )
    # mesh_frame.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    vis.add_geometry(mesh_frame)

    display_obj_pcd = o3d.geometry.PointCloud()

    cnt = 0
    while True:

        color, depth = camera.shoot()
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = depth * depth_scale
        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth)

        # cv2.imshow("color",color)
        # cv2.imshow("depth",depth)

        # segmentation
        results = model.predict(color)  # Predict on an image
        result = results[0]

        # Check if the model actually found any masks in this frame
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            mask = masks[0]
            mask_viz = viz_seg_result(mask, color)

            obj_points, colors = make_pcd(cmtx, color, depth, mask, r2c)

            obj_points = filter_pcd(obj_points)

            # object pcd
            obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
            obj_pcd_clean, ind = obj_pcd.remove_statistical_outlier(nb_neighbors=int(obj_points.shape[0] * 0.01), std_ratio=1.0)

            # obj_pcd_clean.translate(-obj_pcd_clean.get_oriented_bounding_box().center)
            display_obj_pcd.points = o3d.utility.Vector3dVector(np.asarray(obj_pcd_clean.points))
            display_obj_pcd.paint_uniform_color([0.5, 0.5, 0.5])

            obb_pcd_clean = obj_pcd_clean.get_oriented_bounding_box()
            obb_pcd_clean.color = (0, 1, 0)

            obj_pcd_clean.rotate(obb_mesh.R @ obb_pcd_clean.R.T, center=obb_pcd_clean.center)


            if is_first_frame:
                scaled_template, _ = match_bounding_box_scale(mesh_pcd_clean, obb_mesh, obb_pcd_clean)
                # obb_mesh = scaled_template.get_oriented_bounding_box()
                vis.add_geometry(display_obj_pcd)
                # vis.add_geometry(obb_pcd_clean)
                # vis.add_geometry(obb_mesh)
                vis.add_geometry(mesh_pcd_clean)
                vis.add_geometry(scaled_template)
                is_first_frame = False
            else:

                filtered_poitns = np.asarray(obj_pcd.points)
                scaled_template_pt = np.asarray(scaled_template.points)
                
                # Run your custom ICP
                transformed = run_icp(scaled_template_pt, filtered_poitns)

                # Update the scaled template's Open3D geometry for visualization
                transformed = np.array(transformed)[:,:3]
                scaled_template.points = o3d.utility.Vector3dVector(transformed.astype(np.float64))
                
                vis.update_geometry(display_obj_pcd)
                vis.update_geometry(scaled_template)
        cv2.imshow("Custom OpenCV Transparent Mask", mask_viz)


    # ==========================================
        # 3. REFRESH THE 3D VIEWER
        # ==========================================
        # This is the magic that keeps the window from freezing!
        if not vis.poll_events():
            break # If you click the 'X' on the 3D window, break the loop
        vis.update_renderer()
        # if cnt%100 == 0:
        #     points, colors = make_pcd(cmtx, color, depth, mask)

        key = cv2.waitKey(1) & 0xFF  # process window events

        if key == ord('q'):
            print("Quit")
            camera.stop()
            break

        cnt+=1
    
    reset_realsense()

if __name__ == "__main__":
    main()