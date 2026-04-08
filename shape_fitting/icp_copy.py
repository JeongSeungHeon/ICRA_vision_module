import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    # assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def best_fit_transform_translation_only(A, B):
    '''
    Calculates the least-squares best-fit translation mapping points A to B.
    Rotation is explicitly fixed to the Identity matrix.
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix
      R: mxm rotation matrix (Identity)
      t: mx1 translation vector
    '''

    m = A.shape[1]

    # 1. Translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # 2. Fix Rotation to the Identity matrix (No rotation)
    R = np.identity(m)

    # 3. Calculate optimal translation (Difference of centroids)
    t = centroid_B - centroid_A

    # 4. Build the homogeneous transformation matrix
    T = np.identity(m + 1)
    T[:m, m] = t  # Insert the translation vector into the right column

    return T, R, t

def best_fit_transform_anisotropic(A, B):
    '''
    Calculates best-fit transform mapping A to B with INDEPENDENT axis scaling.
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix
      R: mxm rotation matrix
      t: mx1 translation vector
      scale: 1xm array containing [scale_x, scale_y, scale_z]
    '''

    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # 1. Rotation Matrix (Same SVD approach)
    H = np.dot(AA.T, BB)
    U, S_svd, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # 2. --- NEW: Non-Uniform Scale (S_x, S_y, S_z) ---
    # Mathematically project the BB points back onto the rotated axes of AA
    B_prime = np.dot(R.T, BB.T)  # Shape: (m, N)
    
    scale = np.zeros(m)
    for i in range(m):
        # Least squares optimal scale for each independent axis
        numerator = np.sum(AA[:, i] * B_prime[i, :])
        denominator = np.sum(AA[:, i] ** 2)
        
        # Prevent division by zero if an axis is perfectly flat
        scale[i] = numerator / denominator if denominator > 1e-8 else 1.0

    # 🚨 THE "HEIGHT AND WIDTH" LOCK 🚨
    # If your data is 3D (X, Y, Z), but you ONLY want to optimize width (X) 
    # and height (Y), you must force the depth (Z) scale to stay at 1.0.
    # Uncomment the two lines below if this is what you meant!
    #
    # if m == 3:
    #     scale[2] = 1.0  # Force Z-axis scale to remain unchanged

    # Create a diagonal matrix from our scale values
    S_mat = np.diag(scale)

    # 3. --- UPDATED: Translation ---
    # Apply the scale matrix to centroid_A before rotating
    scaled_A_centroid = np.dot(S_mat, centroid_A.T)
    t = centroid_B.T - np.dot(R, scaled_A_centroid)

    # 4. --- UPDATED: Homogeneous transformation ---
    T = np.identity(m+1)
    # The affine transformation block is Rotation * Scale
    T[:m, :m] = np.dot(R, S_mat) 
    T[:m, m] = t

    return T, R, t, scale

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def nearest_neighbor_strict_1to1(src, dst):
    '''
    Forces strict 1:1 mutually unique pairs. 
    Returns the specific indices of the source and destination points that form unique pairs.
    '''
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    
    distances = distances.ravel()
    indices = indices.ravel()

    # Sort all matches from shortest distance to longest distance
    sorted_order = np.argsort(distances)
    
    seen_dst = set()
    valid_src_indices = []
    valid_dst_indices = []

    # Greedy assignment: The closest pairs get locked in first.
    # If a target point is already claimed, the next source point is out of luck.
    for src_idx in sorted_order:
        dst_idx = indices[src_idx]
        if dst_idx not in seen_dst:
            seen_dst.add(dst_idx)
            valid_src_indices.append(src_idx)
            valid_dst_indices.append(dst_idx)
 
    return valid_src_indices, valid_dst_indices


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    Outputs:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
        final_scale: The overall scale factor applied to A
    '''

    m = A.shape[1]

    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # Notice the 4 unpacked variables here
        T, _, _ = best_fit_transform_translation_only(src[:m,:].T, dst[:m,indices].T)

        src = np.dot(T, src)

        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Get the final overall transformation from original A to the final scaled/moved src
    T, R, t = best_fit_transform_translation_only(A, src[:m,:].T)

    return T