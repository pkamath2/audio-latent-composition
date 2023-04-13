import numpy as np
import os

def get_prototype(w_arr, w_avg):
    X_w = (w_arr - w_avg).T
    U, S, VT = np.linalg.svd(X_w, full_matrices=0)
    w_prototype = np.mean(w_arr, axis=0) - w_avg
    w_prototype = (w_avg + (U[:,:1] @ U[:,:1].T @ w_prototype.T)).T
    return w_prototype

def get_direction(w1_arr, w2_arr, w_avg, name=None, save=False):
    w1_prototype = get_prototype(w1_arr, w_avg)
    w2_prototype = get_prototype(w2_arr, w_avg)
    
    direction = w1_prototype - w2_prototype
    direction_mag = np.linalg.norm(direction)
    
    if save:
        os.makedirs('direction_vectors', exist_ok=True)
        assert name is not None, "Parameter 'name' should not be None while saving vectors."
        np.save('direction_vectors/%s.npy' % name, direction)
        np.save('direction_vectors/%sprototype-1.npy' % name, w1_prototype)
        np.save('direction_vectors/%sprototype-2.npy' % name, w2_prototype)
    return direction, w1_prototype, w2_prototype

def get_direction_localexploration(direction_np, dim=128): 
    random_vectors = np.vstack([direction_np, np.random.randn(dim-1, dim)])
    q, r = np.linalg.qr(random_vectors)
    q = q.T
    print('Are the vectors orthonormal? : ',np.allclose(q@q.T, np.eye(128,128)))
    return q


def transfer_attribute(sample, ref_sample, prototype, direction, mult=1.0):
    sample_direction = prototype - sample
    sample_proj = (sample_direction @ direction)[0]
    ref_sample_direction = (prototype - ref_sample)
    ref_sample_proj = (ref_sample_direction @ direction)[0]
    
    sample_modified = sample + ((sample_proj-ref_sample_proj) * direction * mult)
    return sample_modified
