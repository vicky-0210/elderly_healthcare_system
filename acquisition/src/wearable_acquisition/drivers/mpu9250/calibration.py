"""Fit a per-device six-face accelerometer affine and stationary gyro bias."""
import numpy as np


def fit_calibration(faces,stationary):
    poses=np.asarray(faces,dtype=float); rest=np.asarray(stationary,dtype=float)
    if poses.ndim!=3 or poses.shape[0]!=6 or poses.shape[2]!=3 or poses.shape[1]<100:
        raise ValueError("Supply >=100 stationary accelerometer samples per labeled face (+X,-X,+Y,-Y,+Z,-Z)")
    if rest.ndim!=2 or rest.shape[1]!=3 or len(rest)<100 or not np.isfinite(rest).all() or not np.isfinite(poses).all():
        raise ValueError("Supply finite stationary gyro and face measurements")
    if poses.std(1).max()>.04 or rest.std(0).max()>1.5: raise ValueError("Calibration capture is moving or noisy")
    means=poses.mean(1)
    design=np.column_stack((means,np.ones(6)))
    target=np.repeat(np.eye(3),2,axis=0)*np.tile([1,-1],3)[:,None]
    transform,_,rank,_=np.linalg.lstsq(design,target,rcond=None)
    error=np.linalg.norm(design@transform-target,axis=1).max()
    if rank!=4 or error>.08 or np.linalg.cond(transform[:3])>1.5 or np.max(np.abs(transform[:3]-np.eye(3)))>.30:
        raise ValueError("Misidentified faces, excessive cross-axis error or scale")
    return dict(acc_transform=transform[:3].tolist(),acc_offset=transform[3].tolist(),
        gyro_bias=rest.mean(0).tolist(),max_face_residual_g=float(error))


def apply(physical,cal):
    x=np.asarray(physical,dtype=float).copy()
    x[:,:3]=x[:,:3]@np.asarray(cal["acc_transform"])+cal["acc_offset"]
    x[:,3:]-=cal["gyro_bias"]
    return x
