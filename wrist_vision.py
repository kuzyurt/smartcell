"""RGB wrist perception using calibrated rays, never simulator object state."""
import cv2
import numpy as np
import mujoco


class WristCalibration:
    """Camera extrinsics computed from robot joints and fixed mount calibration."""
    def __init__(self, model, mount):
        self.model = model
        self.data = mujoco.MjData(model)
        self.arm = [int(model.joint(n).qposadr[0]) for n in
                    ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')]
        self.grip = int(model.joint('gripper').qposadr[0])
        self.body = int(model.body('camera_mount').id)
        self.pos = mount.stock_pos.copy()
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, mount.stock_quat)
        self.rot = mat.reshape(3,3)
        self.fovy = mount.stock_fovy

    def pose(self, joints, grip):
        self.data.qpos[self.arm] = joints
        self.data.qpos[self.grip] = grip
        mujoco.mj_kinematics(self.model,self.data)
        rot = self.data.xmat[self.body].reshape(3,3)
        return self.data.xpos[self.body] + rot @ self.pos, rot @ self.rot


def plane_points(uv, camera, rotation, fovy, z, size=128):
    """Back-project pixel centres onto a known horizontal work surface."""
    uv = np.asarray(uv, dtype=float)
    f = .5 * size / np.tan(np.radians(fovy) / 2)
    ray = np.stack(((uv[..., 0] + .5 - size/2)/f,
                    -(uv[..., 1] + .5 - size/2)/f,
                    -np.ones(uv.shape[:-1])), axis=-1) @ np.asarray(rotation).T
    scale = (z - np.asarray(camera)[2]) / ray[..., 2]
    return np.asarray(camera) + ray * scale[..., None]


def cube_top(rgb, camera, rotation, fovy, belt_y=.245, top_z=.657):
    """Find the coloured part's bright top face in the conveyor work area."""
    a = np.asarray(rgb, dtype=float)
    red, green, blue = a.transpose(2,0,1)
    mask = ((red > 65) & (green > 25) & (red > green*1.05)
            & (red > blue*.95) & (blue > green*.40) & (blue < green*2.5))
    yy,xx = np.indices(mask.shape)
    xyz = plane_points(np.stack((xx,yy),axis=-1),camera,rotation,fovy,top_z)
    mask &= (np.abs(xyz[...,1]-belt_y)<.05) & (xyz[...,0]>-.3) & (xyz[...,0]<.5)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    candidates=[]
    for k in range(1,n):
        if stats[k,cv2.CC_STAT_AREA]<12:
            continue
        y,x=np.where(labels==k)
        points=xyz[y,x,:2].astype(np.float32)
        rect=cv2.minAreaRect(points)
        dims=np.array(rect[1])
        if np.any(dims < .012) or np.any(dims > .05):
            continue
        candidates.append(dict(xy=np.array(rect[0]), dims=dims,
                               area=len(x), angle=float(rect[2]),
                               uv=np.array([x.mean(),y.mean()])))
    return max(candidates,key=lambda c:c['area']) if candidates else None


def tote_pose(rgb,camera,rotation,fovy,prior=None):
    """Fit a known-size hollow tote silhouette to RGB pixels (centre unknown)."""
    rgb=np.asarray(rgb,dtype=float)
    r,g,b=rgb.transpose(2,0,1)
    observed=(b>r*1.3)&(b>g*1.05)&(b>30)
    # The gripper and held part occlude the tote; do not penalize those pixels.
    valid=~((r>g*1.2)&(r>b*1.15))
    if observed.sum()<45:
        return None
    f=64/np.tan(np.radians(fovy)/2)
    # Calibrated fixture dimensions; no per-episode positions are consulted.
    boxes=[]
    for y in (-1,1):
        boxes.append([[-.108,.108],[(-.098 if y<0 else .080),(-.080 if y<0 else .098)],[.703,.715]])
        boxes.append([[-.100,.100],[(-.090 if y<0 else .080),(-.080 if y<0 else .090)],[.625,.703]])
    for x in (-1,1):
        boxes.append([[(-.108 if x<0 else .090),(-.090 if x<0 else .108)],[-.080,.080],[.703,.715]])
        boxes.append([[(-.100 if x<0 else .090),(-.090 if x<0 else .100)],[-.080,.080],[.625,.703]])
    corners=np.array([[[x,y,z] for x in box[0] for y in box[1] for z in box[2]] for box in boxes])
    edges=[(a,b) for a in range(8) for b in range(a+1,8) if (a^b) in (1,2,4)]
    camera=np.asarray(camera)
    rotation=np.asarray(rotation)
    obs_count=observed.sum()
    def score(xy):
        points=corners+np.r_[xy,0]
        pc=(points-camera)@rotation
        predicted=np.zeros((128,128),dtype=np.uint8)
        for face in pc:
            front=face[:,2]<-.002
            if not front.any():
                continue
            if not front.all():
                clipped=list(face[front])
                for a,b in edges:
                    if front[a]!=front[b]:
                        alpha=(-.002-face[a,2])/(face[b,2]-face[a,2])
                        clipped.append(face[a]+alpha*(face[b]-face[a]))
                face=np.asarray(clipped)
            uv=np.stack((64+f*face[:,0]/-face[:,2],64-f*face[:,1]/-face[:,2]),axis=-1)
            uv=np.clip(uv,-10000,10000).round().astype(np.int32)
            cv2.fillConvexPoly(predicted,cv2.convexHull(uv),1)
        p=predicted.astype(bool)&valid
        intersect=(p&observed).sum()
        # Dice loss handles partially visible tote edges, with occluders masked.
        return float(2*intersect/max(1,p.sum()+obs_count))
    best=(-1,None)
    coarse=[]
    if prior is None:
        for x in np.arange(-.02,.321,.02):
            for y in np.arange(-.32,-.079,.02):
                value=score(np.array([x,y]))
                coarse.append((value,np.array([x,y])))
                if value>best[0]: best=(value,np.array([x,y]))
        step=.01
    else:
        best=(score(np.asarray(prior)),np.asarray(prior))
        step=.012
    for h in (step,step/3,step/9):
        center=best[1].copy()
        for dx in (-2,-1,0,1,2):
            for dy in (-2,-1,0,1,2):
                xy=center+np.array([dx,dy])*h
                value=score(xy)
                if value>best[0]: best=(value,xy)
    near=[]
    for dx in range(-4,5):
        for dy in range(-4,5):
            xy=best[1]+np.array([dx,dy])*.01
            if score(xy)>=best[0]-.025:
                near.append(xy)
    span=np.ptp(np.array(near),axis=0) if near else np.array([1.,1.])
    if coarse:
        peak=max(v for v,p in coarse)
        plausible=np.array([p for v,p in coarse if v>=peak-.04])
        span=np.maximum(span,np.ptp(plausible,axis=0))
    return dict(xy=best[1],score=best[0],pixels=int(obs_count),span=span)


class PickServo:
    """Visual approach, jaw alignment and lift; only RGB and robot joints enter."""
    def __init__(self, ik, calibration, close_s=.64, takeover_z=.735):
        self.ik = ik
        self.calibration = calibration
        self.close_s=close_s
        self.takeover_z=takeover_z
        self.phase = 'base'
        self.observation = None
        self.last_seen = None
        self.yaw = None
        self.close_t = None
        self.hold = None
        self.lift_xy = None
        self.events = []
        self.retry_t=None
        self.retries=0

    def observe(self, rgb, q, grip, now):
        if self.phase in ('close','lift','done'):
            return
        camera,rotation=self.calibration.pose(q,grip)
        cube=cube_top(rgb,camera,rotation,self.calibration.fovy)
        if cube is None or np.any(np.abs(cube['dims']-.032)>.007):
            return
        self.observation = (now,cube)
        self.last_seen = now
        # A square's two face normals are equivalent; choose closest to +Y.
        ang=np.radians(cube['angle'])
        options=[ang+k*np.pi/2 for k in range(-2,3)]
        reference=np.pi/2 if self.yaw is None else np.arctan2(self.yaw[1],self.yaw[0])
        ang=min(options,key=lambda a:abs(np.arctan2(np.sin(a-reference),np.cos(a-reference))))
        y=np.array([np.cos(ang),np.sin(ang),0.])
        # Once near the jaws, side-face pixels and occlusion bias rectangle yaw.
        # Learn yaw from the whole visible square while still above the part.
        site,_=self.ik.pose(q)
        if self.yaw is not None and (site[2]<.71 or np.any(np.abs(cube['dims']-.032)>.0025)):
            return
        self.yaw=y if self.yaw is None else .5*self.yaw+.5*y
        self.yaw /= np.linalg.norm(self.yaw)

    def step(self, now, q, grip):
        site,rot=self.ik.pose(q)
        if self.phase=='done':
            return None
        if self.phase=='base':
            if self.observation is None:
                return None
            t,cube=self.observation
            if cube['xy'][0]<.025 or now-t>.12 or site[2]>self.takeover_z or site[1]<.20:
                return None
            self.phase='orient'
            self.events.append(dict(t=now,phase=self.phase))
        if self.phase in ('orient','approach'):
            t,cube=self.observation
            # A correctly aligned jaw can occlude the top during the final
            # descent. Continue the last visual track using the configured
            # conveyor speed rather than abandoning a nearly completed grasp.
            if now-t>1.0:
                self.phase='done'
                self.events.append(dict(t=now,phase='lost_view'))
                return None
            yaw=self.yaw
            ring=np.array([-yaw[1],yaw[0]])
            offset=ring*.0136+yaw[:2]*.00185
            xy=cube['xy']+np.array([.08*(now-t+.25),0.]) + offset
            if self.phase=='orient':
                jaw_xy=rot[:2,0]/max(np.linalg.norm(rot[:2,0]),1e-8)
                angular_error=np.arccos(np.clip(jaw_xy@yaw[:2],-1,1))
                if angular_error<.08 and np.linalg.norm(site[:2]-xy)<.05:
                    self.phase='approach'
                    self.events.append(dict(t=now,phase=self.phase))
            goal=np.r_[xy,.75 if self.phase=='orient' else .641]
            # Keep the descent above the moving part, and track it at low height.
            delta=goal-site
            target=site+delta*min(1.,.010/max(np.linalg.norm(delta),1e-9))
            arm,err,_=self.ik.solve(target,q0=q,yaw=yaw,outer=4,iters=12,restarts=0)
            arm=q+np.clip(arm-q,-.22,.22)
            alignment=np.linalg.norm((cube['xy']+np.array([.08*(now-t+.12),0.])+offset)-site[:2])
            if self.phase=='approach' and site[2]<.648 and alignment<.008:
                self.phase='close'
                self.close_t=now
                close_goal=np.r_[cube['xy']+np.array([.08*(now-t+.15),0.])+offset,.641]
                self.hold,_,_=self.ik.solve(close_goal,q0=q,yaw=yaw,outer=4,iters=12,restarts=0)
                self.events.append(dict(t=now,phase=self.phase,xy_error=float(alignment)))
                return self.hold,True
            return arm,False
        if self.phase=='close':
            if now-self.close_t<self.close_s:
                return self.hold,True
            if grip<-.10 and self.retries<2:
                self.phase='retry'
                self.retry_t=now
                self.retries+=1
                self.events.append(dict(t=now,phase=self.phase))
                return self.hold,False
            self.phase='lift'
            self.lift_xy=site[:2].copy()
            self.events.append(dict(t=now,phase=self.phase,grip=float(grip)))
        if self.phase=='retry':
            if now-self.retry_t>.24 and self.observation is not None and now-self.observation[0]<.12:
                self.phase='approach'
                self.events.append(dict(t=now,phase=self.phase))
            elif now-self.retry_t>1.0:
                self.phase='done'
                self.events.append(dict(t=now,phase='retry_no_view'))
                return None
            return self.hold,False
        if self.phase=='lift':
            if site[2]>.795:
                self.phase='done'
                self.events.append(dict(t=now,phase=self.phase))
                return None
            goal=np.r_[self.lift_xy,.82]
            delta=goal-site
            target=site+delta*min(1.,.008/max(np.linalg.norm(delta),1e-9))
            arm,err,_=self.ik.solve(target,q0=q,yaw=self.yaw,outer=4,iters=12,restarts=0)
            return q+np.clip(arm-q,-.20,.20),True


class PlaceServo:
    """Guide the held part to a tote fitted from wrist pixels, at rim clearance."""
    def __init__(self, ik, calibration):
        self.ik=ik
        self.calibration=calibration
        self.phase='base'
        self.center=None
        self.yaw=None
        self.events=[]
        self.last_fit=-1.

    def observe(self,rgb,q,grip,now,lifted,released=False):
        site,rot=self.ik.pose(q)
        if released:
            self.phase='done'
        if not lifted or self.phase=='done' or site[1]>.13 or site[2]<.755 or now-self.last_fit<.16:
            return
        self.last_fit=now
        camera,rotation=self.calibration.pose(q,grip)
        fit=tote_pose(rgb,camera,rotation,self.calibration.fovy,prior=self.center)
        if fit is None or fit['score']<.85 or fit['pixels']<250 or np.any(fit['span']>.04):
            return
        if not (-.05<fit['xy'][0]<.35 and -.36<fit['xy'][1]<-.065):
            return
        if self.center is None:
            self.center=fit['xy']
            self.phase='align'
            self.yaw=rot[:,0].copy()
            self.yaw[2]=0
            self.yaw/=np.linalg.norm(self.yaw)
            self.events.append(dict(t=now,phase=self.phase,center=self.center.tolist(),score=fit['score']))
        else:
            self.center=.7*self.center+.3*fit['xy']

    def step(self,now,q,grip,base_arm,lifted,released):
        if released:
            self.phase='done'
        if self.phase=='done' or not lifted:
            return None
        site,rot=self.ik.pose(q)
        if self.phase=='base':
            target,_=self.ik.pose(base_arm)
            if target[2]>.79 or target[1]>.13:
                return None
            # Inspect the rim above its physical clearance before committing
            # to the final descent. The policy continues supplying lateral aim.
            target[2]=.79
            yaw=rot[:,0].copy()
            yaw[2]=0
            yaw/=np.linalg.norm(yaw)
            delta=target-site
            target=site+delta*min(1.,.008/max(np.linalg.norm(delta),1e-9))
            arm,err,_=self.ik.solve(target,q0=q,yaw=yaw,outer=4,iters=12,restarts=0)
            return q+np.clip(arm-q,-.18,.18),True
        if grip>.5:
            self.phase='done'
            self.events.append(dict(t=now,phase=self.phase))
            return None
        yaw=self.yaw
        offset=np.array([-yaw[1],yaw[0]])*.0136+yaw[:2]*.00185
        xy=self.center+offset
        distance=np.linalg.norm(site[:2]-xy)
        if self.phase=='align' and distance<.012:
            self.phase='descend'
            self.events.append(dict(t=now,phase=self.phase,center=self.center.tolist()))
        height=.79 if self.phase=='align' else .725
        delta=np.r_[xy,height]-site
        target=site+delta*min(1.,.007/max(np.linalg.norm(delta),1e-9))
        arm,err,_=self.ik.solve(target,q0=q,yaw=yaw,outer=4,iters=12,restarts=0)
        return q+np.clip(arm-q,-.18,.18),True
