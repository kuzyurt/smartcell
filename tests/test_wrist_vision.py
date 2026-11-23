"""Independent geometry, image and observation-only controller contracts."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "smartcell"))
import cv2
import numpy as np
import mujoco
from wrist_vision import cube_top, plane_points, WristCalibration


class WristVisionTests(unittest.TestCase):
    def test_plane_projection_recovers_metric_coordinates(self):
        camera=np.array([.11,.20,.90])
        rotation=np.eye(3)
        points=np.array([[.10,.24,.657],[.13,.25,.657],[.08,.22,.657]])
        pc=(points-camera)@rotation
        f=64/np.tan(np.radians(61)/2)
        uv=np.c_[63.5+f*pc[:,0]/-pc[:,2],63.5-f*pc[:,1]/-pc[:,2]]
        np.testing.assert_allclose(plane_points(uv,camera,rotation,61,.657),points,atol=1e-12)

    def test_square_localization_across_yaw_brightness_and_tint(self):
        camera=np.array([.1,.245,.9])
        f=64/np.tan(np.radians(61)/2)
        xy=np.array([.112,.247])
        for deg in (3,24,44,46,61,87):
            angle=np.radians(deg)
            rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*.016@rot.T+xy
            uv=np.c_[63.5+f*(corners[:,0]-camera[0])/(camera[2]-.657),
                     63.5-f*(corners[:,1]-camera[1])/(camera[2]-.657)]
            for tint in ((1,1,1),(.6,1,.8),(1,.6,1)):
                image=np.zeros((128,128,3),dtype=np.uint8)
                color=tuple((np.array([235,110,105])*tint*.75).astype(int).tolist())
                cv2.fillConvexPoly(image,uv.round().astype(np.int32),color)
                # A pure-red robot surface must not be confused with the top.
                image[5:25,5:25]=[255,0,0]
                fit=cube_top(image,camera,np.eye(3),61)
                self.assertIsNotNone(fit,(deg,tint))
                self.assertLess(np.linalg.norm(fit['xy']-xy),.0015,(deg,tint))

    def test_camera_calibration_matches_render_pose_without_object_state(self):
        from conveyor_env import ConveyorPickEnv
        from cam_variant import Mount
        env=ConveyorPickEnv()
        try:
            mount=Mount(env.model)
            calibration=WristCalibration(env.model,mount)
            self.assertEqual(int(env.model.opt.integrator),int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST))
            dofs=[int(env.model.joint(n).dofadr[0]) for n in
                  ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')]
            env.data.ctrl[env._arm_aid]=[-1.44,-1.465,.3,1.657,-1.449]
            for _ in range(200):
                mujoco.mj_step(env.model,env.data)
                mount.update(env.data)
                photo_q=env.data.qpos[env._arm_qpos]-env.model.opt.timestep*env.data.qvel[dofs]
                position,rotation=calibration.pose(photo_q,.64)
                np.testing.assert_allclose(position,env.data.cam_xpos[mount.cid],atol=1e-12)
                np.testing.assert_allclose(rotation,env.data.cam_xmat[mount.cid].reshape(3,3),atol=1e-12)
            real_state=env.data.qpos.copy()
            calibration.data.qpos[env._cube_qpos[0]:env._cube_qpos[0]+3]=[10,-5,3]
            position2,rotation2=calibration.pose(photo_q,.64)
            np.testing.assert_allclose(position2,position,atol=1e-12)
            np.testing.assert_allclose(rotation2,rotation,atol=1e-12)
            np.testing.assert_array_equal(real_state,env.data.qpos)
        finally:
            env.close()


if __name__=='__main__':
    unittest.main()
