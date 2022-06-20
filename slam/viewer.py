import numpy as np
import cv2

import OpenGL.GL as gl
import pypangolin as pangolin

import time
from multiprocessing import Process, Queue


def pango_DrawLines(points):
    return pangolin.glDrawLines(points)

def pango_DrawBoxes(cameras, sizes):
    for (c, s) in zip(cameras, sizes):
        pangolin.glDrawRectPerimeter(c);

class DynamicArray(object):
    def __init__(self, shape=3):
        if isinstance(shape, int):
            shape = (shape,)
        assert isinstance(shape, tuple)

        self.data = np.zeros((1000, *shape))
        self.shape = shape
        self.ind = 0

    def clear(self):
        self.ind = 0

    def append(self, x):
        self.extend([x])
    
    def extend(self, xs):
        if len(xs) == 0:
            return
        assert np.array(xs[0]).shape == self.shape

        if self.ind + len(xs) >= len(self.data):
            self.data.resize(
                (2 * len(self.data), *self.shape) , refcheck=False)

        if isinstance(xs, np.ndarray):
            self.data[self.ind:self.ind+len(xs)] = xs
        else:
            for i, x in enumerate(xs):
                self.data[self.ind+i] = x
            self.ind += len(xs)

    def array(self):
        return self.data[:self.ind]

    def __len__(self):
        return self.ind

    def __getitem__(self, i):
        assert i < self.ind
        return self.data[i]

    def __iter__(self):
        for x in self.data[:self.ind]:
            yield x




class MapViewer(object):
    def __init__(self, system=None, config=None):
        self.system = system
        self.config = config

        self.saved_keyframes = set()

        # data queue
        self.q_pose = Queue()
        self.q_active = Queue()
        self.q_points = Queue()
        self.q_colors = Queue()
        self.q_graph = Queue()
        self.q_camera = Queue()
        self.q_image = Queue()
        self.q_objects = Queue()

        # message queue
        self.q_refresh = Queue()
        # self.q_quit = Queue()

        self.view_thread = Process(target=self.view)
        self.view_thread.start()

    def update(self, refresh=False):
        while not self.q_refresh.empty():
            refresh = self.q_refresh.get()

        self.q_image.put(self.system.current.image)
        self.q_pose.put(self.system.current.pose.matrix())

        points = []
        for m in self.system.reference.measurements():
            if m.from_triangulation():
                points.append(m.mappoint.position) 
        self.q_active.put(points)

        lines = []
        for kf in self.system.graph.keyframes():
            if kf.reference_keyframe is not None:
                lines.append(([*kf.position, *kf.reference_keyframe.position], 0))
            if kf.preceding_keyframe != kf.reference_keyframe:
                lines.append(([*kf.position, *kf.preceding_keyframe.position], 1))
            if kf.loop_keyframe is not None:
                lines.append(([*kf.position, *kf.loop_keyframe.position], 2))
        self.q_graph.put(lines)

        objects = []
        for feat_id, feature in self.system.object_level_map.items(): 
            objects.append(feature.my_object) 
        if len(objects) > 0:
            self.q_objects.put(objects)

        if refresh:
            print('****************************************************************', 'refresh')
            cameras = []
            for kf in self.system.graph.keyframes():
                cameras.append(kf.pose.matrix())
            self.q_camera.put(cameras)


            points = []
            colors = []
            for pt in self.system.graph.mappoints():
                points.append(pt.position)
                colors.append(pt.color)
            if len(points) > 0:
                self.q_points.put((points, 0))
                self.q_colors.put((colors, 0))
        else:
            cameras = []
            points = []
            colors = []
            for kf in self.system.graph.keyframes()[-20:]:
                if kf.id not in self.saved_keyframes:
                    cameras.append(kf.pose.matrix())
                    self.saved_keyframes.add(kf.id)
                    for m in kf.measurements():
                        if m.from_triangulation():
                            points.append(m.mappoint.position)
                            colors.append(m.mappoint.color)
            if len(cameras) > 0:
                self.q_camera.put(cameras)
            if len(points) > 0:
                self.q_points.put((points, 1))
                self.q_colors.put((colors, 1))

    def stop(self):
        self.update(refresh=True)
        self.view_thread.join()

        qtype = type(Queue())
        for x in self.__dict__.values():
            if isinstance(x, qtype):
                while not x.empty():
                    _ = x.get()
        print('viewer stopped')


    def view(self):
        pangolin.CreateWindowAndBind('Viewer', 1024, 768)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc (gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        panel = pangolin.CreatePanel('menu')
        panel.SetBounds(
            pangolin.Attach.Frac(0.5),
            pangolin.Attach.Frac(1.0),
            pangolin.Attach.Frac(0.0),
            pangolin.Attach.Frac(175 / 1024.))

        var_ui = pangolin.Var("ui")
        # checkbox
        var_ui.m_follow_camera = (True, pangolin.VarMeta(toggle=True))
        var_ui.m_show_points = (True, pangolin.VarMeta(toggle=True))
        var_ui.m_show_keyframes = (True, pangolin.VarMeta(toggle=True))
        var_ui.m_show_graph = (True, pangolin.VarMeta(toggle=True))
        var_ui.m_show_image = (True, pangolin.VarMeta(toggle=True))

        # button
        var_ui.m_replay = (False, pangolin.VarMeta(toggle=False))
        var_ui.m_refresh = (False, pangolin.VarMeta(toggle=False))
        var_ui.m_reset = (False, pangolin.VarMeta(toggle=False))

        if self.config is None:
            width, height = 400, 250
            viewpoint_x = 0
            viewpoint_y = -500   # -10
            viewpoint_z = -100   # -0.1
            viewpoint_f = 2000
            camera_width = 1.
        else:
            width = self.config.view_image_width
            height = self.config.view_image_height
            viewpoint_x = self.config.view_viewpoint_x
            viewpoint_y = self.config.view_viewpoint_y
            viewpoint_z = self.config.view_viewpoint_z
            viewpoint_f = self.config.view_viewpoint_f
            camera_width = self.config.view_camera_width

        proj = pangolin.ProjectionMatrix(
            1024, 768, viewpoint_f, viewpoint_f, 512, 389, 0.1, 5000)
        look_view = pangolin.ModelViewLookAt(
            viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)

        # Camera Render Object (for view / scene browsing)
        scam = pangolin.OpenGlRenderState(proj, look_view)

        # Add named OpenGL viewport to window and provide 3D Handler
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(
            pangolin.Attach(0.0),
            pangolin.Attach(1.0),
            pangolin.Attach(175 / 1024.),
            pangolin.Attach(1.0),
            -1024 / 768.)
        dcam.SetHandler(pangolin.Handler3D(scam))


        # image
        # width, height = 400, 130
        dimg = pangolin.Display('image')
        dimg.SetBounds(
            pangolin.Attach(0),
            pangolin.Attach(height / 768.),
            pangolin.Attach(0.0),
            pangolin.Attach(width / 1024.))
        # dimg.SetLock(pangolin.Lock(0), pangolin.Lock(2))

        texture = pangolin.GlTexture(width, height, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image = np.ones((height, width, 3), 'uint8')



        pose = pangolin.OpenGlMatrix()   # identity matrix
        following = True

        active = []
        replays = []
        graph = []
        loops = []
        mappoints = DynamicArray(shape=(3,))
        colors = DynamicArray(shape=(3,))
        cameras = DynamicArray(shape=(4, 4))

        object_poses = DynamicArray(shape=(4,4))
        object_sizes = DynamicArray(shape=(3,))

        while not pangolin.ShouldQuit():

            if not self.q_pose.empty():
                pose.Matrix()[:] = self.q_pose.get()

            follow = var_ui.m_follow_camera
            if follow and following:
                scam.Follow(pose, True)
            elif follow and not following:
                scam.SetModelViewMatrix(look_view)
                scam.Follow(pose, True)
                following = True
            elif not follow and following:
                following = False


            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)


            # show graph
            graph = []
            loops = []
            if not self.q_graph.empty():
                qgraph = self.q_graph.get()
                loops = [np.array(e[0][:3]) for e in qgraph if e[1] == 2]
                graph = [np.array(e[0][:3]) for e in qgraph if e[1] < 2]
            if var_ui.m_show_graph:
                if len(graph) >= 2:
                    gl.glLineWidth(1)
                    gl.glColor3f(0.0, 1.0, 0.0)
                    n = len(graph)
                    pango_DrawLines(graph[:(n//2)*2])
                if len(loops) >= 2:
                    gl.glLineWidth(2)
                    gl.glColor3f(0.0, 0.0, 0.0)
                    n = len(graph)
                    pango_DrawLines(loops[:(n//2)*2])

                gl.glPointSize(4)
                gl.glColor3f(1.0, 0.0, 0.0)
                gl.glBegin(gl.GL_POINTS)
                posem = pose.Matrix()
                gl.glVertex3d(posem[0, 3], posem[1, 3], posem[2, 3])
                gl.glEnd()

            # show objects 
            if not self.q_objects.empty(): 
                poses = []
                sizes = []
                object_list = self.q_objects.get()
                for obj in object_list:
                    poses.append(obj.wTq.matrix())
                    v = np.copy(obj.v)
                    # FIXME: why need this for visualization? 
                    v[1], v[2] = v[2], v[1]
                    sizes.append(v)
                object_poses.extend(poses)
                object_sizes.extend(sizes)
            gl.glLineWidth(3)
            gl.glColor3f(1.0, 0.0, 1.0)
            # pangolin
            # .DrawBoxes(object_poses.array(), object_sizes.array())

            # Show mappoints
            if not self.q_points.empty():
                pts, code = self.q_points.get()
                cls, code = self.q_colors.get()
                if code == 1:     # append new points
                    mappoints.extend(pts)
                    colors.extend(cls)
                elif code == 0:   # refresh all points
                    mappoints.clear()
                    mappoints.extend(pts)
                    colors.clear()
                    colors.extend(cls)

            if var_ui.m_show_points:
                gl.glPointSize(2)
                 # easily draw millions of points
                pangolin.glDrawPoints(mappoints.array()) #, colors.array())


                if not self.q_active.empty():
                    active = self.q_active.get()

                gl.glPointSize(3)
                gl.glBegin(gl.GL_POINTS)
                gl.glColor3f(1.0, 0.0, 0.0)
                for point in active:
                    gl.glVertex3f(*point)
                gl.glEnd()


            if len(replays) > 0:
                n = 300
                gl.glPointSize(4)
                gl.glColor3f(1.0, 0.0, 0.0)
                gl.glBegin(gl.GL_POINTS)
                for point in replays[:n]:
                    gl.glVertex3f(*point)
                gl.glEnd()
                replays = replays[n:]


            # show cameras
            if not self.q_camera.empty():
                cams = self.q_camera.get()
                if len(cams) > 20:
                    cameras.clear()
                cameras.extend(cams)
                
            if var_ui.m_show_keyframes:
                gl.glLineWidth(1)
                gl.glColor3f(0.0, 0.0, 1.0)
                # pangolin.DrawCameras(cameras.array(), camera_width)

            
            # show image
            if not self.q_image.empty():
                image = self.q_image.get()
                if image.ndim == 3:
                    image = image[::-1, :, ::-1]
                else:
                    image = np.repeat(image[::-1, :, np.newaxis], 3, axis=2)
                image = cv2.resize(image, (width, height))
            if var_ui.m_show_image:
                texture.Upload(image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                dimg.Activate()
                gl.glColor3f(1.0, 1.0, 1.0)
                texture.RenderToViewport()


            if var_ui.m_replay:
                replays = mappoints.array()

            if var_ui.m_reset:
                m_show_graph.SetVal(True)
                m_show_keyframes.SetVal(True)
                m_show_points.SetVal(True)
                m_show_image.SetVal(True)
                m_follow_camera.SetVal(True)
                follow_camera = True

            if var_ui.m_refresh:
                self.q_refresh.put(True)
            


            pangolin.FinishFrame()
