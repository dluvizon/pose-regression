import queue
import threading

import numpy as np
import scipy

import pygame
pygame.init()
import pygame.camera
pygame.camera.init()
import pygame.display

import matplotlib.pyplot as plt

video_device = '/dev/video0'
cam_res = (1280, 720)

weights_file = 'reception_mpii_weights_tf_ch_last_v1.h5'
TF_WEIGHTS_PATH = \
        'https://github.com/dluvizon/pose-regression/releases/download/0.1.1/' \
        + weights_file
md5_hash = '0f41d21e6c049ca590b520367f950f7f'
cache_subdir = 'models'

try:
    print ('Opening device ' + str(video_device) + ' with resolution ' +
            str(cam_res))
    cam = pygame.camera.Camera(video_device, cam_res)
    cam.start()
    cam_res = cam.get_size()
    print ('Device started with resolution ' + str(cam_res))
except Exception as e:
    print ('Got an exception: ' + str(e))
    sys.exit()

img_size = (min(cam_res), min(cam_res))
print ('Cropping current frame size to ' + str(img_size))

hmsurf_size = (170, 170)
lateral_margin = 2*hmsurf_size[1]
screen_size = (img_size[0] + lateral_margin, img_size[1])
print ('Screen size: ' + str(screen_size))

win_size = (256, 256)
input_shape = win_size + (3,)
print ('Network input shape ' + str(input_shape))


# Load keras and posereg libs
from keras.models import Model
from keras.utils.data_utils import get_file

import posereg
from posereg import pa16j


# Define the colors we will use in RGB format
BLUE =    ( 64,  64, 255)
GREEN =   (  0, 228,  16)
RED =     (255,  16,  32)
YELLOW =  (255, 255,   0)
MAGENTA = (255,   0, 255)

links = pa16j.links
cmap = pa16j.cmap
color = [GREEN, RED, BLUE, YELLOW, MAGENTA]


def get_frame(frame, skipframe=False):
    if skipframe:
        for i in range(3):
            """Stupid way to discard internal buffered frames,
            since there is no easy way to control it using pygame.
            """
            img = cam.get_image()
    else:
        img = cam.get_image()

    x1 = int((cam_res[0] - img_size[0]) / 2)
    x2 = x1 + img_size[0]
    frame.blit(img, (0,0), (x1, 0, x2, img_size[1]))


def surface_to_array(win, frame):
    pygame.transform.scale(frame, win.get_size(), win)
    x = pygame.surfarray.pixels3d(win).copy()
    x = x.transpose((1, 0, 2)).astype(np.float32)
    x /= 255.
    x -= 0.5
    x *= 2.
    return x.reshape((1,) + x.shape)


def draw_pose(screen, pose, visible, w, h, prob_thr=0.):
    pose = pose.squeeze()
    visible = visible.squeeze()
    pose[:,0] *= w
    pose[:,1] *= h
    for i in links:
        if ((visible[i[0]] > prob_thr) and (visible[i[1]] > prob_thr)):
            c = color[cmap[i[0]]]
            pygame.draw.lines(screen, c, False, pose[i,:], 10)

def draw_heatmaps(screen, surf, hm, thr=0.5, vmin=-15, vmax=10):
    hm_idx = [
            ( 8, 0*hmsurf_size[0], 0*hmsurf_size[1]),   # R. wrist
            ( 9, 1*hmsurf_size[0], 0*hmsurf_size[1]),   # L. wrist
            ( 6, 0*hmsurf_size[0], 1*hmsurf_size[1]),   # R. elbow
            ( 7, 1*hmsurf_size[0], 1*hmsurf_size[1]),   # L. elbow
            ( 3, 0*hmsurf_size[0], 2*hmsurf_size[1]),   # Head
            ( 0, 1*hmsurf_size[0], 2*hmsurf_size[1]),   # Pelvis
            (12, 0*hmsurf_size[0], 3*hmsurf_size[1]),   # R. knee
            (13, 1*hmsurf_size[0], 3*hmsurf_size[1])]   # L. knee

    for idx in hm_idx:
        h = np.transpose(hm[:,:,idx[0]].copy(), (1, 0))
        h[h < vmin] = vmin
        h[h > vmax] = vmax
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cm = np.zeros((34, 34, 3))
        cm[1:33, 1:33, :] = cmap(norm(h))[:,:,0:3]
        cm = scipy.ndimage.zoom(cm, (5, 5, 1), order=1)
        pygame.surfarray.pixels3d(surf)[:,:,:] = np.array(255.*cm, dtype=int)
        screen.blit(surf, (idx[1] + img_size[0], idx[2]))


def thread_grab_frames(queue_frames, queue_poses):
    win = pygame.Surface(win_size)
    frame = pygame.Surface(img_size)
    hmsurf = pygame.Surface(hmsurf_size)
    screen = pygame.display.set_mode(screen_size)

    while True:
        get_frame(frame)
        x = surface_to_array(win, frame)
        queue_frames.put(x)

        screen.blit(frame, (0,0))
        pred = queue_poses.get()

        # Unpack received data
        x = pred[-1][0]
        hm = pred[-2][0]
        v = pred[-3]
        p = pred[-4]

        draw_pose(screen, p, v, img_size[0], img_size[1], prob_thr=0.7)
        draw_heatmaps(screen, hmsurf, hm)

        pygame.display.update()


def main_thread():

    # Build the model and load the pre-trained weights on MPII
    model = posereg.build(input_shape, pa16j.num_joints, export_heatmaps=True)
    weights_path = get_file(weights_file, TF_WEIGHTS_PATH, md5_hash=md5_hash,
            cache_subdir=cache_subdir)
    model.load_weights(weights_path)

    queue_frames = queue.Queue(2)
    queue_poses = queue.Queue(2)
    proc = threading.Thread(target=thread_grab_frames,
            args=(queue_frames, queue_poses))
    proc.daemon = True
    proc.start()

    clock = pygame.time.Clock()

    show_fps_cnt = 0
    while True:
        x = queue_frames.get()
        pred = model.predict(x)
        pred.append(x) # Append the input frame
        queue_poses.put(pred)

        clock.tick()
        show_fps_cnt += 1
        if show_fps_cnt == 10:
            show_fps_cnt = 0
            print ('fps: ' + str(clock.get_fps()))

if __name__ == "__main__":
    main_thread()

