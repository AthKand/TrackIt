#!/usr/bin/env python


import cv2
import time
import yaml
import ctypes
import argparse
import logging
import threading
import pkg_resources
from pathlib import Path
from queue import Queue
import multiprocessing as mp
from collections import deque
from datetime import date 

import numpy as np
import os

from util.defaults import *
from util.utilities import buf_to_numpy, fmt_time, save_to_file
from lib.grabber import Grabber
from lib.writer import Writer
from lib.tracker import Tracker
from lib.qt_gui import *

# shared buffer for transferring frames between threads/processes
SHARED_ARR = None

from PyQt6.QtWidgets import QApplication, QWizard, QWizardPage, QVBoxLayout, QLineEdit, QLabel
from PyQt6.QtCore import QDir




class TrackWizard(QWizard):
    def __init__(self, parent=None):
        super(TrackWizard, self).__init__(parent)

        self.addPage(EnterTrackMasterPage())
        self.addPage(EnterMouseIdPage())
        self.addPage(EnterDatePage())

        self.button(QWizard.WizardButton.FinishButton).clicked.connect(self.run_experiment)

    def run_experiment(self):
        enter_tm = self.field("enter_track_master")
        enter_mouse_id = self.field("enter_mouse_id")
        enter_date = self.field("enter_date")
        enter = [enter_mouse_id, enter_date, enter_tm]

        print('\n \n')

        output_path = Path('./logs')  # Path.home() / 'Videos'
        logfile = output_path / "log_{}_{}".format(enter[1], enter[2])


        logging.basicConfig(level=logging.INFO, format='(%(threadName)-9s) %(message)s')

        fh = logging.FileHandler(str(logfile))
        fhf = logging.Formatter('%(levelname)s : [%(threadName)-9s] - %(message)s')
        fh.setFormatter(fhf)
        logging.getLogger('').addHandler(fh)

        # Construct the shared array to fit all frames
        cfg_path = './resources/default_config.yml'

        if not Path(cfg_path).resolve().exists():
            raise FileNotFoundError('Config file not found!')


        with open(cfg_path, 'r') as cfg_f:
            cfg = yaml.load(cfg_f, Loader=yaml.FullLoader)

        vfile = fopen('Select Video File(s)', ftype = 'video')
        print('video file imported...')

        savedir = fopen('Select Save Directory', ftype = 'dir')
        save_path = QDir(savedir).absolutePath()

        if vfile is not None:
            cfg['frame_sources'] = vfile

        cfg['outpath'] = output_path
        num_bytes = cfg['frame_width'] * (cfg['frame_height'] + FRAME_METADATA_H) * cfg['frame_colors'] * len(
            cfg['frame_sources'])
        SHARED_ARR = mp.Array(ctypes.c_ubyte, num_bytes)
        logging.debug('Created shared array: {}'.format(SHARED_ARR))

        # Load node array
        nodes_path = './resources/default_nodes.yml'
        print('Node file loaded! ')
        if not Path(nodes_path).resolve().exists():
            raise FileNotFoundError('Node file not found!')

        with open(nodes_path, 'r') as nf:
            nodes = yaml.load(nf, Loader=yaml.FullLoader)

        mp.freeze_support()

        print( '\n \n')

        ht = HexTrack(cfg=cfg, nodes=nodes['nodes'], shared_arr=SHARED_ARR, enter = enter, save = save_path)
        ht.loop()

        # Rest of the HexTrack initialisation and starting code here
        # e.g. your previous code that was inside 'if __name__ == '__main__':'
        # ... 




class HexTrack:
    def __init__(self, cfg, nodes, shared_arr, enter, save):
        threading.current_thread().name = 'HexTrack'
        nsp = enter[0]+"_"+enter[1]+"_"+enter[2]

        self.mouse = enter[0]
        self.date = enter[1]

        # Control events
        self.ev_stop = threading.Event()
        self.ev_recording = threading.Event()
        self.ev_tracking = threading.Event()
        self.ev_trial_active = threading.Event()
        self.t_phase = cv2.getTickCount()

        self._loop_times = deque(maxlen=30)

        # List of video sources
        self.cfg = cfg
        self.sources = cfg['frame_sources']

        # dummy
        self.denoising = False

        self.w = cfg['frame_width']
        self.h = cfg['frame_height'] + FRAME_METADATA_H
        self.c = cfg['frame_colors']

        self.use_frame = {'cam0' : np.zeros((self.h, self.w, self.c)), 'cam1' : np.zeros((self.h, self.w, self.c))}
        self.paused_frame = {}
        #print(self.use_frame['cam0'].shape)

        # Shared array population
        with shared_arr.get_lock():
            self._shared_arr = shared_arr
            logging.debug('Grabber shared array: {}'.format(self._shared_arr))
            self.frame = buf_to_numpy(self._shared_arr, shape=(self.h * len(self.sources), self.w, self.c))
            self.use_frame['cam0'] = self.frame[:self.h, :, :]
            self.use_frame['cam1'] = self.frame[self.h:, :, :]
        self.paused_frame['cam0'] = np.zeros_like(self.use_frame['cam0'])
        self.paused_frame['cam1'] = np.zeros_like(self.use_frame['cam1'])

        # Allocate scrolling frame with node visits
        #self.disp_frame = np.zeros((self.h, self.w + NODE_FRAME_WIDTH, 3), dtype=np.uint8)
        self.disp_frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # Frame queues for video file output
        self.queues = [Queue(maxsize=16) for _ in range(len(self.sources))]

        # Frame acquisition objects

        self.grabbers = [Grabber(cfg=self.cfg, source=self.sources[n], arr=self._shared_arr, out_queue=self.queues[n],
                                 trigger_event=self.ev_stop, idx=n) for n in range(len(self.sources))]

        # Video storage writers
        self.writers = [
            Writer(cfg=self.cfg, in_queue=self.queues[n], ev_alive=self.ev_stop, ev_recording=self.ev_recording,
                   ev_trial_active=self.ev_trial_active, idx=n) for n in range(len(self.sources))]

        # Online tracker
        self.trackers = [Tracker(cfg=self.cfg, nodes=nodes[n], idx=n) for n in range(len(self.sources))]

        # Start up threads/processes
        for n in range(len(self.sources)):
            self.grabbers[n].start()
            self.writers[n].start()

        #cv2.namedWindow('HexTrack', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('cam0', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('cam1', cv2.WINDOW_AUTOSIZE)

        logging.debug('HexTrack initialization done!')

        # save paths and log checkers
        self.save = os.path.join(save, nsp) + '{}'.format('.txt')
        self.save_nodes = [None]
        self.all_save_nodes = {}
        self.trial_count = 1
        self._tracker_deque = deque(maxlen=2)
        self._last_two = deque(maxlen=2)

        self.record_detections = False
        self.paused = False

        self.bw = False 

    def loop(self):
        frame_idx = 0
        FLAGGER = 0
        t0 = cv2.getTickCount()

        with open(self.save, 'a+') as file:
            file.write(f"Mouse id: {self.mouse} , Date: {self.date} \n")
        
        while not self.ev_stop.is_set():
            if not all([grabber.is_alive() for grabber in self.grabbers]):
                self.stop()
                break

            if self.paused:
                frame_cam0 = self.paused_frame['cam0']
                frame_cam1 = self.paused_frame['cam1']
            else:
                # Using copy prevents the image buffer to be overwritten by a new incoming frame
                # Question is, what is worse - waiting with a lock until drawing is complete,
                # or the overhead of making a full frame copy.
                # TODO: Blit the frame here into an allocated display buffer
                with self._shared_arr.get_lock():
                    frame_cam0 = self.use_frame['cam0'].copy()
                    frame_cam1 = self.use_frame['cam1'].copy()

                for tracker in self.trackers:
                    if tracker.id == 0 :
                        tracker.apply(frame_cam0, self.record_detections)
                    else:
                        tracker.apply(frame_cam1, self.record_detections)
                    if tracker.update_node:
                        if self.save_nodes[-1] != tracker.last_node:
                            self.save_nodes.append(tracker.last_node)
                            self._last_two.append(tracker.last_node)
                            self._tracker_deque.appendleft(tracker.id)


                            #print(self._tracker_deque)

                    #tracker.last_node is not None
                    tracker.annotate()

                frame_idx += 1
                delta = NODE_FRAME_STEP_SCROLL

                self.disp_frame[:, :self.w] = frame_cam0

                # trial_active = self.ev_trial_active.is_set()
                # fg_col = NODE_FRAME_FG_ACTIVE if trial_active else NODE_FRAME_FG_INACTIVE
                # bg_col = NODE_FRAME_BG_ACTIVE if trial_active else NODE_FRAME_BG_INACTIVE

                # self.disp_frame[delta:, self.w:] = self.disp_frame[:-delta, self.w:]
                # self.disp_frame[:delta, self.w:] = bg_col

                # self.disp_frame[:delta, self.w + NODE_FRAME_WIDTH - SYNC_FRAME_WIDTH:] = NODE_FRAME_BG_INACTIVE

                # Timestamp overlay
                
                self.add_overlay(frame_cam0, (cv2.getTickCount() - self.t_phase) / cv2.getTickFrequency())
                self.add_overlay(frame_cam1, (cv2.getTickCount() - self.t_phase) / cv2.getTickFrequency())

                if not self.bw:
                    for tracker in self.trackers:
                        if tracker.id == 0:
                            if tracker.show_cam:
                                cv2.imshow('cam0', frame_cam0)
                            else:
                                cv2.imshow('cam0', cv2.cvtColor(frame_cam0, cv2.COLOR_RGB2GRAY))
                        else:
                            if tracker.show_cam:
                                cv2.imshow('cam1', frame_cam1)
                            else:
                                cv2.imshow('cam1', cv2.cvtColor(frame_cam1, cv2.COLOR_RGB2GRAY))
                else:
                    cv2.imshow('cam0', frame_cam0)
                    cv2.imshow('cam1', frame_cam1)

            #cv2.imshow('HexTrack', self.disp_frame)


            # What annoys a noisy oyster? Denoising noise annoys the noisy oyster!
            # This is for demonstrating a slow processing step not hindering the acquisition/writing threads
            if self.denoising and ALLOW_DUMMY_PROCESSING:
                t = cv2.getTickCount()
                dn = cv2.fastNlMeansDenoisingColored(self.frame, None, 6, 6, 5, 15)
                logging.debug((cv2.getTickCount() - t) / cv2.getTickFrequency())
                cv2.imshow('denoised', dn)

            # Check for keypresses and such
            self.process_events()

            elapsed = ((cv2.getTickCount() - t0) / cv2.getTickFrequency()) * 1000
            self._loop_times.appendleft(elapsed)
            t0 = cv2.getTickCount()

    def add_overlay(self, frame, t):
        """Overlay of time passed in normal/recording mode with recording indicator"""
        t_str = fmt_time(t)
        ox, oy = 4, 4
        # osx = 15
        thickness = 1
        font_scale = 1.2

        ts, bl = cv2.getTextSize(t_str, FONT, font_scale, thickness + 2)

        if not self.ev_recording.is_set():
            bg, fg = (0, 0, 0), (255, 255, 255)
            radius = 0
        else:
            bg, fg = (255, 255, 255), (0, 0, 0)
            radius = 8

        cv2.rectangle(frame, (ox - thickness, frame.shape[0] - oy + thickness),
                      (ox + ts[0] + 2 * radius, frame.shape[0] - oy - ts[1] - thickness), bg, cv2.FILLED)

        cv2.putText(frame, t_str, (ox, frame.shape[0] - oy), FONT, font_scale, fg,
                    thickness, lineType=METADATA_LINE_TYPE)

        if self.ev_recording.is_set():
            cv2.circle(frame, (ox + ts[0] + radius, frame.shape[0] - ts[1] // 2 - oy), radius, (0, 0, 255), -1)

        if len(self._loop_times):
            fps_str = 'D={:.1f}fps'.format(1000 / (sum(self._loop_times) / len(self._loop_times)))
        else:
            fps_str = 'D=??.?fps'
        cv2.putText(frame, fps_str, (ox + 170, frame.shape[0] - oy - 1), FONT, 1.0, fg, lineType=METADATA_LINE_TYPE)

    def process_events(self):
        # Event loop call
        key = cv2.waitKey(25)

        # Process Keypress Events
        if key == ord('q'):
            print(self.all_save_nodes)
            self.stop()

        elif key == ord('b'):
            self.bw = not self.bw


        elif key == ord('r'):
            # Start or stop recording
            self.t_phase = cv2.getTickCount()
            if not self.ev_recording.is_set():
                self.ev_recording.set()
            else:
                self.ev_recording.clear()

        elif key == ord(' '):
            # Pause display (not acquisition!)
            self.paused = not self.paused
            if self.paused:
                self.paused_frame['cam0'] = self.use_frame['cam0'].copy()
                self.paused_frame['cam1'] = self.use_frame['cam1'].copy()


        elif key == ord('d'):
            # Enable dummy processing to slow down main loop
            # demonstrates the backend continuously grabbing and
            # writing frames even of display/tracking is slow
            self.denoising = not self.denoising

        elif key == ord('m'):
            for tracker in self.trackers:
                tracker.has_mask = False

        elif key in [ord('t'), 85, 98,ord('.')]:
            # Start/stop a trial period
            if not self.ev_trial_active.is_set():
                self.ev_trial_active.set()
                self.record_detections = True
            else:
                self.ev_trial_active.clear()
                self.all_save_nodes[f'Trial{self.trial_count}'] = self.save_nodes[1:]
                save_to_file(self.save, self.save_nodes[1:], self.trial_count)
                if len(self.save_nodes) > 1:
                    self.trial_count += 1 
                self.save_nodes = [None]
                self.record_detections = False
            logging.info('Trial {}'.format(
                '++++++++ active ++++++++' if self.ev_trial_active.is_set() else '------- inactive -------'))

        # Detect if close button of hextrack was pressed.
        # May not be reliable on all platforms/GUI backends
        if cv2.getWindowProperty('cam0', cv2.WND_PROP_AUTOSIZE) < 1 and cv2.getWindowProperty('cam1', cv2.WND_PROP_AUTOSIZE) < 1:
            self.stop()

    def stop(self):
        self.ev_stop.set()
        logging.debug('Join request sent!')

        # Shut down Grabbers
        for grabber in self.grabbers:
            grabber.join()
        logging.debug('All Grabbers joined!')

        # Shut down Writers
        for writer in self.writers:
            writer.join()
        logging.debug('All Writers joined!')
        cv2.destroyAllWindows()
        raise SystemExit


if __name__ == '__main__':

    app = QApplication([])
    wizard = TrackWizard()
    wizard.show()
    app.exec()