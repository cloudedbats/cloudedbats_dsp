#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Project: http://cloudedbats.org
# Copyright (c) 2017-2018 Arnold Andreasson 
# License: MIT License (see LICENSE.txt or http://opensource.org/licenses/mit).

import time
import queue
import threading

class SoundStreamManager(object):
    """ Manager class for sound processing. 
        The module also contains base classes for sources, processing 
        algorithms and targets. All parts are running in separate 
        threads connected by queues. 
        Dataflow:
            Source ---> Queue ---> Process ---> Queue ---> Target 
    """
    def __init__(self, 
                source_object=None, 
                process_object=None, 
                target_object=None,
                source_queue_max=1000, # Max items.
                target_queue_max=1000,): # Max items.
        """ """
        self.source_queue = queue.Queue(maxsize=source_queue_max)
        self.target_queue = queue.Queue(maxsize=target_queue_max)
        #
        self._source = source_object
        self._process = process_object
        self._target = target_object
        #
        self._source.setup(self)
        self._process.setup(self)
        self._target.setup(self)
        #
        self._source_thread = None
        self._process_thread = None
        self._target_thread = None
         
    def start_streaming(self):
        """ """
        # Start target in thread.
        self._target_thread = threading.Thread(target=self._target.target_exec, args=[])
        self._target_thread.start()
        # Start process in thread.
        self._process_thread = threading.Thread(target=self._process.process_exec, args=[])
        self._process_thread.start()
        # Start source in thread.
        self._source_thread = threading.Thread(target=self._source.source_exec, args=[])
        self._source_thread.start()

    def stop_streaming(self, stop_immediate=False):
        """ """
        if stop_immediate:
            # Stop all.
#             self._source.stop()
#             self._process.stop()
#             self._target.stop()
            self._target.stop()
            self._process.stop()
            self._source.stop()
        else:
            # Stop source only. 
            self._source.stop()


class SoundSourceBase(object):
    """ Base class for sound sources. Mainly files or streams. """
    
    def __init__(self):
        """ """
        self._active = False
        self.source_queue = None
    
    def setup(self, manager_object):
        """ """
        self.source_queue = manager_object.source_queue
    
    def push_item(self, item, skip_if_full=False):
        """ """
        if skip_if_full:
            try: self.source_queue.put(item, block=False)
            except queue.Full: pass # Skip.
        else:
            self.source_queue.put(item, block=True, timeout=None)
    
    def stop(self):
        """ """
        self._active = False
    
    def source_exec(self):
        """ Abstract method. Override in subclass. """
        # Example and test implementation:
        self._active = True
        item_counter = 1
        while self._active:
            item = 'Item number: ' + str(item_counter)
            item_counter += 1
            self.push_item(item)
#             self.push_item(item, skip_if_full=True)
            #
            if item_counter > 1000:
                print('Source terminated.')
                self.push_item(None) # Terminate.
                self._active = False


class SoundProcessBase(object):
    """ Base class for sound processing algorithms. """
    def __init__(self):
        """ """
        self._active = False
        self.source_queue = None
        self.target_queue = None
    
    def setup(self, manager_object):
        """ """
        self.source_queue = manager_object.source_queue
        self.target_queue = manager_object.target_queue

    def pull_item(self):
        """ """
        return self.source_queue.get()
        
    def push_item(self, item):
        """ """
        self.target_queue.put(item, block=True, timeout=None)
    
    def stop(self):
        """ """
        self._active = False
    
    def process_exec(self):
        """ Abstract method. Override in subclass. """
        # Example and test implementation:
        self._active = True
        while self._active:
            item = self.pull_item()
            if item is None:
                print('Process terminated.')
                self.push_item(None) # Terminate.
                self._active = False
            else:
                item = item.upper() # Processing step.
                self.push_item(item)


class SoundTargetBase(object):
    """ Base class for sound targets. Mainly files or streams. """
    def __init__(self):
        """ """
        self._active = False
        self.target_queue = None
    
    def setup(self, manager_object):
        """ """
        self.target_queue = manager_object.target_queue
    
    def pull_item(self):
        """ """
        return self.target_queue.get()
        
    def stop(self):
        """ """
        self._active = False
    
    def target_exec(self):
        """ Abstract method. Override in subclass. """
        # Example and test implementation:
        self._active = True
        while self._active:
            item = self.pull_item()
            if item is None:
                print('Target terminated.')
                self._active = False # Terminated.
            else:
                print('Target: ' + item)



# === MAIN ===    
if __name__ == "__main__":
    """ """
    print('Test started.')
    source = SoundSourceBase()
    process = SoundProcessBase()
    target = SoundTargetBase()
    stream_manager = SoundStreamManager(
                        source, 
                        process, 
                        target,
                        source_queue_max=20,
                        target_queue_max=20)
    stream_manager.start_streaming()    
    time.sleep(0.01)
#     stream_manager.stop_streaming(stop_immediate=True)
    print('Test finished.')
