from collections import deque

class FrameBuffer:
    """
    FIFO buffer for storing a fixed number of frames.
    """
    def __init__(self, max_size=40):
        self.buffer = deque(maxlen=max_size)

    def push(self, frame):
        self.buffer.append(frame.copy())

    def get_clip(self):
        return list(self.buffer)
