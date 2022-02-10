import numpy as np
import config as p
REF_IMG_RESO = 128
BOX_NUM = 27
MAX_STEP = 300
class Memory():

    def __init__(self, capacity):
        super(Memory, self).__init__()

        self.ref_size = REF_IMG_RESO
        self.box_num = BOX_NUM
        self.capacity = capacity
        self.max_step = MAX_STEP

        self.s_mem = np.zeros((self.capacity, 1, self.ref_size, self.ref_size))

        self.a_mem = np.zeros((self.capacity, 1))
        self.r_mem = np.zeros((self.capacity, 1))

        self._s_mem = np.zeros((self.capacity, 1, self.ref_size, self.ref_size))
        self.memory_counter=0
        self.done=False

    def store(self, s, a, r, s_, done):
        index = self.memory_counter % self.capacity

        self.s_mem[index, :] = s
        self.a_mem[index, :] = a
        self.r_mem[index, :] = r
        self._s_mem[index, :] = s_
        self.done=done

        self.memory_counter += 1

    def clear(self):
        self.memory_counter = 0

    def sample(self, num):
        if self.memory_counter < self.capacity:
            indices = np.random.choice(self.memory_counter, size=num)
        else:
            indices = np.random.choice(self.capacity, size=num)

        bs = self.s_mem[indices, :]
        ba = self.a_mem[indices, :]
        br = self.r_mem[indices, :]
        bs_ = self._s_mem[indices, :]
        bdone=self.done
        return bs, ba, br, bs_, bdone
