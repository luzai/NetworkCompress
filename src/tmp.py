# to verify because import tf in GA or in Config cause the problem
import multiprocessing as mp
import GAClient
import subprocess, os

# my_env = os.environ.copy()
# # todo may return : 1. mem not enough 2. model cannot fit
# PATH = "/new_disk_1/luzai/App/mpy/bin:$PATH"
os.environ['PATH'] = '/new_disk_1/luzai/App/mpy/bin:' + os.environ['PATH']
subprocess.call("which python".split())

queue = mp.Queue()

for name in ['ga_iter_2_ind_7']:
    GAClient.run_self(queue, name, epochs=1, verbose=1, limit_data=True)
print queue.get()
