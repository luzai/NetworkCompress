# to verify because import tf in GA or in Config cause the problem
import multiprocessing as mp
import GAClient

queue = mp.Queue()
# tt = []
# for name in ['ga', 'ga', 'ga' ,'ga_iter_1_ind_1']:
#     print name
#     t = mp.Process(target=GAClient.run, args=(queue, name))
#     t.start()
#     tt.append(t)
# # ResourceExhaustedError
# for t in tt:
#     t.join()

for name in ['ga_iter_0_ind_0']:#, 'ga_iter_1_ind_1']:
    GAClient.run_self(queue, name, epochs=1, verbose=1,limit_data=True)
print queue.get()
# import  subprocess
# subprocess.call("which python".split())
