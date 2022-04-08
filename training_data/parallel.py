import multiprocessing
from joblib import Parallel, delayed
# from tqdm import tqdm
import time
from tqdm import tqdm

num_cores = multiprocessing.cpu_count()
inputs = 10* [1]
inputs = tqdm(inputs)

def my_function(random_param):
    print(random_param)

if __name__ == "__main__":
    start = time.process_time()
    Parallel(n_jobs=num_cores)(delayed(my_function)("Hi") for i in inputs)
    # your code here    
    print(time.process_time() - start)


