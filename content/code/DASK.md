---
title: "DASK"
date: 2018-12-05T15:52:43-05:00
showDate: true
draft: false
---

# TOC

1. ## [Tom Augspurger Talks](#tom)

2. ## [Official: Dask.distributed & Kubernetes](#official)

3. ## [Matthew Rocklin Talks](#rocklin)

4. ## [Jim Crist Talks](#jim)

5. ## [Long SciPy Tutorial](#long)

6. ## [DataCamp](#datacamp)

7. ## [Streamz](#streamz)

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

%config InlineBackend.figure_format = 'retina'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Reference [Tom Augspurger Github](https://github.com/TomAugspurger) <a id="tom"></a>

- [Tabular Data in SKL and DaskML](https://tomaugspurger.github.io/sklearn-dask-tabular.html)
- [TPOT with Dask](https://mybinder.org/v2/gh/dask/dask-examples/master?filepath=machine-learning%2Ftpot.ipynb)

### [Examples](https://github.com/dask/dask-examples)

# Scaling Pains

- Model Complexity VS Data
  ![Scaling](https://image.slidesharecdn.com/scalable-ml-180514203801/95/scalable-machine-learning-with-dask-tom-augspurger-18-1024.jpg?cb=1526330748)
- Distributed SKL - using DASK to distribute computation on cluster (Large Model - Smaller Datasets)

> **Single-Machine Parallelism with SKL**
>
> 1. control CPU processors (`n_jobs=-1`)

> **Multi-Machine with Dask**

- `from sklearn.externals import joblib`
- `import dask_ml.joblib`
- `clf = RandomForestCalssifier(n_estimators=200, n_jobs=-1)`
- `with joblib.parallel_backend("dask", scatter=[X,y]): clf.fit(X,y)`

- Caveats
  1. Data must fit RAM
  2. Data shipped to each worker
     - Each parallel task expensive
     - Should be many parallel tasks
- First: Must all data be used?

> **Sampling may allow / plotting learning curve by data size to inspect improvement**

- Second: Parallel Meta-estimators

  - Train on subset
  - Predict for large dataset in parallel

  > wrap data X with dask.dataframe then clf.predict(X)

- Dask_ML on scalable, parallel algos (to dig) [Example of Pipeline in SKL](https://git.io/vAi7C)

- Distributed System: Peer with systems like XGBoost or TensorFlow

```python
import dask_ml.xgboost as xgb
df = dd.read_csv()
booster = xgb.train(client, params, X, y)
```

## Example of dask_ml

```python
#reading the csv files
import dask.dataframe as dd
df = dd.read_csv('blackfriday_train.csv')
test=dd.read_csv("blackfriday_test.csv")

#having a look at the head of the dataset
df.head()

#finding the null values in the dataset
df.isnull().sum().compute()

#defining the data and target
categorical_variables = df[['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']]
target = df['Purchase']

#creating dummies for the categorical variables
data = dd.get_dummies(categorical_variables.categorize()).compute()

#converting dataframe to array
datanew=data.values

#fit the model
from dask_ml.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(datanew, target)

#preparing the test data
test_categorical = test[['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']]
test_dummy = dd.get_dummies(test_categorical.categorize()).compute()
testnew = test_dummy.values

#predict on test and upload
pred=lr.predict(testnew)


from dask.distributed import Client
client = Client() # start a local Dask client

import dask_ml.joblib
from sklearn.externals.joblib import parallel_backend
with parallel_backend('dask'):

    # Create the parameter grid based on the results of random search 
     param_grid = {
    'bootstrap': [True],
    'max_depth': [8, 9],
    'max_features': [2, 3],
    'min_samples_leaf': [4, 5],
    'min_samples_split': [8, 10],
    'n_estimators': [100, 200]
    }

    # Create a based model
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor()

    
# Instantiate the grid search model
import dask_searchcv as dcv
grid_search = dcv.GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3)
grid_search.fit(data, target)
grid_search.best_params_

```

# Official Doc Dask.distributed & Kubernetes <a id="official"></a>

## QuickStart Official Doc

#### Restart Cluter/Scheduler at error

`c.restart()`

```python
from dask.distributed import Client, progress

client = Client('172.17.0.2:8786')
client
```



<table style="border: 2px solid white;">
<tr>
<td style="vertical-align: top; border: 0px solid white">
<h3>Client</h3>
<ul>
  <li><b>Scheduler: </b>tcp://172.17.0.2:8786
  <li><b>Dashboard: </b><a href='http://172.17.0.2:8787/status' target='_blank'>http://172.17.0.2:8787/status</a>
</ul>
</td>
<td style="vertical-align: top; border: 0px solid white">
<h3>Cluster</h3>
<ul>
  <li><b>Workers: </b>3</li>
  <li><b>Cores: </b>6</li>
  <li><b>Memory: </b>6.29 GB</li>
</ul>
</td>
</tr>
</table>



### Map and Submit

> Coupled to launch compu on cluster - sending (FUNC, *ARG) to remote WORKERS for processing -> returning FUTURE object referring to remote DATA on CLUSTER -> FUTURE returns immediately while comp run remotely in background

```python
def square(x):
    return x ** 2
def neg(x):
    return -x

A = client.map(square, range(10))
B = client.map(neg, A)
total = client.submit(sum, B)
total.result()
```



```
-285
```



### Gather

> map/submit return Future, lightweight tokens referring to results on cluster. By default they STAY ON CLUSTER

> Gather results to LOCAL machine either with `Future.result` method for a single future, or with the `Client.gather` for many futures at once

```python
total

total.result() # result for single future

client.gather(A) # gather for many futures
```



<b>Future: sum</b> <font color="gray">status: </font><font color="black">finished</font>, <font color="gray">type: </font>int, <font color="gray">key: </font>sum-7f3e220448a7f71ff037b16dd2be51d1





```
-285
```





```
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```



### Setup Network

- Using CLI (`dask-scheduler`)

> Various ways to deploy these CLI on cluster
>
> - manual SSH into nodes
> - auto system like SGE/Torque/Yarn/Mesos

- #### NOTE both scheduler/worker neet to accept TCP connections, default port 8786 and random for workers - if firewall need `--port -worker-port` keywords

- Using SSH (`dask-ssh` opens several SSH to target nodes INIT taking hostnames / IP

  - `dask-ssh 192.1680.0.1 192.168.0.2 ...`
  - `dask-ssh 192.168.0.{1,2,3,4}`
  - `dask-ssh --hostfile hostfile.txt` # list of IPs
  - dependency: `pip install paramiko`

- Using Shared Network File System NFS and Job Scheduler (some clusters profit from NFS to communicate shceduler location to workers)

  - `dask-scheduler --scheduler-file /path/to/scheduler.json \ dask-worker --scheduler-file /path/to/scheduler.json ...`
  - client = Client(scheduler_file='/path/to/scheduler.json')
  - refer doc for detail SGE's qsub example

- Using Python API session manually (listening port and Tornado IOLoop)

  ```python
  from distributed import Scheduler
  from tornado.ioloop import IOLoop
  from threading import Thread
  loop = IOLoop.current()
  t = Thread(target=loop.start, daemon=True)
  t.start()
  s = Scheduler(loop=loop)
  s.start('tcp://:8786')
  
  # on nodes
  from distributed import Worker
  from tornado.ioloop import IOLoop
  from threading import Thread
  loop = IOLoop.current()
  t = Thread(target=loop.start, daemon=True)
  t.start()
  w = Worker('tcp://127.0.0.1:8786', loop=loop)
  w.start() # choose randomly assigned port
  ```

- Using LocalCluster (`from distributed import LocalCluster \ client = LocalCluster(processes=False)` IOLoop in background thread)

- Using AWS (see Cloud Deployments)

- Using GC `dask-kubernetes` + Google Kubernetes Engine

- Cluster Resource Managers (Kubernetes)

  - https://github.com/martindurant/dask-kubernetes
  - https://github.com/ogrisel/docker-distributed
  - https://github.com/hammerlab/dask-distributed-on-kubernetes/

### Custom INIT

- `--preload` allows python file refer to Doc

### Client (Primary Entry Point overall)

- Ways of interaction:
  1. Client caters most `concurrent.futures` interface with `.submit .map` `Futures` obj
  2. Registers as default Dask scheduler, thus running all collections (4 of them: array, bag, df, delayed)
  3. Extra methods operating data remotely (API list)

#### Concurrent.futures

- Submit single func call with submit and many with map
  - `client.submit(func, 10); client.map(func, range(1000))`
  - Results LIVE on workers, while submitting as FUTURES will go to machine where stored and run once completed! (ASYNC)
    - `y = client.submit(func, x) # a future`
    - `total = client.submit(sum, L) # Map on L, a list of Futures`
  - Gather back results by `Future.result` for single and `client.gather` for many futures at once
    - `x.result(); client.gather(L)`
    - BUT aim to minimise commOverhead to local process, best to leave on cluster ops remotely with `submit, map, get, compute

#### Dask (Parent lib, auto-parallel algo on dsets)

- Client obj registers as default Scheduler - ALL `.compute()` will auto-start using distributed system on Dask objects
  - `my_dataframe.sum().compute() # now using client system by default`
  - Stop it by `set_as_default=False` when starting Client

> #### Dask normal `.compute()` SYNCHRONOUS`, blocking interpreter till complete - `dask.distributed` allows ASYNCHRONOUS trigging ops in background and persist in MEMO while taking on other works - typically handled with `Client.persist` for large result set and `Client.compute` smaller result
>
> - `df = client.persist(df) # trigger all compu keep df in MEMO`

#### Pure Func by Default

- Default assuming functions are **PURE**, if not `pure=False` 
- Client link a key to all compu, accessed on Future obj
  - `from operator import add`
  - `x = client.submit(add, 1, 2)`
  - `x.key` # hash code
  - KEY same across ALL compu with same input across all machines - running above on any machine in ENV getting SAME KEY
  - Scheduler avoids redundant compu - if result already in MEM, submit or map idempotent in common case
  - Maybe bad for **IMPURE** func like `random` hence 2 calls to same func produce diff results by `pure=False`
    - `client.submit(np.random.random, 1000, pure=False).key != another`

#### Tornado Coroutines

- ASYNC ENV then blocking funcs listed above become ASYN equivalents - MUST start client with `asynchronous=True` plus `yield` or `await` blocking functions

  ```python
  @gen.coroutine
  def f():
      client = await Client(asynchronous=True)
      future = client.submit(func, *args)
      result = await future
      return result
  ```

  - If reusing same client in ASYNC and SYNC ENV, apply that keyword at EACH method call

    ```python
    client = Client() # normal blocking client
    @gen.coroutine
    def f():
        futures = client.map(func, L)
        results = await client.gather(futures, asynchronous=True)
        return results
    ```

### API see Doc

### Q&A

- How to use external modules? `client.upload_file` which supports both standalone file and setuptools' .egg files for larger modules
- Too many open file descriptors? Linux system refer to User Level FD Limits
- Dask handle Data Locality? Yes both in MEM and Disk, e.g. `dask.dataframe.read_csv('hdfs://path/to/files.*.csv')` signalling name node to see locality

### Diagnosis

#### Task On and Off Times 

- Serialisation GRAY
- Dependency gathering from peers RED
- Disk I/O collecting local data ORANGE
- Execution times COLOURED per Taks
- Custom dashboard `Scheduler plugin`

#### Statistical Profiling

- Per 10ms each worker process checks what each worker threads doing, grab call stack and adds to counting data structure - recorded and clearered by second to set record 
- `/profile` plot on worker zooming into task level or `Client.profile` query data directly delivering raw data structure used to produce plots
- 10ms and 1s params controlled by `profile-interval` and `profile-cycle-interval` entries in CONFIG.YAML

### Efficiency

- Parallel computing done well is responsive and rewarding, BUT speed-bumps ahead

#### Leave data on cluster 

- Wait as long as possible to gatehr data locally - if querying large piece of data on cluster often FASTER to SUBMIT func onto that data then to bring local

- E.g. query shape of NPArray on cluster choose:

  1. SLOW: gather array to local process, call `.shape`

  2. FAST: Send lambda func up to cluster to compute shape

     ```python
     x = client.submit(np.random.random, (1000,1000)) # x is future
     # SLOW
     x.result().shape() # slow down by data transfer
     # FAST
     client.submit(lambda a: a.shape, x).result()
     ```

#### User larger tasks

- #### scheduler adds aout 1ms overhead per task or Future obj, slow if billions of tasks - if func run faster than 100ms then might not see any speedup from using distributed computing !!

- SOLUTION: BATCH INPUT INTO LARGER CHUNKS

  ```python
  # SLOW
  future = client.map(f, seq)
  len(futures) # 100000000000 avoid large numbers of futures
  # FAST
  def f_many(chunk):
      return [f(x) for x in chunk]
  from toolz import partition_all
  chunks = partition_all(10000000, seq) # collect into groups of size 1000
  futures = client.map(f_many, chunks)
  len(futures) # 1000 compu on larger pieces of data at once
  ```

#### Adjust betweeen Threads and Processes!!!

- Default single Worker runs many compu in parallel maxing out threads on machine cores! Using pure Python func may not be optimal thus prefer to run several worker processes on each node, each using one thread! When config cluster may want to use this:
  - `dask-worker ip:port --nprocs 8 --nthreads 1`
- NOTE if primarily using NP, PD, SciPy, SKL, Numba or other C/Fortran/LLVM/Cython accelerated libs then not an issue, code likely optimal for use with MULTI-THREADING

#### DONT GO Distributed

- Consider Dask and concurrent.futures modules with simiarl API operating on single machine - accelerating code through other means - better algo, data structures, stroage format, or Cython etc 10x speed boost

### LIMIT

#### Performance

- **Central scheduler spends 00s us (Microsecond) per task - for optimality, TASK DURATION > 10-100ms**

- Dask cannot parallelise within individual task - should be comfortable size so as not to overwhlem any particular worker

- Dask assigns tasks to worker heristically - often right but non-optimal decision

- #### Workers are just Python processes, inheriting all pros and cons of Python - Not bound or limit in any way, PRODUCTION may wish to run dask-workers within CONTAINERS !!

#### Assumptions on FUNC and DATA

- All func must be **serialiseable** either with PICKLE or COULDPICKLE, often the case bar farily exotic cases check by
  - `from cloudpickle import dumps, loads \ loads(dumps(my_object))`
- All data must be serialisable either pickle or coud pickel or using dask custom serialisation system
- Dask may run func multi-times, such as if worker holding an middle result dies - any side effects should be idempotent!!

### Security

- Dask enables remote execution of arbitrary code, hsuld only HOST dask-workers within network trusted

### Data Locality

- Data movement needlessly limits performance
  - `futures = client.scatter(range(10)` each worker with 2 cores and scatter out 

#### User Control

- complex algo finer control - current hardwar GPUs or database connetions may restrict est of valid workers for particular task
- Thus `workers=` options:
  - `future = client.submit(func, *args, workers=['Alice'])
  - required data always assigned to Alice, this restriction is only preference not strict, `allow_other_workers=True` signal extreme case
  - extra scatter func supports broadcast enforcing all data sent to all workers rather than round-robined - if new workers arrive will not auto-receive this data: `futures =client.scatter([1,2,3], broadcast=True)`
  - naming can be use or IP `dask-worker scheduler_address:8786 --name Alice`

#### Worker with Compute/Persist

- `worker=` keyword in `scatter, submit, map`

```python
client.submit(f, x, workers='127.0.0.1')
client.submit(f, x, workers='127.0.0.1:55852')
client.submit(f, x, workers=['192.168.1.101', '192.168.1.100'])
    # more complex compu, specify certain parts of compu run on certain workers
x = delayed(f)(1)
y = delayed(f)(2)
z = delayed(g)(x, y)
future = client.compute(z, workers={z: '127.0.0.1',
                                    x: '192.168.0.1'})

future = client.compute(z, workers={(x, y): ['192.168.1.100', '192.168.1.101:9999']})
future = client.compute(z, workers={(x, y): '127.0.0.1'},
                        allow_other_workers=True)
future = client.compute(z, workers={(x, y): '127.0.0.1'},
                        allow_other_workers=[x])
df = dd.read_csv('s3://...')
df = client.persist(df, workers={df: ...})
```

### Managing Computation

- **Data and Computation in Dask.distributed always in 1 of 3 states
  1. Concrete values in local MEM, e.g. integer 1 or NPArray in local process
  2. Lazy computations in dask graph, perhaps stored in dask.delayed or dask.dataframe
  3. Running compu or remote data, via Future pointing to compu currently in flight

#### Dask Collections to Concrete Values

- Turn any dask collection into concrete value by `.compute()` or `dask.compute`
  - Blocking until compu done, going straight from lazy dask collection to a concrete value in local MEM
  - Most typical, great when data already in MEM and want small, fast results at local process
  - `df = dd.read_csv('s3://...') \ df.value.sum().compute()`
  - HOWEVER, this way breaks down if trying to bring entire Dset back to local RAM `MemoryError()`
    - Forcing wait till compu done before handing back control of interpreter

#### Dask Collection to Futures

- Async submit lazy dask graphs to run on cluster with `client.compute` and `client.persist`
- Return Future at once - further queried to determine state of compu
  - `.compute` takes collection return single future
    - `total = client.compute(df.sum()) \ total # future \ total.result() # block until done`
  - As single future result MUST fit single worker machine, only works when results small fit RAM
    - FAIL: `future = client.compute(df)` - blows up RAM
    - GOOD: use `client.persist`
  - `.persist` submits task graph behind collection, getting Futures for call of top-most task (e.g. one Future for each Pandas DF in Dask.df)
  - Then returns copy of collection pointing to these futures instead of previous graph 
  - New collection semantically same but now points to actively running data not lazy graph
    - `df.dask` - recipe to compute df in chunks
    - `df = client.persist(df)` - start compu
    - `df.dask` - now points to running futures
  - collection returned at once while compu in run detached on cluster - ending all futures done then more queries on it very fast
  - **Typically the workflow defined a compu with `dask.dataframe, dask.delayed` until a point where nice Dset to work from, then persist that collection to cluster then perform many fast queries off the resulting collection**

#### Concrete Value to Futures

- Get futures from few ways: 1) above, by wrapping Futures with Dask collections , 2) submitting data / task directly to cluster by `client.scatter, client.submit or client.map`

```python
futures = client.scatter(args) # Send data
future = client.submit(function, *args, **kwrags) # Send single task
futures = client.map(function, sequence, **kwargs) # Send many tasks
```

- now `*args or **kwargs` can be nromal Python obj or other `Future` if to link tasks with dependencies
- **unlike Dask collections (dask.delayed) these task submissions happen at once, the concurrent.futures interface very similar to dask.delayed except that execution is immediate not lazy**

#### Futures to Concrete Values

- Turn each Future into concrete value in local process via `future.result()` - can convert collection of futures into values `client.gather`

#### Futures to Dask Collections

- As Collection to futures, common to have currently computing Future within Dask graphs
- This enables further compu on top of currently running - most often done with `dask.delayed` workflows on custom compu:
  - `x = delayed(sum)(futures)`
  - `y = delayed(product)(futures)`
  - `future = client.compute(x + y)`
- Mixing two forms allow building and submitting compu in stages like `sum(...) + product(...)` 
- This often valuable if want to wait to see values of certain parts of compu before further proceeding
- Submitting many ocmpu at ocne allows scheduler to be slightly more intelligent when determinig what gets trun

### Managing MEM

- Storing results of tasks in distr.MEM of worker nodes - tracking all data free as need 
- Done result cleared from MEM soonest - result kept in MEM if either:
  1. A client holds a future pointing to this task - data should persist in RAM to gather data on demand
  2. Task is necessary for ongoing compu working to produce final results pointed to by futures - these tasks removed once no ongoing tasks required
- When users hold future or persisted collections (which contain many such futures inside `.dask` attr) they pin those results to active MEM

- **when user deletes futures or collections from local process scheduler removes linked data from Dsitributed RAM, FOR this relationship, distributed MEM reflects state of local MEM, a user may free distributed MEM on cluster by deleting persisted collections in local session**

#### Creating Futures

```python
Client.submit(func, *args, **kwargs) # submit func to scheduler
Client.map(func, *iterables, **kwargs) # map a func on seq of args
Client.compute(collections[, sync, . . . ]) # compu dask collection on cluster
Client.persist(collections[, . . . ]) # persist dask collections on cluster
Client.scatter(data[, workers, broadcast, . . . ]) # scatter data into distr.mem
```

**The submit and map methods handle raw Python functions. The compute and persist methods handle Dask collections like arrays, bags, delayed values, and dataframes. The scatter method sends data directly from the local process.**

#### Persisting Collections

- calls to `client.compute or client.persist` submit task graphs to cluster and retur future pointing to particular ouptut tasks
- compute returns single future per input, persist reutns a copy of collection with each block or partition repalced by single future, inshort **use `persist` to keep full collection on cluster and `compute` when want a small result as a single future - persist is more common and often used with collections:**

```python
>>> # Construct dataframe, no work happens >>> df = dd.read_csv(...)
>>> df = df[df.x > 0]
>>> df = df.assign(z = df.x + df.y)
>>> # Pin data in distributed ram, this triggers computation >>> df = client.persist(df)
>>> # continue operating on df
```

> Build compu parsing CSV, filtering, adding col, up till this point all LAZY - simply a recipe to graph in df object -> `.persist(df)` cut this graph off df sending it up to scheduler receiving future in return creating new df with shallow graph pointing right to them - more or less at once - continue working on new df while cluster running graph in back

#### Difference with dask.compute

- `client.persist(df), client.compute(df)` ASYNC so differ from `df.compute()` or `dask.copute`, which blocks until result available
- `.compute()` NOT persist any data on cluster - also brings entire result back to local -> unwise to use on large data
- BUT `compute()` very easy for smaller result as concrete result most other tools expect !
- Typically use ASYNC ones to set up large collections and `df.compute()` for fast analyses:

```python
>>> # df.compute() # This is bad and would likely flood local memory
>>> df = client.persist(df) # This is good and asynchronously pins df
>>> df.x.sum().compute() # This is good because the result is small
>>> future = client.compute(df.x.sum()) # This is also good but less intuitive
```

#### Clearing data

- Remove data from cluster RAM by removing collection from local process - remote data removed once all Futures poiting to it removed from all client machiens `del df # deleting local data often deletes remote data`
  - if this the only copy then will trigger cluster to delete as well
  - BUT if multiple copies or other colections based on it then have to delete them all!
  - `df2 = df[df.x < 10] \ del df # would not delete, as df2 still tracks the futures`

#### Hard Clearing ddata

- `client.cancel(df) # kills df, df2 and all other dependent compu`
- OR, retart cluster

#### Resilience

- results not intentionally COPIED unless necessary for compu on other nodes - resilience via recompu by keeping provenance of any result - if a worker node down scheduler able to recompu ALL its results
- FULL graph for any desired future kept until no references to future

### Advanced techniques

- 







# Matthew Rocklin Talks <a id="rocklin"></a>

## DASK

- Dynamic task shceduler for generic applications
- Handles data locality, resilience, work stealing, etc
- With 10ms roundtrip latencies and 200us overheads
- Native Pythn lib respecting protocols
- Lightweight and well supported

### Single Machine Scheduler

- Parallel CPU: uses multiple threads or processes
- Minimise RAM: choose tasks to remove intermediates
- Low overhead: ~100us per task
- Concise: ~1000 LOC
- Real world workloads: Under heavy load by diff projects

### Distributed Scheduler

- Distributed: One scheduler coordinates many workers
- Data local: moves compu to correct worker
- Asynchronous: continous non-blocking conversation
- Multi-user: several users share same system
- HDFS Aware: works well with HDFS, S3, YARN etc
- Solidly supports: dask.array, dask.dataframe, dask.bag, dask.delayed, concurrent.futures
- Less Concise: ~5000 LOC Tornado TCP application 

> all of logic is hackable Python, separate from Tornado

### Concurrent.futures (PEP 3148)

```python
from dask.distributed import as_completed

data = range(100)

futures = []
for x in data:
    if x % 2 == 0:
        future = client.submit(inc, x)
    else:
        future = client.submit(dec, x)
    futures.append(future)

done = as_completed(futures)

while True:
    try:
        a = next(done)
        b = next(done)
    except StopIteration:
        break
        
    future = client.submit(add, a, b)
    done.add(future)
```

### Async/Await

```python
async def f():
    total = 0
    async with Client('dask-scheduler:8786', start=False) as client:
        futures = client.map(inc, range(100))
        async for future in as_completed(futures):
            result = await future
            total += result
    print(total)

from tornado.ioloop import IOLoop
IOLoop.current().add_callback(f)
```

> By reusing existing API and protocols, Dask enables parallelsiation of existing codebases with minimal refactoring

​    

### `dask.distributed` Scheduler (Even on Single Machine)

> Keynotes
>
> 1. Motivation to use dask.distributed shceduler
> 2. Jim Crist's talk on bokeh visualisation
> 3. concurrent futures API
>    - flexible like dask.delayed
>    - real-time control
>    - works great with collections
>    - fully async/await compliant

> Hard and Fun DevOps
>
> 1. Collections (array, bag, dataframe)
>    - Dense linalg / Sparse arrays / Streaming Pandas
>    - GeoPandas, ML (Tom Augspurger, Jim Crist)
> 2. Asynchronous Algo
>    - Parameter server style algo (GLM)

- Advanced scheduler on local mahcine
  1. get diagnostics visualisation via Bokeh
  2. get new features (e.g. .persist())
  3. scale out if necessary
  4. almost always more efficient than multiprocessing scheduler
- Ligthweight
  1. Worker stepup, task submission, result retrieval, shutdown:

```python
%%time
with Client(processes=False) as client:
    future = client.submit(lambda x: x + 1, 10)
    print(future.result())
```

### Customised Programme with Dask (Example)

#### Futures API

```python
# start a local clsuter
from dask.distributed import Client
client = Client()

# Submit single task to run in background 
# Worker runs add(1,2), stores restul in local RAM
from operator import add
future = client.submit(add, 1, 2) 

# Learn about status asynchronously
future # status: finished

# block and gather result
future.result()
```

#### Track dependencies ON-THE-FLY

```python
x = client.sumbit(f, 1)
y = client.submit(f, 2)
z = client.submit(g, x, y) # submit task on futures

# updates happen in background
futures = [client.submit(f, x) for x in L]

# manipulate computations on the fly
# submit new tasks during execution
# even while previous tasks still flying
finished = [future for future in futures if future.status == 'finished']
results = client.gather(finished)
new_futures = [client.submit(g,x) for x in ...]
```

#### Convenient methods exist to support asynchronous workloads

```python
# iterate over futures as they complete 
# part of standard concurrent.futures API
# Quit early if having a good enough result 
# cancel remaining work
from dask.distributed import as_completed

future = [client.sumbit(func, *args) for x in L]

iterator = as_completed(futures)

best = 0
for future in iterators:
    result = future.result()
    best = max(best, result)
    if best > 100: 
        break
        
client.cancel(iterator.futures)

# Or continue submit more tasks 
# add to iterator 
# simple way to create asynchronous iterative algo
total = 0
for future in iterators:
    result = future.result()
    total += result
    if result > 10:
        a = client.submit(func, ...) # submit more work
        b = client.submit(func, ...)
        iterator.add(a) # add to iterator
        iterator.add(b)
        
# EX: computational
client = Client('localhost:8766', timeout=1000)
client

def rosenbrock(point):
    """compute rosenbrock func and return point minimal"""
    time.sleep(0.1)
    score = (1 - point[0])**2 + 2 * (point[1] - point[0]**2)
    return point, score

scale = 5 # initial random perturbation
best_point = (1, 2) # best point so far
best_score = float('inf')

initial = [(random.uniform(-5, 5), random.uniform(-5, 5))
           for i in range(10)]

futures = [client.submit(rosenbrock, point) for point in initial]

running = as_completed(futures)

for res in running:
    point, score = res.result()
    if score < best_score:
        best_score, best_point = score, point
        print("Current best (%.3f, %.3f)."
              "Scale: %.3f" % (best_point + (scale,)))
        
    x, y = best_point
    new_point = (x + random.uniform(-scale, scale),
                 y + random.uniform(-scale, scale))
    new_point = client.submit(rosenbrock, new_point)
    
    running.add(new_point)
    
    scale *= 0.99
    
    if scale < 0.001:
        break
```

#### Worker Starts Client/Scheduler!

- Submit tasks from tasks

```python 
# tasks can get their own client 
# Remote client controls cluster
# Task-on-worker can do anything you can do locally

from dask.distributed import get_client, get_worker, secede, fire_and_forget

def func(...):
    client = get_client()
    futures = [client.submit(...) for ...]
    results = client.gather(futures)
    return sum(results)

future = client.submit(func, ...)

# EX: fibonnacii
def fib(n):
    if n == 0 or n == 1:
        return n
    else:
        client = get_client()
        a = client.submit(fib, n - 1)
        b = client.submit(fib, n - 2)
        return a.result() + b.result()
    
future = client.submit(fib, 1000)
```

- Multi-client coordination

```python
# multiple clients, communicating

# multi-producer/consumer queue
# send along small data for futures
from dask.distributed import Queue
q = Queue()
future = client.scatter(my_numpy_array)
q.put(123)
x = q.get()

# Global singleton value
# send along small data or futures
from dask.distributed import Variable
v = Variable()
future = client.scatter(my_numpy_array)
v.set(123)
x = v.get()
```

- Multi-consumer Multi-producer system

```python
# Workers start clients
# Tasks can submit more tasks
# can do anything you can do locally
def producer():
    client = get_client()
    while not stop.get():
        data = get_data()
        future = client.scatter(data)
        q.put(future)
        
def consumer():
    client = get_client()
    while not stop.get():
        future = q.get()
        data = future.result()
        # do stuff with data
q = Queue()
stop = Variable()
stop.set(False)

producers = [client.submit(producer, ...) for i in range(n)]
consumers = [client.submit(consumer, ...) for i in range(m)]
```

- Fully async await compliant

```python
# support async/await syntax
# support Tornado and AsyncIO event loops
async def f():
    client = await Client(asynchronous=True)
    
    futures = [client.submit(f, x) for x in L]
    async for future in as_completed(futures):
        result = await future
        # do things with result
```

- Specify resource constriants like RAM or GPU

```python
# specify resources contrs
# Good for GPU high RAM tasks
dask-worker ... --resources "GPU=2 FOO=1"
dask-worker ... --resources "GPU=1 MEMORY=100e9"

future = client.submit(func, x, resources={'GPU': 1})
future = client.submit(func, x, resources={'MEMORY': 60e9})
```

# Jim Crist Talks <a id="jim"></a>

### Parallel NumPy and Pandas through Task Scheduling

> Collections -> Graphs -> Schedulers
>
> 1. Collections (array, bag, dataframe, imperative)
> 2. Graphs
> 3. Schedulers (synchronous, threaded, multiprocessing, distributed)

> Collections build task graphs -> Schedulers execute task graphs -> Graph specification = uniting interface

> Dask Specification
>
> 1. Dictionary of {name: task}
> 2. Tasks are tuples of (func, args...) (lispy syntax)
> 3. Args can be names, values, or tasks

> Decoupling between Collections/Graphing/Scheduler makes possible to creating graph directly to problem

```python
def load(filename):
    pass
def clean(data):
    pass
def analyze(sequence_of_data):
    pass
def store(result):
    with open(filename, 'w') as f:
        f.write(result)
        
dsk = {'load-1': (load, 'myfile.a.data'),
       'load-2': (load, 'myfile.b.data'),
       'load-3': (load, 'myfile.c.data'),
       'clean-1': (clean, 'load-1'),
       'clean-2': (clean, 'load-2'),
       'clean-3': (clean, 'load-3'),
       'analyze': (analyze, ['clean-%d' % i for i in [1, 2, 3]]),
       'store': (store, 'analyze')}

# Alternatively: dask.imperative
@do
def load(filename):
    pass
@do
def clean(data):
    pass
@do
def analyze(sequence_of_data):
    pass
@do
def store(result):
    with open(filename, 'w') as f:
        f.write(result)

files = ['myfile.a.data',...]
loaded = [load(f) for f in files]
cleaned = [clean(i) for i in loaded]
analyzed = analyze(cleaned)
stored = store(analyze)
```



- Example - dask.bag (any non-NumPy, non-DataFrame, collections)

```python
import dask.bag as db

b = db.from_castra('reddit.castra' columns=['subreddit', 'body'],
                   npartitions=8)
matches_subreddit = b.filter(lambda x: x[0] == 'MachineLearning')
words = matches_subreddit.pluck(1).map(to_words).concat()
top_words = words.frequencies().topk(100, key=1).compute()

from wordcloud import WordCloud

wc = WordCloud()
wc = generate_from_frequencies(top_words)
wc.to_image()
```

### DataFrames on Cluster

```python
from gcsfs import GCSFileSystem
gcs = GCSFileSystem(token='cloud')

gcs.ls('path/to/csv')

import dask.dataframe as dd

df = dd.read_csv('gcs://path/to/csv' parse_dates=['datecolumns'], storage_options={'token':'cloud'})
df = client.persis(df)
progress(df)
```

#### Parallelise Normal Python code

```python
%%time
zs = []
for i in range(256):
    x = inc(i)
    y = dec(x)
    z = add(x, y)
    zs.append(z)
    
zs = dask.persist(*zs)
total = dask.delayed(sum)(zs)

total.compute()

# example
futures = client.map(lambda x: x + 1, range(1000))
total = client.submit(sum, futures)
total.result()
```

## Long SciPy Tutorial 2017 <a id="long"></a>

```python
# METHODS / ATTRIBUTES ACCESS ON DELAYED
# ALL TOUCHED DELAYED :D

from dask import delayed, compute

x = delayed(np.arange)(10)
y = (x + 1)[::2].sum()

y.visualize(color="order", rankdir="LR")

# .COMPUTE() for single, COMPUTE() for multiple output

min, max = compute(y.min(), y.max())

min,max # sharing mid values (y = f(x))
```



![png](/Users/Ocean/Desktop/DASK/output_15_0.png)





```
(25, 25)
```



> BEST to dask-LOAD data instead of dask it after

`df = delayed(pd.read_csv)(file)`

### Dask.dataframe

- `dask.dataframe.read_csv` only reads first lines of first file 
- MAY incur dtype errors if missing
- SOLUTION
  1. specify dtype 
  2. `assume_missing` to make dask assume col to be int (which disallow missing) are in fact floats (which allows)

```python
dd.read_csv(filename, ... dtype={'col1': str, 'col2': float, 'col3': bool})
```

**Just use normal Pandas syntax!**
`df.Column.max().compute()`

**See divisions**

```python
df2 = df.set_index('Year')
df2.divisions # (1990,...)
df2.npartitions
```

**Custom Code and DD**

> EX, previously `to_timestamp` not emulated, or need for custom operations

- wrapper: `map_partitions, map_overlap, reduction`

```python
# individual wrapping
hours = df.Time // 100
hours_timedelta = hours.map_partitions(pd.to_timedelta, unit='h')
mins = df.Time % 100
mins_timedelta = mins.map_partitions(pd.to_timedelta, unit='m')

timestamp = df.Date + hours_timedelta + mins_timedelta

# functional wrapping
def compute_timestamp(df):
    hours = df.Time // 100
    hours_timedelta = pd.to_timedelta(hours, unit='h')
    mins = df.Time % 100
    mins_timedelta = d.to_timedelta(minutes, unit='m')
    return df.Date + hours_timedelta + mins_timedelta
timestamp = df.map_partitions(compute_timestamp)
```

**Dask.array / Dask.stack**

```python
import h5py
from glob import glob
import os

# generate dataset by prep 
filenames = sorted(glob(os.path.join('data', 'weather-big', '*.hdf5')))
dsets = [h5py.File(filename, mode='r')['/t2m'] for fileanme in filenames]

arrays = [da.from_array(dset, chunks=(500,500)) for dset in dsets]

x = da.stack(arrays, axis=0)

result = x.mean(axis=0)

fig = plt.figure(figsize=(16,8))
plt.imshow(result, cmap='RdBu_r')
```

**BAG**

```python
# generating JSON from prep

import dask.bag as db

b = db.from_sequence([1, 2, 3])

b = db.read_text(os.path.join('data', 'account.*.json.gz'))
b.npartitions

c = (b.filter(iseven).map(lambda x: x**2)) 
c.compute()

# read json.gz as lines
lines.take(1) # head()-like

js = lines.map(json.loads) # disk-loaded as json
js.take(1)

# Queries
js.filter(lambda record: record['name'] == 'Alice').take(5)

def count_trans(d):
    return {'name': d['name'],
            'count': len(d['transacitons'])}
(js.filter(lambda record: record['name'] == 'Alice').map(count_trans).take(5))

# pluck: select a field, as from dict, element[filed]
(js.filter(lambda record: record['name'] == 'Alice').map(count_trans).pluck('count').take(5))

# flatten to de-nest
(js.filter(lambda record: record['name'] == 'Alice').map(count_trans).flatten().pluck('amount').mean().compute())

# use foldby instead of groupby (do on DF)
# need {key func to group, binary ops passed to reduce per group, combine binary ops on results of two reduce calls on diff parts}
# Reduction must be associative
(b.foldby(lambda x: x% 2,
          binop=lambda acc, x: acc if acc > x else x,
          combine=lambda acc1, acc2: acc1 if acc1 > acc2 else acc2).compute())

# Example finding # people sharing name
(js.foldby(key='name',
           binop=lambda total, x : total + 1,
           initial=0,
           combine=lambda a, b : a + b,
           combine_initial=0)
 .compute())
```

**Diagnostics**

```python
# Aid in profiling parallel execution seeing bottleneck
from dask.diagnostics import Profiler, ResourceProfiler, visualize
from bokeh.io import output_notebook
output_notebook()

with Profiler() as p, ResourceProfiler(0.25) as r:
    largest_delay.computet()
visualize([r, p]);

# while tasks running, GIL restrict parallelism during early pd.read_csv (mostly byte ops)
# NOTE: diagnostics ONLY useful profiling SINGLE MACHINE - dask.distributed scheduler 8787!
```

**SCHEDULER: (1) threaded - thread-pool, multi-processing on thread-pool; (2) serial - single-thread good for debugging; (3) distributed - multi-node or local**

> #### Client() default-creating ONE WORKER PER CORE

```python
client = Client()

%time _ = largest_delay.compute(get=client.get)

# WHY this FASTER than THREADED scheduler ??
# INFACT no need `get=client.get` as distributed scheduler takes over as default scheduler for all collections when Client created

# Locally
from dask.distributed import LocalCluster

client = Client(LocalCluster(n_workers=8))
```



> ### Cluster Creation: Cluster-specific DASK-CLI available (ES2, Kubernetes)

> #### dask-worker `schedulerIP` will default SINGLE WORKER PROCESS with #THREADS AS CORES AVAILABLE

> ### ACCESSING SAME DATASET among WORKERS
>
> - S3 storage - DASK.DF support reading directly from S3

```python
columns = ['Year', etc as need]

df = dd.read_csv('gcs://filepath/199*.csv',
            parse_dates={'Date': [0,1,2]},
            dtype={'col1': object, ...}
            usecols=columns,
            storage_options={'token': 'cloud'})
```

**Persist on RAM for processing datasets**

- Bytes stored on Diagnostic Page show RAM usage 
- Fast pandas ops as IN-MEMORY
- How large each partition? `df.map_partitions(pd.DataFrame.memory_usage().sum()).compute()`

**INDEXING is KEY**

> Many DF ops (loc-indexing, groupby-apply, joins) MUCH faster on sorted index; e.g. knowing WHICH part of dataset to compute, else needing to SEARCH FULL

> Pandas model has sorted index column, Dask.df copies it and knowns min-max values of each partition's index (DEFAULT NO INDEX)

> #### HOWEVER: if setting `Date` column as index then FASTER - calling `set_index` + `persist` => new set of DF partitions stored IN-MEM, sorted along index col - DASK shuffles data by date, set index per partition, store in cluster-MEM

> ### Relatively COSTLY - but gain certain query-speed

```python
df = df.set_index('Date').persist()

# Now KNOWN Divisions !!
df.npartiions
df.known_divisions # True
df.divisions # output names of partitions
df.loc['1992-05-05'].compute() # now FAST
df.loc['1992-05-05'].visualize(optimize_graph=True) # only looking at single partition instead FULL SEARCH
```

> ### TIME SERIES INDEX: `DatetimeIndex` pandas are supported 

```python
%matplotlib inline

(df.DepDelay
 .resample('1M')
 .mean()
 .fillna(method='ffill')
 .compute()
 .plot(figsize=(10,5)))
```

## DISTRIBUTED FEATURES

- `CLIENT.SUBMIT` takes (FUNC, *ARG) applying on CLUSTER -> returning FUTURES representing result to be computed

- `repr` reveals 'pending' since ASYNCHRONOUS, can do other ops while it computes (if wait until completed use `wait(result)`) - BLOCK till completed

- `future.result()` pull data out back to local disk

- `client.gather(future1, future2...)` pull back many futures

- **ALTERNATIVE way to exec work on cluster - submit/map with input as future, computation moves to data rather than other way around, and client (local python session) need never see the middle values - similar to building graph by `delayed` which can be used here with futures** :

  ```python
  x = delayed(inc)(1)
  total = delayed(add)(x, y)
  
  fut = client.compute(total)
  fut
  fut.result() # passing compu to cluster while freeing machine to do other works !!
  
  # convert to submit()
  x = client.submit(inc, 1)
  total = client.submit(add, x, y)
  
  print(total) # a future
  total.result() # blocks untill compu done
  ```

  > #### NOTE difference: total.compute() completes immediately

- **Future API emulates map/reduce (client.map()) - middle results as futures can be passed to new tasks WITHOUT having to return to local from cluster - new work assigned using output of previous jobs not started yet !!**

> **En general, any dask ops executed using `.compute()` can be submitted for ASYNC execution using `client.compute()` instead, applied to all collections: (here async enables continuous submission of works (perhaps based on result of calculation), or follow the progress of computation**

```
​```python
import dask.bag as db

res = (db.from_sequence(range(10))
    .map(inc)
    .filter(lambda x : x % 2 == 0)
    .sum())

f = client.compute(res)

# progress must be last line of cell to show up
progress(f)

client.gather(f)
​```
```

- Asynchronous Computation: one benefit of it enables DYNAMIC COMPUTATION adjusting as things progress (naive search by looping results as stream, submit new points to compute as others are running) USEFUL FOR PARALLEL ALGO requiring some level of SYNCHRONISATION

## Two Easy Ways to SKL+DASK

1. Use Dask Joblib backend
2. Use dklearn projects drop-in repalcements for `Pipeline, GridSearchCV, RndomSearchCV`

```python
#Joblib

from joblib import parallel_backend
with parallel_backend('dask.distributed', scheduler_host='scheduler-address:8786'):
    # your now-clustered sklearn code here
    
# Dask-learn pipeline repalcement

from dklearn.grid_search import GridSearchCV
from dklearn.pipeline import Pipeline
```

- Neither perfect - but easiest to try

### Joblib

- SKL already paral across-cores using joblib, extensible map operation
- If extending Joblib to clusters then adding parallelism from joblib-enabled SKL at once

```python
# sequential code demo joblib
from time import sleep
def slowinc(x):
    sleep(1)
    return x + 1

%timeit [slowinc(i) for i in range(10)]

# parallel code
from joblib import Parallel, delayed

%timeit Parallel(n_jobs=-1)(delayed(slowinc)(i) for i in range(10))
```

```
10 s ± 7.36 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
5.03 s ± 3.65 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

### Distributed Joblib

- API for other parallel systems to step in acting as execution engine - `parallel_backend` context manager to run with hundres or thousands of cores in nearby cluster
- Main value for SKL users is that SKL already uses `joblib.Parallel` inside - e.g. `n_jobs` or using JOBLIB together with `Dask.distributed` to parallelise across multi-node cluster

```python
from dask.distributed import Client
client = Client()

from sklearn.externals import joblib

with joblib.parallel_backend('dask'): # scheduler_host='scheduler-address:8786'):
    print(Parallel()(delayed(slowinc)(i) for i in list(range(100))))
```

```
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
```



```python
with joblib.parallel_backend('dask'):
    estimator = GridSearchCV(...) # use joblib with Dask cluster
```

### Limitations

- From Dask's view JL not ideal - it always collect middle results back to main process instead of leaving them on cluster until needed - still, given wide use of Joblib-enabled workflows (particularly within SKL) this is a simple thing t otry if haing cluster nearby with a possible large payoff

### Dask-Learn Pipeline and GridSearch

- Dask variants of SKL Pipeline, GSCV and RandomSCV better handle nested parallelism
- so if replace following imports may get both better single-threaded performance AND the ability to scale out to cluster

```python
# full example

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000,
                           n_features=500,
                           n_classes=2,
                           n_redundant=250,
                           random_state=42)

from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from dklearn.pipeline import Pipeline

logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca),
                       ('logistic', logistic)])


#Parameters of pipelines can be set using ‘__’ separated parameter names:
grid = dict(pca__n_components=[50, 100, 150, 250],
            logistic__C=[1e-4, 1.0, 10, 1e4],
            logistic__penalty=['l1', 'l2'])

# from sklearn.grid_search import GridSearchCV
from dklearn.grid_search import GridSearchCV

estimator = GridSearchCV(pipe, grid)

estimator.fit(X, y)
```

- SKL performs this ~ 40s while Dask ML drop-in ~10sec, also if adding followling lines to connect to running cluster the scaling

> Quickstart on Dask.distributed

```python
pip install dask distributed --upgrade
from dask.distributed import Client
client = Client() # set up local cluster on machine
client # info on scheduler app and process/core
# OR setup hard way using multi-workers
# on shell CLI
$ dask-scheduler
# dask-worker 127.0.0.1:8786
client = Client('127.0.0.1:8786')
# Map and Submit Func
A = client.map(square, range(10))
B = client.map(neg, A)
total = client.submit(sum, B)
total.result()
# Gather
total # function yet completed
total.result() # result for single future
client.gather(A) # gather for many futures
client.restart() # run at error
```

### Better

- [Dask and SKL - Parallelism](http://jcrist.github.io/dask-sklearn-part-1.html)
- Joblib + Dask.distributed is easy but leaves some speed on table - not clear how ask can help SKL codebase without being too invasive

## Convex Optimisation Algo with Dask

> Many ML models depend on Convex Optimisaiotn alog like Newton's method, SGD and others - both pgramatic and mathy - bridging math and distributed system; 

### Prototyping Algo in Dask

- Choices
  1. Parallel multi-dimensional ARRAY to const algo from common ops like matrix multiplication, SVD, etc - mirroring math-algo but lacks flexibility
  2. Create algo by hand tracking ops on each chunks of in-RAM data and dependencies 

### Example - fitting large LM using array parallelism and customised from Dask

```python
import dask
import dask.array as da
import numpy as np

from dask.distributed import Client

client = Client()

## create inputs with a bunch of independent normals
beta = np.random.random(100)  # random beta coefficients, no intercept
X = da.random.normal(0, 1, size=(1000000, 100), chunks=(100000, 100))
y = X.dot(beta) + da.random.normal(0, 1, size=1000000, chunks=(100000,))

## make sure all chunks are ~equally sized
X, y = dask.persist(X, y)
client.rebalance([X, y])
```

> X is dask array on 10 chunks each (100000,100) and X.dot(beta) runs smoothly for both mnumpy and dask.array, so able to write code working in either world

**Caveat** 0 if X is numpy array and beta is dask.array, X.dot(beta) will ouput RAM numpy array, often not desirable - FIX by `multipledispathch` to handle odd ege cases

### Array Programming

- if you can write iterative array-based algo in Numpy, then able to write iterative parallel algo in Dask
- e.g. computing beta* from normal equation

[FULL ARTICLE](http://matthewrocklin.com/blog/work/2017/03/22/dask-glm-1)

# DataCamp <a id="datacamp"></a>

# Working with BIG DATA

- Data > one machine 
- Kilo-Mega-Giga-Tera-...
  - bit-2^3 (byte)...

## Time and Bit

- Scaled to RAM = 1s
  - SSD = 7-21 min
  - Rotational Disk = 2.5hr - 1day
  - Internet (SF-NY) = 3.9days

## Querying Python interpreter's Memory Usage

- below in code as example

```python
import psutil, os

def memory_footprint():
    '''Returns memory (MB) being used by Python process'''
    mem = psutil.Process(os.getpid()).memory_info().rss
    return (mem / 1024**2)

before = memory_footprint()

N = (1024**2) // 8 # Number of floats filling 1 MB

x = np.random.randn(50*N) # Random array filling 50 MB

after = memory_footprint()

print('Memory before: {} MB'.format(before))
print('Memory after: {} MB'.format(after))

x.nbytes
x.nbytes // (1024**2)

df = pd.DataFrame(x)

df.memory_usage(index=False)
```

```
Memory before: 154.4765625 MB
Memory after: 204.66015625 MB
```





```
52428800
```





```
50
```





```
0    52428800
dtype: int64
```



## Think data in CHUNKs

- Load and preprocess (filter etc) in chunks
- Memory used per chunk sequentially at a time

```python
# Filtering WDI data in chunks

# Load CSV from zip url using requests, zipfile, io libraries !!

import requests, zipfile, io

dfs = []

req = requests.get('https://assets.datacamp.com/production/course_4299/datasets/WDI.zip')
zip = zipfile.ZipFile(io.BytesIO(req.content))
zip.filelist

# filter in chunks then concatenate as one df
for chunk in pd.read_csv(zip.open('WDI.csv'), chunksize=1000):
    is_urban = chunk['Indicator Name']=='Urban population (% of total)'
    is_AUS = chunk['Country Code']=='AUS'
    filtered = chunk.loc[is_urban & is_AUS]
    dfs.append(filtered)
pd.concat(dfs)
```



```
[<ZipInfo filename='WDI.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=10590570 compress_size=1140029>,
 <ZipInfo filename='__MACOSX/' filemode='drwxrwxr-x' external_attr=0x4000>,
 <ZipInfo filename='__MACOSX/._WDI.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=526 compress_size=326>]
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country Name</th>
      <th>Country Code</th>
      <th>Indicator Name</th>
      <th>Indicator Code</th>
      <th>Year</th>
      <th>value</th>
      <th>Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>875</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1980</td>
      <td>85.760</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>950</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1981</td>
      <td>85.700</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1026</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1982</td>
      <td>85.640</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1983</td>
      <td>85.580</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1176</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1984</td>
      <td>85.520</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1251</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1985</td>
      <td>85.460</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1328</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1986</td>
      <td>85.400</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1404</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1987</td>
      <td>85.400</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1479</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1988</td>
      <td>85.400</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1554</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1989</td>
      <td>85.400</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1640</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1990</td>
      <td>85.400</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1717</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1991</td>
      <td>85.400</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1796</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1992</td>
      <td>85.566</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1873</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1993</td>
      <td>85.748</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1950</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1994</td>
      <td>85.928</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>2029</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1995</td>
      <td>86.106</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>2107</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1996</td>
      <td>86.283</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>2186</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1997</td>
      <td>86.504</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>2264</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1998</td>
      <td>86.727</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>2341</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>1999</td>
      <td>86.947</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>2428</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2000</td>
      <td>87.165</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>2506</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2001</td>
      <td>87.378</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>2591</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2002</td>
      <td>87.541</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>2671</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2003</td>
      <td>87.695</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>2752</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2004</td>
      <td>87.849</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>2834</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2005</td>
      <td>88.000</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>2918</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2006</td>
      <td>88.150</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>3001</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2007</td>
      <td>88.298</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>3085</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2008</td>
      <td>88.445</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>3168</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2009</td>
      <td>88.590</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>3259</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2010</td>
      <td>88.733</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>3339</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2011</td>
      <td>88.875</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>3420</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2012</td>
      <td>89.015</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>3499</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2013</td>
      <td>89.153</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>3575</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2014</td>
      <td>89.289</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>3640</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Urban population (% of total)</td>
      <td>SP.URB.TOTL.IN.ZS</td>
      <td>2015</td>
      <td>89.423</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
  </tbody>
</table>

</div>



### Managing Data with Generators

- def filter function
- apply using list_comprehension

```chunks = [filter_func(chunk) for chunk in pd.read_csv(filename, chunksize=1000)]```

- Instead of list-comp, lazy evaluation method of generator saves memory

``` chunks = (filter_func(chunk) for chunk in pd.read_csv(filename, chunksize=1000))```

```
- yield on run, one at a time
```

~~~sum = (chunk['feature'].sum() for chunk in chunks)
    sum(sum)```
    - only when used will gen

### Load multiple files via Generator

```template = 'filename_2015-{:02d}.csv'```
    - string formating expression
```filenames = (template.format(k) for k in range(1,13))```
    - each item in filenames now yield 'names containing date from 01 to 12'


```python
# Read multiple files in zip

req = requests.get('https://assets.datacamp.com/production/course_4299/datasets/flightdelays.zip', stream=True)
zip = zipfile.ZipFile(io.BytesIO(req.content))
zip.namelist

~~~



```
[<ZipInfo filename='flightdelays/' filemode='drwxr-xr-x' external_attr=0x4000>,
 <ZipInfo filename='flightdelays/flightdelays-2016-4.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=10901697 compress_size=1737653>,
 <ZipInfo filename='__MACOSX/' filemode='drwxrwxr-x' external_attr=0x4000>,
 <ZipInfo filename='__MACOSX/flightdelays/' filemode='drwxrwxr-x' external_attr=0x4000>,
 <ZipInfo filename='__MACOSX/flightdelays/._flightdelays-2016-4.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=2083 compress_size=1451>,
 <ZipInfo filename='flightdelays/flightdelays-2016-5.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=11342052 compress_size=1820857>,
 <ZipInfo filename='__MACOSX/flightdelays/._flightdelays-2016-5.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=2083 compress_size=1451>,
 <ZipInfo filename='flightdelays/flightdelays-2016-2.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=10014549 compress_size=1601161>,
 <ZipInfo filename='__MACOSX/flightdelays/._flightdelays-2016-2.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=2083 compress_size=1451>,
 <ZipInfo filename='flightdelays/flightdelays-2016-3.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=11357646 compress_size=1835871>,
 <ZipInfo filename='__MACOSX/flightdelays/._flightdelays-2016-3.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=2083 compress_size=1451>,
 <ZipInfo filename='flightdelays/flightdelays-2016-1.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=10546302 compress_size=1699366>,
 <ZipInfo filename='__MACOSX/flightdelays/._flightdelays-2016-1.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=2083 compress_size=1451>]
```



```python
# Flight delay case

# func for % delayed
def pct_delayed(df):
    n_delayed = (df['DEP_DELAY']>0).sum() 
    return n_delayed  * 100 / (len(df))

# Make file-list from above zip object
filenames = ['flightdelays/flightdelays-2016-{:01d}.csv'.format(k) for k in range(1,6)]

dataframes = (pd.read_csv(zip.open(file)) for file in filenames)

monthly_delayed = [pct_delayed(df) for df in dataframes]
```

## Generator for delaying computing and saving memory usage: DASK to simplify

```python
from dask.delayed import delayed

def func(x):
    return sqrt(x + 4)

func = delayed(func)

type(func)

# using Decorator @ to combine above 2 cells
@delayed
def func(x):
    return sqrt(x+4)

type(func)
```



```
dask.delayed.DelayedLeaf
```





```
dask.delayed.DelayedLeaf
```



## Visualising complex dependency loops / computations

```python
# Make 3 @delayed func

@delayed
def increment(x):
    return x+1
@delayed
def double(x):
    return 2*x
@delayed
def add(x,y):
    return x+y

data = [1, 2, 3, 4, 5]

output = []

for x in data:
    a = increment(x)
    b = double(x)
    c = add(a,b)
    output.append(c)

total = sum(output)

total
output

total.visualize()
```



```
Delayed('add-58eba218d09a0bd7b2482817167c0184')
```





```
[Delayed('add-9715190e-f684-4214-9062-707a45773e27'),
 Delayed('add-c4e48f00-0a10-4da9-a9ff-453d8867d781'),
 Delayed('add-0506bbf8-17c2-4960-9e51-376de9fbaefc'),
 Delayed('add-c26bf38a-c1c2-4015-86ac-6984dd2c58e6'),
 Delayed('add-7c1f8287-53bb-4951-97b4-106bf3445149')]
```





![png](/Users/Ocean/Desktop/DASK/output_37_2.png)



```python
# Request zip file URL
req = requests.get('https://assets.datacamp.com/production/course_4299/datasets/nyctaxi.zip', stream=True)
zip = zipfile.ZipFile(io.BytesIO(req.content))
zip.filelist
```



```
[<ZipInfo filename='nyctaxi/' filemode='drwxr-xr-x' external_attr=0x4000>,
 <ZipInfo filename='nyctaxi/yellow_tripdata_2015-03.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=9755811 compress_size=2598820>,
 <ZipInfo filename='__MACOSX/' filemode='drwxrwxr-x' external_attr=0x4000>,
 <ZipInfo filename='__MACOSX/nyctaxi/' filemode='drwxrwxr-x' external_attr=0x4000>,
 <ZipInfo filename='__MACOSX/nyctaxi/._yellow_tripdata_2015-03.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=2091 compress_size=1449>,
 <ZipInfo filename='nyctaxi/.DS_Store' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=6148 compress_size=178>,
 <ZipInfo filename='__MACOSX/nyctaxi/._.DS_Store' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=120 compress_size=53>,
 <ZipInfo filename='nyctaxi/yellow_tripdata_2015-02.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=9983787 compress_size=2633156>,
 <ZipInfo filename='__MACOSX/nyctaxi/._yellow_tripdata_2015-02.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=2091 compress_size=1450>,
 <ZipInfo filename='nyctaxi/yellow_tripdata_2015-01.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=12480085 compress_size=3377085>,
 <ZipInfo filename='__MACOSX/nyctaxi/._yellow_tripdata_2015-01.csv' compress_type=deflate filemode='-rw-r--r--' external_attr=0x4000 file_size=572 compress_size=335>]
```



```python
# Example using cab delay data

filenames_cab = ['nyctaxi/yellow_tripdata_2015-{:02d}.csv'.format(k) for k in range(1,4)]

@delayed
def long_trips(df):
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.seconds
    is_long_trip = df.duration > 1200
    result_dict = {'n_long':[sum(is_long_trip)],
                  'n_total':[len(df)]}
    return pd.DataFrame(result_dict)

@delayed
# RECALL to add zip.open() in this case for URL
def read_file(fname):
    return pd.read_csv(zip.open(fname), parse_dates=[1,2])

# Make Totals object combining above two func Read and Slice
totals = [long_trips(read_file(fname)) for fname in filenames_cab]

annual_totals = sum(totals)

# delayed_object.compute() ONLY called everything here !!
annual_totals = annual_totals.compute() 

print(annual_totals['n_long'] / annual_totals['n_total'])
```

```
0    0.175269
dtype: float64
```

## Dask Arrays and Chunking

- Extending Numpy Array
- Share many Numpy methods/attributes
  - shape, ndim, nbytes, dtype, size etc
  - max, min, mean, std, var, sum, prod
  - reshape, repeat, stack, flatten, transpose
  - round, real, imag, conj, dot

### Dask scheduler auto-assign multiple processors/threads concurrently available !

```python
a = np.random.rand(10000)

import dask.array as da


# quick ex
x = da.random.random()
y = x.dot(x.T) - x.mean(axis=0)


# Convert Numpy Array to Dask Array with chunking
a_dask = da.from_array(a, chunks=len(a)//4)

# View DaskArray chunks
a_dask.chunks

n_chunks = 4

chunk_size = len(a) // n_chunks

result = 0

# Comparing chunk size between Numpy and Dask
# In Numpy
for k in range(n_chunks):
    offset = k * chunk_size # track offset explicitly
    a_chunk = a[offset:offset + chunk_size] # slice chunk
    result += a_chunk.sum()

print(result)
```



```
((2500, 2500, 2500, 2500),)
```



```
5050.320824178792
```



```python
# Redo with Dask Array

result = a_dask.sum()

# dask array automates slicing
result 

result.compute()

# Visualise its task-graph
result.visualize(rankdir="LR") # rankdir forces 'left-right horizontal layout'
```



```
dask.array<sum-aggregate, shape=(), dtype=float64, chunksize=()>
```





```
5050.320824178792
```





![png](/Users/Ocean/Desktop/DASK/output_42_2.png)



```python
# Test timing of Array Computation with h5py file

import h5py, time

# req = requests.get('https://ndownloader.figshare.com/files/7024985')
# with open('sample.hd5f', 'wb') as f:
#     f.write(req.content)
    
with h5py.File('texas.hdf5', 'r') as dset:
    list(dset.keys())
    dist = dset['load'][:]
    
dist_dask8 = da.from_array(dist, chunks=dist.shape[0]//8)

# Chaining creation to leave no gap
time_start = time.time(); \
mean8 = dist_dask8.mean().compute(); \
time_end = time.time();

time_elapsed = (time_end - time_start) * 1000 # miliseconds

print(f'Elapsed time: {time_elapsed} ms')
```



```
['load']
```



```
Elapsed time: 44.425249099731445 ms
```

## Multi-Dimension Array Wrangling ~ similar to Numpy

```python
req = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00445/Absenteeism_at_work_AAA.zip')
zip = zipfile.ZipFile(io.BytesIO(req.content))
zip.filelist

temp = zip.open('Absenteeism_at_work.csv')
temp
```



```
[<ZipInfo filename='Absenteeism_at_work.arff' compress_type=deflate external_attr=0x20 file_size=91190 compress_size=8478>,
 <ZipInfo filename='Absenteeism_at_work.csv' compress_type=deflate external_attr=0x20 file_size=45232 compress_size=6822>,
 <ZipInfo filename='Absenteeism_at_work.xls' compress_type=deflate external_attr=0x20 file_size=141824 compress_size=17245>,
 <ZipInfo filename='Attribute Information.docx' compress_type=deflate external_attr=0x20 file_size=13429 compress_size=10719>,
 <ZipInfo filename='UCI_ABS_TEXT.docx' compress_type=deflate external_attr=0x20 file_size=44114 compress_size=22064>]
```





```
<zipfile.ZipExtFile name='Absenteeism_at_work.csv' mode='r' compress_type=deflate>
```



```python
# Load csv into Numpy

data = np.loadtxt(zip.open('Absenteeism_at_work.csv'), delimiter=';', skiprows=1, usecols=(1,2,3,4), dtype=np.int64)

data.shape
type(data)
```



```
(740, 4)
```





```
numpy.ndarray
```



```python
data_dask = da.from_array(data, chunks=(740,2))

result = data_dask.std(axis=0)

result.compute() # deferred computation call
```



```
array([8.42770571, 3.43396433, 1.42071379, 1.11107957])
```



```python
# Read hdf5 file into Numpy
electricity = h5py.File('texas.hdf5', 'r')
electricity = electricity['load'][:]

type(electricity)
electricity.shape
```



```
numpy.ndarray
```





```
(35136,)
```



```python
# This time-series array is flat comprising 3 years of grid data in 15-min interval
# Converting to multi-array as (year, day, 15-min)
electricity_3d = electricity.reshape((1, 365))
```

```
---------------------------------------------------------------------------

ValueError                                Traceback (most recent call last)

<ipython-input-118-b62cc72f6543> in <module>
      1 # This time-series array is flat comprising 3 years of grid data in 15-min interval
      2 # Converting to multi-array as (year, day, 15-min)
----> 3 electricity_3d = electricity.reshape((1, 365))
```

```
ValueError: cannot reshape array of size 35136 into shape (1,365)
```

### Wrangling with Arrays

- array.reshape((d1,d2,d3)) in the context of data
- array algebra **along specific dimension**
- e.g. max number of 2nd and 3rd dimension ```array.max(axis=(1,2))```

#### Sample Code

```python
# Import h5py and dask.array
import h5py
import dask.array

# List comprehension to read each file: dsets
dsets = [h5py.File(f)['tmax'] for f in filenames]

# List comprehension to make dask arrays: monthly
monthly = [dask.array.from_array(d, chunks=(1,444,922)) for d in dsets]
```

- monthly comprises 4 solitary dask.array with original shape (12,444,922) chunked by (1,444,922), equating to 12 chunks per dask.array
- Then to **STACK** them as one ```dask.stack(array, axis=0)``` row-wise ! Resulting dimension is (4, 12, 444, 922) in total and chunked as 4x12 (1,1,444,922)

```python
# Stack with the list of dask arrays: by_year
by_year = da.stack(monthly, axis=0)

# Print the shape of the stacked arrays
print(by_year.shape)

# Read the climatology data: climatology
dset = h5py.File('tmax.climate.hdf5')
climatology = da.from_array(dset['/tmax'], chunks=(1,444,922))

# Reshape the climatology data to be compatible with months
climatology = climatology.reshape(1,12,444,922)
```

- Further slicing with dask.array.nanmean() function ignoring missing value

```python
# Compute the difference: diff
diff = (by_year-climatology)*9/5
# Compute the average over last two axes: avg
avg = da.nanmean(diff, axis=(-1,-2)).compute()
# Plot the slices [:,0], [:,7], and [:11] against the x values
x = range(2008,2012)
f, ax = plt.subplots()
ax.plot(x,avg[:,0], label='Jan')
ax.plot(x,avg[:,7], label='Aug')
ax.plot(x,avg[:,11], label='Dec')
ax.axhline(0, color='red')
ax.set_xlabel('Year')
ax.set_ylabel('Difference (degrees Fahrenheit)')
ax.legend(loc=0)
plt.show()
```

## Dask DataFrame ~ Pandas DF

- dask.dataframe as dd
- **High-level Scalable Pandas DF**

```python
# Using WDI csv dataset
import dask.dataframe as dd

req = requests.get('https://assets.datacamp.com/production/course_4299/datasets/WDI.zip')
zip = zipfile.ZipFile(io.BytesIO(req.content))
zip.NameToInfo

df = dd.read_csv(zip.extract('WDI.csv'))
df.head()

df.groupby(df.name).value.mean()
```

```
---------------------------------------------------------------------------

NameError                                 Traceback (most recent call last)

<ipython-input-1-3e3642d3fe4c> in <module>
      2 import dask.dataframe as dd
      3 
----> 4 req = requests.get('https://assets.datacamp.com/production/course_4299/datasets/WDI.zip')
      5 zip = zipfile.ZipFile(io.BytesIO(req.content))
      6 zip.NameToInfo
```

```
NameError: name 'requests' is not defined
```



```python
# Boolean series where 'Indicator Code' is 'EN.ATM.PM25.MC.ZS': toxins
toxins = df['Indicator Code'] == 'AG.LND.TRAC.ZS'
# Boolean series where 'Region' is 'East Asia & Pacific': region
region = df['Region'] == 'East Asia & Pacific'

# Filter the DataFrame using toxins & region: filtered
filtered = df.loc[toxins & region]

# Groupby and Compute mean
yearly_mean = filtered.groupby('Year').mean().compute()

yearly_mean['value'].plot.line()
plt.show()
```



```
<matplotlib.axes._subplots.AxesSubplot at 0x7f1763ae4e80>
```



![png](/Users/Ocean/Desktop/DASK/output_53_1.png)

## Timing Dask DF Loading and Computation

- Quick example of 12 2GB files loading and averaging reveals Dask DF takes about 3min at compute call, which Pandas loading 1 file 43s

#### Decision to both largely depends on whehter

1. Data size fit into I/O (disk) and/or CPU (RAM)
2. Requires Pandas methods non-existent in Dask

#### Example analysing full-year taxi tipping

```python
# Read all .csv files: df
df = dd.read_csv('taxi/*.csv', assume_missing=True)

# Make column 'tip_fraction'
df['tip_fraction'] = df['tip_amount'] / (df['total_amount'] - df['tip_amount'])

# Convert 'tpep_dropoff_datetime' column to datetime objects
df['tpep_dropoff_datetime'] = dd.to_datetime(df['tpep_dropoff_datetime'])

# Construct column 'hour'
df['hour'] = df['tpep_dropoff_datetime'].dt.hour

# Filter rows where payment_type == 1: credit
credit = df.loc[df['payment_type'] == 1]

# Group by 'hour' column: hourly
hourly = credit.groupby('hour')

# Aggregate mean 'tip_fraction' and print its data type
result = hourly['tip_fraction'].mean()
print(type(result))

# Perform the computation
tip_frac = result.compute()

# Print the type of tip_frac
print(type(tip_frac))

# Generate a line plot using .plot.line()
tip_frac.plot.line()
plt.ylabel('Tip fraction')
plt.show()
```

## Dask Bag and Globbing

- List of nested kinds: list, dict, string, etc
- Normally test file containing one \n separated text

```python
import glob

req = requests.get('https://assets.datacamp.com/production/course_4299/datasets/sotu.zip')
zip = zipfile.ZipFile(io.BytesIO(req.content))
zip.extractall()

filenames = glob.glob('sotu/*.txt')
filenames = sorted(filenames)

import dask.bag as db

speeches = db.read_text(filenames)
print(speeches.count().compute())
```

```
237
```



```python
# Call .take(1): one_element
one_element = speeches.take(1)

# Extract first element of one_element: first_speech
first_speech = one_element[0]

# Print type of first_speech and first 60 characters
print(type(first_speech))
print(first_speech[:61])
```

```
<class 'str'>
 Fellow-Citizens of the Senate and House of Representatives: 
```



```python
# Call .str.split(' ') from speeches and assign it to by_word
by_word = speeches.str.split(' ')

# Map the len function over by_word and compute its mean
n_words = by_word.map(len)
avg_words = n_words.mean()

# Print the type of avg_words and value of avg_words.compute()
print(type(avg_words))
print(avg_words.compute())

# Convert speeches to lower case: lower
lower = speeches.str.lower()

# Filter lower for the presence of 'health care': health
health = lower.filter(lambda s:'health care' in s)

# Count the number of entries : n_health
n_health = health.count()

# Compute and print the value of n_health
print(n_health.compute())


```

```python
# Call db.read_text with congress/bills*.json: bills_text
bills_text = db.read_text('congress/bills*.json')

# Map the json.loads function over all elements: bills_dicts
bills_dicts = bills_text.map(json.loads)

# Extract the first element with .take(1) and index to the first position: first_bill
first_bill = bills_dicts.take(1)[0]

# Print the keys of first_bill
print(first_bill.keys())


# Filter the bills: overridden
overridden = bills_dicts.filter(veto_override)

# Print the number of bills retained
print(overridden.count().compute())

# Get the value of the 'title' key
titles = overridden.pluck('title')

# Compute and print the titles
print(titles.compute())


# Define a function lifespan that takes a dictionary d as input
def lifespan(d):
    # Convert to datetime
    current = pd.to_datetime(d['current_status_date'])
    intro = pd.to_datetime(d['introduced_date'])

    # Return the number of days
    return (current - intro).days

# Filter bills_dicts: days
days = bills_dicts.filter(lambda s:s['current_status']=='enacted_signed').map(lifespan)

# Print the mean value of the days Bag
print(days.mean().compute())
```

## All together Detailed Analysis

```python
# Define @delayed-function read_flights
@delayed
def read_flights(filename):

    # Read in the DataFrame: df
    df = pd.read_csv(filename, parse_dates=['FL_DATE'])

    # Replace 0s in df['WEATHER_DELAY'] with np.nan
    df['WEATHER_DELAY'] = df['WEATHER_DELAY'].replace(0, np.nan)

    # Return df
    return df


# Loop over filenames with index filename
for filename in filenames:
    # Apply read_flights to filename; append to dataframes
    dataframes.append(read_flights(filename))

# Compute flight delays: flight_delays
flight_delays = dd.from_delayed(dataframes)

# Print average of 'WEATHER_DELAY' column of flight_delays
print(flight_delays['WEATHER_DELAY'].mean().compute())


# Define @delayed-function read_weather with input filename
@delayed
def read_weather(filename):
    # Read in filename: df
    df = pd.read_csv(filename, parse_dates=['Date'])

    # Clean 'PrecipitationIn'
    df['PrecipitationIn'] = pd.to_numeric(df['PrecipitationIn'], errors='coerce')

    # Create the 'Airport' column
    df['Airport'] = filename.split('.')[0]

    # Return df
    return df



# Loop over filenames with filename
for filename in filenames:
    # Invoke read_weather on filename; append resultt to weather_dfs
    weather_dfs.append(read_weather(filename))

# Call dd.from_delayed() with weather_dfs: weather
weather = dd.from_delayed(weather_dfs)

# Print result of weather.nlargest(1, 'Max TemperatureF')
print(weather.nlargest(1, 'Max TemperatureF').compute())


# Make cleaned Boolean Series from weather['Events']: is_snowy
is_snowy = weather['Events'].str.contains('Snow').fillna(False)

# Create filtered DataFrame with weather.loc & is_snowy: got_snow
got_snow = weather.loc[is_snowy]

# Groupby 'Airport' column; select 'PrecipitationIn'; aggregate sum(): result
result = got_snow.groupby('Airport')['PrecipitationIn'].sum()

# Compute & print the value of result
print(result.compute())


def percent_delayed(df):
    return (df['WEATHER_DELAY'].count() / len(df)) * 100

# Print time in milliseconds to compute percentage_delayed on weather_delays
t_start = time.time()
print(percent_delayed(weather_delays).compute())
t_end = time.time()
print((t_end-t_start)*1000)

# Call weather_delays.persist(): persisted_weather_delays
persisted_weather_delays = weather_delays.persist()

# Print time in milliseconds to compute percentage_delayed on persisted_weather_delays
t_start = time.time()
print(percent_delayed(persisted_weather_delays).compute())
t_end = time.time()
print((t_end-t_start)*1000)



# Group persisted_weather_delays by 'Events': by_event
by_event = persisted_weather_delays.groupby('Events')

# Count 'by_event['WEATHER_DELAY'] column & divide by total number of delayed flights
pct_delayed = by_event['WEATHER_DELAY'].count() / persisted_weather_delays['WEATHER_DELAY'].count() * 100

# Compute & print five largest values of pct_delayed
print(pct_delayed.nlargest(5).compute())

# Calculate mean of by_event['WEATHER_DELAY'] column & return the 5 largest entries: avg_delay_time
avg_delay_time = by_event['WEATHER_DELAY'].mean().nlargest(5)

# Compute & print avg_delay_time
print(avg_delay_time.compute())
```

# Streamz - Streaming Data Analysis Pythonic with Dask <a id="streamz"></a>

1. Streamz.core
   - map, accumulate
   - time control and back pressure
   - jupyter integration
2. Streamz.dataframe
   - stream of pandas df
   - with pandas API
   - plotting with Holoviews/Bokeh
3. Streamz.dask
   - full effecting on top of Dask
   - adds millisec overhead and 10-20ms latency
   - scales

```python
from IPython.display import HTML

HTML('<div style="position:relative;height:0;padding-bottom:56.25%"><iframe width="560" height="315" src="https://www.youtube.com/embed/yI_yZoUaz60" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>')
```



<div style="position:relative;height:0;padding-bottom:56.25%"><iframe width="560" height="315" src="https://www.youtube.com/embed/yI_yZoUaz60" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>



```python
from streamz import Stream

stream = Stream()
```

```python
stream.emit(100) # push data into stream BUT YET to stream data

# once below defined, stream becomes active with map()
```

```
101
200
```



```python
def inc(x):
    return x + 1
def double(x):
    return 2 *x

a = stream.map(inc).map(print)
b = stream.map(double).map(print)
```

```python
stream.visualize()
```



![png](/Users/Ocean/Desktop/DASK/output_68_0.png)



### Code to create random JSON data

```python
from datetime import datetime
import json
import random

i = 0
record_names = ['Alice', 'Bob', ' Charlie']

def create_record():
    global i
    i += 1
    record = {'name': random.choice(record_names),
             'i': i,
             'x': random.random(),
             'y': random.randint(0, 10),
             'time': str(datetime.now())}
    return json.dumps(record)
```

```python
create_record() # random stream of data
```



```
'{"name": "Alice", "i": 29, "x": 0.12326123720304571, "y": 2, "time": "2018-11-29 07:29:08.834679"}'
```



### Basic Streams and Map

```python
source = Stream()
source
```

```
Output()
```



```python
# create stream of json-parsed records
records = source.map(json.loads)
records
```

```
Output()
```



```python
# create stream of names
names = records.map(lambda d: d['name'])
names
```

```
Output()
```



```python
# push data into stream
source.emit(create_record())
```

```
{'name': 'Bob',
 'i': 39,
 'x': 0.417140916688034,
 'y': 1,
 'time': '2018-11-29 07:30:16.494038'}
```

## Async Computation

```python
from tornado import gen
import time

def increment(x):
    """ A blocking increment function

    Simulates a computational function that was not designed to work
    asynchronously
    """
    time.sleep(0.1)
    return x + 1

@gen.coroutine
def write(x):
    """ A non-blocking write function

    Simulates writing to a database asynchronously
    """
    yield gen.sleep(0.2)
    print(x)
```

```python
# Within Event Loop: e.g. an app running strictly within event loop

from streamz import Stream
from tornado.ioloop import IOLoop

async def f():
    source = Stream(asynchronous=True)  # tell the stream we're working asynchronously
    source.map(increment).rate_limit(0.500).sink(write)

    for x in range(10):
        await source.emit(x)

IOLoop().run_sync(f)
```

```
---------------------------------------------------------------------------

RuntimeError                              Traceback (most recent call last)

<ipython-input-97-3b7d97fa315e> in <module>
     11         await source.emit(x)
     12 
---> 13 IOLoop().run_sync(f)
```

```
/usr/local/lib/python3.6/dist-packages/tornado/ioloop.py in run_sync(self, func, timeout)
    569                     self.stop()
    570             timeout_handle = self.add_timeout(self.time() + timeout, timeout_callback)
--> 571         self.start()
    572         if timeout is not None:
    573             self.remove_timeout(timeout_handle)
```

```
/usr/local/lib/python3.6/dist-packages/tornado/platform/asyncio.py in start(self)
    130             self._setup_logging()
    131             asyncio.set_event_loop(self.asyncio_loop)
--> 132             self.asyncio_loop.run_forever()
    133         finally:
    134             asyncio.set_event_loop(old_loop)
```

```
/usr/lib/python3.6/asyncio/base_events.py in run_forever(self)
    410         if events._get_running_loop() is not None:
    411             raise RuntimeError(
--> 412                 'Cannot run the event loop while another loop is running')
    413         self._set_coroutine_wrapper(self._debug)
    414         self._thread_id = threading.get_ident()
```

```
RuntimeError: Cannot run the event loop while another loop is running
```

### Mock Continous updates

```python
from tornado import gen
from tornado.ioloop import IOLoop

async def f():
    while True:
        await gen.sleep(0.5)
        await source.emit(create_record(), asynchronous=True)
        
IOLoop.current().add_callback(f)
```

```
'Alice'
```

```
tornado.application - ERROR - Exception in callback functools.partial(<function wrap.<locals>.null_wrapper at 0x7ff9640da158>, <Task finished coro=<f() done, defined at <ipython-input-78-8a3fd1b15f3b>:4> exception=TypeError("object NoneType can't be used in 'await' expression",)>)
Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/tornado/ioloop.py", line 758, in _run_callback
    ret = callback()
  File "/usr/local/lib/python3.6/dist-packages/tornado/stack_context.py", line 300, in null_wrapper
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/tornado/ioloop.py", line 779, in _discard_future_result
    future.result()
  File "<ipython-input-78-8a3fd1b15f3b>", line 7, in f
    await source.emit(create_record(), asynchronous=True)
TypeError: object NoneType can't be used in 'await' expression
```

### Accumulators

```python
# Sum 'x' over time

def binop(totla, new):
    return total + new

records.map(lambda d: d['x']).accumulate(binop, start=0)
```

```
---------------------------------------------------------------------------

NameError                                 Traceback (most recent call last)

<ipython-input-24-b9420dfdd9c4> in <module>
      4     return total + new
      5 
----> 6 records.map(lambda d: d['x']).accumulate(binop, start=0)
```

```
NameError: name 'records' is not defined
```



```python
# Count occurences of names over time

def accumulator(acc, new):
    acc = acc.copy()
    if new in acc:
        acc[new] += 1
    else:
        acc[new] = 1
    return acc

names.accumulate(accumulator, start={})
```

```
---------------------------------------------------------------------------

NameError                                 Traceback (most recent call last)

<ipython-input-25-10c5c3797b77> in <module>
      9     return acc
     10 
---> 11 names.accumulate(accumulator, start={})
```

```
NameError: name 'names' is not defined
```



```python

```

# Streaming + Bokeh Server

- bokeh's true value is serving live-streaming, interactive visual updating real-time data
- [dignostics for distributed system](https://distributed.readthedocs.io/en/latest/web.html)

### Launch Bokeh Servers from Notebook

- Make func which is called when site visited - whatever it wants, here a simple line plot with doc.add_root() method
- This starts a Tornado web server creating new image whenever connected, similar to lib in Tornado or Flask - in this case piggybacks on Jupyter own IOLoop because Bokeh is built on Tornado it can play nicely with other **async** apps like Tornado or Asyncio

```python
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource

def make_document(doc):
    fig = figure(title='Line plot', sizing_mode='scale_width')
    fig.line(x=[1,2,3], y=[1,4,9])
    
    doc.title = "Hellow, world!"
    doc.add_root(fig)
    
apps = {'/': Application(FunctionHandler(make_document))}

server = Server(apps, port=5000)
server.start()
```

### Live Updates

- Doing live visual often means serialising data on server, figuring out how web sockets work, sending data to client/browser and then updating plots in browser
- Bokeh handles this by keeping a **synchronised** table of data on client and the server, the `ColumnDataSource`.
- If defining plots around column data source then pushing more data into source, Bokeh will handle the rest - updating plots in broswer just needs pushing more data into object on server
- Below, make new object upding func adding new record, set up callback to call func every 100ms the graph
- Changing figures (or adding multiple figures, text, or visual elements, etc) full freedom over visual styling 
- Changing around update func can pull data from sensors, shove in more data etc

```python
import random

def make_document(doc):
    source = ColumnDataSource({'x': [], 'y': [], 'color': []})

    def update():
        new = {'x': [random.random()],
               'y': [random.random()],
               'color': [random.choice(['red', 'blue', 'green'])]}
        source.stream(new)

    doc.add_periodic_callback(update, 100)

    fig = figure(title='Streaming Circle Plot!', sizing_mode='scale_width',
                 x_range=[0, 1], y_range=[0, 1])
    fig.circle(source=source, x='x', y='y', color='color', size=10)

    doc.title = "Now with live updating!"
    doc.add_root(fig)

apps = {'/': Application(FunctionHandler(make_document))}

server = Server(apps, port=5001)
server.start()
```

### Real example - Dask's dashboard 

```python
def make_document(doc):
    source = ColumnDataSource({'time': [time(), time() + 1],
                               'idle': [0, 0.1],
                               'saturated': [0, 0.1]})

    x_range = DataRange1d(follow='end', follow_interval=20000, range_padding=0)

    fig = figure(title="Idle and Saturated Workers Over Time",
                 x_axis_type='datetime', y_range=[-0.1, len(scheduler.workers) + 0.1],
                 height=150, tools='', x_range=x_range, **kwargs)
    fig.line(source=source, x='time', y='idle', color='red')
    fig.line(source=source, x='time', y='saturated', color='green')
    fig.yaxis.minor_tick_line_color = None

    fig.add_tools(
        ResetTool(reset_size=False),
        PanTool(dimensions="width"),
        WheelZoomTool(dimensions="width")
    )

    doc.add_root(fig)

    def update():
        result = {'time': [time() * 1000],
                  'idle': [len(scheduler.idle)],
                  'saturated': [len(scheduler.saturated)]}
        source.stream(result, 10000)

    doc.add_periodic_callback(update, 100)
```

## Streaming Dataframes

- [Article](http://matthewrocklin.com/blog/work/2017/10/16/streaming-dataframes-1)

```python
from streamz.dataframe import Random

sdf = Random(freq='1ms', interval='100ms')
```

```python

```