---
title: "DODS"
date: 2019-12-18T15:29:39+08:00
showDate: true
draft: false
---

# Data Science

**Raw**

`Data Sciencen_array.dtype.name`

`n_array.ravel` - flatten

`pd.Series(np.random.rand(5))`

`d['some'].isnull().value_counts()` then `dropna()`

`df2.fillna(df2.mean())`

!!! String Regex in Pandas !!!

`df['text'][0:5].str.extract('(\w+)\s(\w+)')` (other ops with `.str.` like `.upper()`, `.len()`, `.replace('DISTRICT$', 'DIST')`)

```python
df['NO. OBESE'].groupby(d['GRADE LEVEL']).aggregate([sum, mean, std])
```



**Inference**

```python
>>> from scipy.stats import binom
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)
>>> x = [0, 1, 2, 3, 4, 5, 6]
>>> n, p = 6, 0.5
>>> rv = binom(n, p)
>>> ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
          label='Probablity')
>>> ax.legend(loc='best', frameon=False)
>>> plt.show()


>>> from scipy.stats import poisson
>>> rv = poisson(20)
>>> rv.pmf(23)
0.066881473662401172
# With the Poisson function, we define the mean value, which is 20 cars. The rv.pmf function gives the probability, which is around 6%, that 23 cars will pass the bridge.

>>> from scipy import stats
>>> stats.bernoulli.rvs(0.7, size=100)

>>> classscore = np.random.normal(50, 10, 60).round()
>>> plt.hist(classscore, 30, normed=True) #Number of breaks is 30
>>> stats.zscore(classscore)
>>> prob = 1 - stats.norm.cdf(1.334)
>>> prob
0.091101928265359899
>>> stats.norm.ppf(0.80) # get the z-score at which the top 20% score marks

# The z-score for the preceding output that determines whether the top 20% marks are at 0.84 is as follows:
>>> (0.84 * classscore.std()) + classscore.mean()
55.942594176524267

# We multiply the z-score with the standard deviation and then add the result with the mean of the distribution. This helps in converting the z-score to a value in the distribution. The 55.83 marks means that students who have marks more than this are in the top 20% of the distribution.
# The z-score is an essential concept in statistics, which is widely used. Now you can understand that it is basically used in standardizing any distribution so that it can be compared or inferences can be derived from it.


# Let's understand this concept with an example where the null hypothesis is that it is common for students to score 68 marks in mathematics.
# Let's define the significance level at 5%. If the p-value is less than 5%, then the null hypothesis is rejected and it is not common to score 68 marks in mathematics.
# Let's get the z-score of 68 marks:
>>> zscore = ( 68 - classscore.mean() ) / classscore.std()
>>> zscore
  2.283

>>> prob = 1 - stats.norm.cdf(zscore)
>>> prob
0.032835182628040638
# So, you can see that the p-value is at 3.2%, which is lower than the significance level. This means that the null hypothesis can be rejected, and it can be said that it's not common to get 68 marks in mathematics.

# Type-1 Error == False Positive (falsely rejecting Null Hypothesis)
# Type-2 Error == False Negative (falsely accepting NULL)

>>> stats.pearsonr(mpg,hp)
>>> stats.spearmanr(mpg,hp)
# We can clearly see that the Pearson correlation has been drastically affected due to the outliers, which are from a correlation of 0.89 to 0.47.
# The Spearman correlation did not get affected much as it is based on the order rather than the actual value in the data.

>>> stats.ttest_ind(class1_score,class2_score)
```



**Chi^2**

$X^2 = [(n-1)*s^2] / \sigma^2$

If repeat-sample and define chi-square statistics, PDF obtained:

$Y = Y_0 * (X^2)^{v/2-1} *e^{-X2/2}$

> Y0 a constant depending on # DF, X2 the chi-square stats, v = n - 1 the # DF

- goodness of fit

```python
# e.g. dice rolling 36 times each expected frequency == 6
expected = np.array([6,6,6,6,6,6])
# observed frequency as 
observed = np.array([7,5,3,9,6,6])
# H0: observed value ~= expected value
stats.chisquare(observed, expected) # (3.333333333333, 0.64874323423)
# first stats latter p-value, unable to rejcet H0 
```

- test of independence (pair of categorical variables independence)

```python
# e.g. male-female frequency on 3 categorical book genres
men_women = np.array([
  [100, 120, 60],
  [350, 200, 90],
])
stats.chi2_contingency(men_women) # (28.23912391, 6.9912312323e-07, 2, array...)
# rejecting H0 showing association between gender and genre read, third value DF, fourth expected frequencies
```



**ANOVA** (F-test)

Test differences between means. 

$H_o: \mu_1 = \mu_2 = … \mu_k$

`stats.f_oneway(class1, class2, class2)` >>> (stats, p-value)



# Fluent Python

**DATA MODEL**

```python
import collections
Card = collections.namedtuple('Card', ['rank', 'suit'])
class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()
    def __init__(self):
	    self._cards = [Card(rank, suit) for suit in self.suits
                       for rank in self.ranks]
    def __len__(self):
 	   return len(self._cards)
    def __getitem__(self, position):
		return self._cards[position]
    
beer_card = Card('7', 'diamonds') # Card(rank='7', suit='..')
```

- Point is FrenchDeck, it responds to len() and accessing [] because of dunders !

- Pros of using special dunders!! (1) users of classes has pythonic method unlike `.method()` and (2) easier to benefit rich SL like `random.choice` to pick random cards !!!

- `__getitem__` delegates `[]` to `self._cards` to auto-slice according the class nature, while also making deck **iterable** `for card in deck: print(card)`

- If collection has no `__contains__` , the `in` operator does a sequential scan!

- Simply the two dunders can seize all benefits of `sorted` 

  ```python
  suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)
  def spades_high(card):
      rank_value = FrenchDeck.ranks.index(card.rank)
      return rank_value * len(suit_values) + suit_values[card.suit]
  for card in sorted(deck, key=spades_high): # doctest: +ELLIPSIS
     print(card)
  # auto sorting due to dunders delegation
  ```

- dunders or special methods are meant to be called by interpreter NOT coder: no `my_object.__len__()` but `len(my_object)` and if it's instance of user defined class, the Python calls the `__len__` instance method coded

- But for built-in types `list` etc, interpreter shortcut CPython uses `PyVarObjet C` struct in MEM faster than calling method

- Most special calls implicit like `for i in x` invokes `iter(x)` in turn may call `x.__iter__()` if exists

- Unless metaprogramming, often use built-in calss `len, iter, str,` etc 

- Example `v = Vector(3, 4); abs(v) >>> 5.0` the magnitude for consistency:

  ```python
  from math import hypot
  
  class Vector:
      def __init__(self, x=0, y=0):
          self.x = x
          self.y = y
  	def __repr__(self):
          return 'Vector...'
      def __abs__(self):
          return hypot(self.x, self.y)
      def __bool__(self):
          return bool(abs(self))
      def __add__(self, other):
          x = self.x + other.x
          ...
          return Vector(x, y)
      def __mul__(self, scalar):
          return Vector(self.x * scalar, self.y * scalar)
  ```

**TYPES**

Container sequences

- `list`, `tuple`, `collections.deque` holding varied types
- `str, bytes, bytearray, memoryview, array.array` holding single type

Mutability

- `list, bytearray, array.array, collections.deque, memoryview` mutable
- `tuple, str, bytes` immutable HENCE hold only **primitive types** like char, bytes, numbers

**generator expressions vital for non-list creation**

- `array.array('I', (ord(sth) for i in sth))` creating array
- without creating memory `for i in ((c, s) for c in colors for s in sizes): ...` 

**tuple is nameless records due to indexing**

```python
>>> lax_coordinates = (33.9425, -118.408056)
>>> city, year, pop, chg, area = ('Tokyo', 2003, 32450, 0.66, 8014)
>>> traveler_ids = [('USA', '31195855'), ('BRA', 'CE342567'),
...
('ESP', 'XDA205856')]
>>> for passport in sorted(traveler_ids):
...
print('%s/%s' % passport)
...
BRA/CE342567
ESP/XDA205856
USA/31195855
>>> for country, _ in traveler_ids:
...
print(country)
...
USA
BRA
ESP

# namedTuple
>>> from collections import namedtuple
>>> City = namedtuple('City', 'name country population coordinates')
>>> tokyo = City('Tokyo', 'JP', 36.933, (35.689722, 139.691667))
>>> tokyo
City(name='Tokyo', country='JP', population=36.933, coordinates=(35.689722, 139.691667))
>>> tokyo.population
36.933
>>> tokyo.coordinates
(35.689722, 139.691667)
>>> tok

>>> City._fields
('name', 'country', 'population', 'coordinates')
>>> LatLong = namedtuple('LatLong', 'lat long')
>>> delhi_data = ('Delhi NCR', 'IN', 21.935, LatLong(28.613889, 77.208889))
>>> delhi = City._make(delhi_data)
>>> delhi._asdict()
OrderedDict([('name', 'Delhi NCR'), ('country', 'IN'), ('population',
21.935), ('coordinates', LatLong(lat=28.613889, long=77.208889))])
>>> for key, value in delhi._asdict().items():
print(key + ':', value)
```

1. `_fields` is a tuple with field names of the class
2. `_make()` inits named tuple from an iterable; `City(*delhi_data)` would do the same
3. `_asdict()` returns a `collections.OrderedDict` built form named tuple instance

**bisect and insort**

```python
import bisect
import sys
HAYSTACK = [1, 4, 5, 6, 8, 12, 15, 20, 21, 23, 23, 26, 29, 30]
NEEDLES = [0, 1, 2, 5, 8, 10, 22, 23, 29, 30, 31]
ROW_FMT = '{0:2d} @ {1:2d}
{2}{0:<2d}'
def demo(bisect_fn):
for needle in reversed(NEEDLES):
position = bisect_fn(HAYSTACK, needle)
offset = position * ' |'
print(ROW_FMT.format(needle, position, offset))
if __name__ == '__main__':
if sys.argv[-1] == 'left':
    bisect_fn = bisect.bisect_left
else:
	bisect_fn = bisect.bisect
print('DEMO:', bisect_fn.__name__)
print('haystack ->', ' '.join('%2d' % n for n in HAYSTACK))
demo(bisect_fn)


# An interesting application of bisect is to perform table lookups by numeric values, for example to convert test scores to letter grades, as in Example 2-18. Example 2-18. Given a test score, grade returns the corresponding letter grade.
>>> def grade(score, breakpoints=[60, 70, 80, 90], grades='FDCBA'):
...
i = bisect.bisect(breakpoints, score)
...
return grades[i]
...
>>> [grade(score) for score in [33, 99, 77, 70, 89, 90, 100]]
['F', 'A', 'C', 'C', 'B', 'A', 'A']

# keeping sorted 
random.seed(1729)
my_list = []
for i in range(SIZE):
    new_item = random.randrange(SIZE*2)
    bisect.insort(my_list, new_item)
    print('%2d ->' % new_item, my_list)
```

**memoryview - generalised NumPy without the maths**

```python
>>> numbers = array.array('h', [-2, -1, 0, 1, 2])
>>> memv = memoryview(numbers)
>>> len(memv)
5
>>> memv[0]
-2
>>> memv_oct = memv.cast('B')
>>> memv_oct.tolist()
[254, 255, 255, 255, 0, 0, 1, 0, 2, 0]
>>> memv_oct[5] = 4
>>> numbers
array('h', [-2, -1, 1024, 1, 2])
```



NumPy detour

```python
>>> import numpy
>>> floats = numpy.loadtxt('floats-10M-lines.txt')
>>> floats[-3:]
array([ 3016362.69195522,
535281.10514262, 4566560.44373946])
>>> floats *= .5
>>> floats[-3:]
array([ 1508181.34597761,
267640.55257131, 2283280.22186973])
>>> from time import perf_counter as pc
>>> t0 = pc(); floats /= 3; pc() - t0
0.03690556302899495
>>> numpy.save('floats-10M', floats)
>>> floats2 = numpy.load('floats-10M.npy', 'r+')
>>> floats2 *= 6
>>> floats2[-3:]
memmap([ 3016362.69195522,
535281.10514262, 4566560.44373946])
```

**deque**

```python
>>> from collections import deque
>>> dq = deque(range(10), maxlen=10)
>>> dq
deque([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], maxlen=10)
>>> dq.rotate(3)
>>> dq
deque([7, 8, 9, 0, 1, 2, 3, 4, 5, 6], maxlen=10)
>>> dq.rotate(-4)
>>> dq
deque([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], maxlen=10)
>>> dq.appendleft(-1)
>>> dq
deque([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9], maxlen=10)
>>> dq.extend([11, 22, 33])
>>> dq
deque([3, 4, 5, 6, 7, 8, 9, 11, 22, 33], maxlen=10)
>>> dq.extendleft([10, 20, 30, 40])
>>> dq
deque([40, 30, 20, 10, 3, 4, 5, 6, 7, 8], maxlen=10)
```



> Besides deque , other Python Standard Library packages implement queues.
> queue
> Provides the synchronized (i.e. thread-safe) classes Queue , LifoQueue and Priori
> tyQueue . These are used for safe communication between threads. All three classes
> can be bounded by providing a maxsize argument greater than 0 to the constructor.
> However, they don’t discard items to make room as deque does. Instead, when the
> queue is full the insertion of a new item blocks — i.e. it waits until some other threadmakes room by taking an item from the queue, which is useful to throttle the num‐
> ber of live threads.
> multiprocessing
> Implements its own bounded Queue , very similar to queue.Queue but designed for
> inter-process communication. There is also has a specialized multiprocess
> ing.JoinableQueue for easier task management.
> asyncio
> Newly added to Python 3.4, asyncio provides Queue , LifoQueue , PriorityQueue
> and JoinableQueue with APIs inspired by the classes in queue and multiprocess
> ing , but adapted for managing tasks in asynchronous programming.
> heapq
> In contrast to the previous three modules, heapq does not implement a queue class,
> but provides functions like heappush and heappop that let you use a mutable se‐
> quence as a heap queue or priority queue



# Jake DS

- wild-search `object.*keyword*?`
- Memory usage profiler `pip install memory_profiler` and `%load_ext`
  - `%memit` similar to `%timeit`
  - BUT line-wise need separate modules `%%file module_name.py` followed by code then import method and `%mprun -f method method(args...)`
- Python Int is more than Int, x is pointer to C struct with 4 pieces
  - `ob_refcnt` reference count silently handling memory alloc-dealloc
  - `ob_type` encoding type of variable
  - `ob_size` specifying size of data members
  - `ob_digit` containing actual integer value in Python
  - Python Integer is a structure containing many meta data
  - C integer is essentially a **label for position in memory whose bytes encode an integer value**

```c
struct _longobject {
    long ob_refcnt;
    PyTypeObject *ob_type;
    size_t ob_size;
    long ob_digit[1];
};
```

- Same logic, a heterogeneous list contains **type info, reference count, other meta** that each element its python object while array is a single pointer to start position + size, unlike list which is a pointer to many pointers each storing the above integer like meta data

**NUMPY**

```python
# basic arrays
np.array()
np.arange()
np.linspace()
np.random.random((3, 3)) # default [0,1]
np.random.normal(0, 1, (3,3)) 
np.random.randint(0, 10, (3, 3)) # 3x3 in [0, 10)

```

- NumPy is the key to python data manipulation
- numpy object is fixed typed hence persist only single type
- reshaping 1D to 2D `X[np.newaxis, :]` 
- multi-D array concatenation best use `vstack` or `hstack` for clarity
- `np.split(arr, list_split_index)` and `hsplit`

VECTORISED operation **UFUNC** 

- NumPy interface for statically typed, compiled routine applied to each element pushing the LOOP into compiled layer underlying 
- vectorised operations in NP effected via *ufuncs* whose main purpose is to fast do repeated ops on values in NP-Array, extremely flexible

- `scipy.special` contains many functions beyond ordinary

```python
from scipy import special
# Gamma functions (generalised factorials)
special.gamma()
special.gammaln()
special.beta()
# error functions (integral of Gaussian)
special.erf()
special.erfc()
special.erfinv()
```

- Advanced UFUNC
  - specifying ouput `np.multiply(x, 10, out=y)` (set `y = np.empty(5)`) and even write results to alternate element `np.power(2, x, out=y[::2]`) 
    - instead of `y[::2] = 2 ** x` which creates temp array holding results followed by copying values into `y`, important speed up for large arrays memory saving !!!
  - Binary ufunc: `reduce` repeated applies ops till only a single result remains
    - `np.add.reduce(x)` calling on `add` ufunc returns sum of all in array
    - `np.add.accumulate(x)` for intermediate results
    - directly by `np.sum, np.prod, np.cumsum, np.cumprod`
  - Outer product: allowing in-line creating multiplication table
    - `np.multiply.outer(x, x)` 
    - `ufunc.at, ufunc.reduceat` will be explored later in indexing

AGGREGATION

- `np.sum` faster due to exec in compiled code than `sum` , best use dot-method of these for clarity
- note `NaN`-safe version of most `np.nanpercentile, np.nanmedian` etc

BROADCASTING (mix-dim binary ufuncs)

- scalar ops as if stretching/duplicating scalar into array of matching size then ops!! Broadcasting shines on **none-duplication** of such ops!

- mismatched arrays BOTH stretched to COMMON shape before ops

- RULES:

  1. If arrays differ in ndim, the shape of fewer is padded with ones on its leading (left) side !!!
  2. If shape mismatch in any dim, the array with shape == 1 in that dim is stretched/broadcasted to match the other shape
  3. If disagreed in any dim and neither == 1, error raised

- Example:

  ```python
  M.shape = (2,3)
  a.shape = (3,) # a vector
  # rule-1 sees a has fewer dim so padded on left with ones
  M.shape -> (2, 3)
  a.shape -> (1, 3)
  # rule-2 see axis-0 disagrees, so streched this dim to match
  M.shape -> (2, 3)
  a.shape -> (2, 3)
  
  # example 2
  a.shape = (3, 1)
  b.shape = (3,)
  # rule-1 to pad shape of b with ones
  b.shape -> (1, 3)
  # rule-2 upgrade each of ones to match size of other array
  a.shape -> (3, 3)
  b.shape -> (3, 3)
  
  # example 3
  M.shape = (3, 2)
  a.shape = (3,)
  # rule-1 padding a with ones
  a.shape -> (1, 3)
  # rule-2 to stretch axis-0 of a to match that of M
  a.shape -> (3, 3)
  # rule-3 raise error for none of axis-0 mismatch in final shape
  
  # Potential Confusion: a and M compatible via padding a's shape 
  # with ones on the right! BUT this is not how BC work!
  # instead, reshape right side via np.newaxis
  a[:, np.newaxis].shape # (3, 1)
  # now matches
  
  np.logaddexp(M, a[:, np.newaxis]) # log(exp(a) + exp(b))
  ```

- Practice: normalise matrix on axis and visualise 2D from functions `x = np.linspace(0, 5, 50), y = np.linspace(0, 5, 50)[:, np.newaxis], z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x), plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap=viridis'), plt.colorbar()`

BOOLEAN MASKING

- `rng = np.random.RandomState(42), x = rng.randint(10, size=(3, 4))`
- counting `np.count_nonzero(x < 6)`  or `np.sum()` which is better for its `axis` 
- checking `np.any(x > 8)` and `np.all(x < 10)`
- boolean ops `& | ^ ~` NP overloads these, as other ops, ufuncs which work element-wise on arrays
- MASKING objects

```python
rainy = (inches > 0)
days = np.arange(365) # mask of all summer days
summer = (days > 172) & (days < 262)
np.median(inches[rainy])
np.median(inches[summer])
```

- `and or` gauge truth on entire object  while `& |` bitwise !! within each object which is always used for array boolean (boolean of array of integer or maybe container of elements might be ill-defined)

```python
bin(42) # 0b101010
bin(59) # 0b111011
bin(42 & 59) # 0b101010 or bitwise 1 0 boolean
```

FANCY INDEXING

- **shape of result reflects shape of INDEX arrays rather than shape of array INDEXED**

- `x array, ind = np.array([[3, 7], [4, 5]]), x[ind]` returns 2x2 array of indexed values of x !!
- Usage: making random arrays

```python
mena = [0, 0]
cov = [[1, 2], 
       [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
X.shape # (100, 2)

# subsetting 
indices = np.random.choice(X.shape[0], 20, replace=False)
selection = X[indices]
# plotting circling effet
plt.scatter(selection[: 0,], selection[:, 1],
            facecolor='none', s=200);
```

- `np.add.at(x, i, 1)` in-place adding and also `reduceat()` for repeated indexing ops rather than otherwise repeated assignment !!

- case: binning data: 

  ```python
  x = np.random.randn(100)
  bins = np.linspace(-5, 5, 20)
  counts = np.zeros_like(bins)
  i = np.searchsorted(bins, x) # find right bin for each x
  np.add.at(counts, i, 1) # add 1 to each of bins
  
  # the counts now reflect the num of points within each bin
  ```

SORTING

```python
#selection sort find and swap till sorted
def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
	return x
```

- default `np.sort` uses **quicksort** though **mergesort, heapsort** also available
- Example: KNN

```python
X = rand.rand(10, 2)
dist_sq = np.sum((X[:, np.newaxis, :]
                 - X[np.newaxis, :, :]) ** 2, axis=-1)
# broadcasting in detail
# for each pair of points, compute diff in corrdinates
# shape of diff = (10, 10, 2)

# argsort to sort along each row, leftmost give indices of NN
nearest = np.argsort(dist_sq, axis=1) # full sort
# partial k sort
K = 2
nearest_partition = np.argpartition(dist_sq, K+1, axis=1)

# plotting
plt.scatter(X[:, 0], X[:, 1], s=100)

# draw lines from each point to its 2NN
for i in range(X.shape[0]):
    for j in nearest_partition[i, :K+1]:
        plt.plot(*zip(X[j], X[i]), color='black')
```

- scaling KNN by **tree-based and approx** algo such as **KD-Tree**

STRUCTURE NUMPY

- `data = np.zeros(4, dtype={'names': ('name', 'age', 'weight'), 'formats': ('U10', 'i4', 'f8')})` (10 length Unicode string, 4-byte integer or int16 and 8-byte float or float64)
- then store heterogeneous data `data['name'] = ['Ocean', ...]` which output list of tuples as record
  - mask ops `data[data['age'] < 30]['name']`
- shortcut for types `np.dtype(['name', 'S10'), ('age', 'i4'), ...])` or without naming
  - `b, i4, u1, f8, c16, S5, U, V` 
- for writing Python API to legacy C library that manipulates structured data, use below construct, as numpy dtype accessed directly by C programs

```python
tp = np.dtype([
    ('id', 'i8'),
    ('mat', 'f8', (3, 3))
])
X = np.zeros(1, dtype=tp)
X[0] # (0, [[0.0, 0.0, 0.0], ...])
X['mat'][0] # the matrix form
```

- dot-access enabler `np.recarray`
  - `data_rec = data.view(np.recarray)` and then `data_rec.age` possible
  - NO, for its overhead !!!



**PANDAS**

- `data.ix[:3, :'pop']` hybrid of `loc, iloc`
- `data.loc[data.density > 100, ['pop', 'density']]`
- Convention! while indexing refers to columns, slicing refers to rows!!
- most numpy ufuncs works on pandas objects
- `A.add(B, fill_value=0)` optional NaN handler, such as `fill = A.stack().mean()`
- pandas equivalents of python ops `add, sub, mul, trudiv, floordiv, mod, pow`

MISSING

- Pandas handles using sentinel values `None` and `NaN`, and `None` CANNOT used in any arbitrary numpy/pandas array but ONLY in arrays with data type `'object'`, or python object
  - any such object will become `dtype=object` meaning best common type inferred is python object, while useful, any ops on data will be done on PYTHON LEVEL!!!, much more overhead than native types
- `NaN` is different, a special floating-point value recognised by all system using standard IEEE representation `np.nan` which can be pushed into compiled code - a virus-like object rendering any ops `NaN` hence non-error but useless
  - one solution is to use numpy `np.nansum` versions
  - special floating-point value, NO equivalent int, string etc

```python
# examples of hanlding
data.isnull()
data[data.notnull()]
# fine-conrol on dropping
# thresh sets min of non-null for keeping
data.dropna(thresh=3)
```

MULTI-INDEXING

- many constructors

```python
df.unstack() # from multi-index single-index
df.stack() # formating multi-index

# from dataframe index
index=[
    ['2010', '2010', '2011', '2011'], 
    ['US', 'Canada', 'US', 'Canada']
]

# Explicit
pd.MultiIndex.from_array([
    ['a', 'a', 'b', 'b'],
    [1, 2, 1, 2]
])
pd.MultiIndex.from_tuples([
    ('a', 1), ('a', 2), ...
])
pd.MultiIndex.from_product([
    ['a', 'b'], [1, 2]
])
pd.MultiIndex(levels=[['a', 'b'], [1, 2]], 
             labels=[[0, 0, 1, 1], [0, 1, 0, 1]])

# naming
df.index.names = ['state', 'year']
# column-wise
columns=pd.MultiIndex.from_product(..., names=[...])

# most accessing and slicing work intuitively
# get around slicing WITHIN index tuples
idx = pd.IndexSlice
df.loc[idx[:, 1], idx[:, 'HR']]

df.unstack(level=0)
df.unstack(level=1)
# retrieve original Series
df.unstack().stack()

# reset multi-index 
df.reset_index(name='population')
# returning multi-index
df.set_index(['state', 'year'])

# aggregation (GroupBy effective)
df.mean(level='year', axis=1)
```



CONCAT

- options
  - `join` - outer, inner, left, right
  - `join_axes` on column names of left df `join_axes=[df1.columns]`
  - `ignore_index=True` creating additional index for right df
  - `verify_integrity` raise error if above not specify
  - `keys` list of label names for data sources, resulting hierarchically indexed series containing data `pd.concat([x, y], keys=['x', 'y'])`

MERGE (relational algebra)

- One2One joins
  - similar to column-wise concatenation, simple `pd.merge(df1, df2)` detects common column and join as key
  - merge generally discards index, except in special case of merges by index (`left_index, right_index` keywords)
- Many2One joins
  - duplicated entries in key columns, resulting df preseves duplicate as appropriate
- Many2Many
  - if key column in both contains duplicates, resulting duplicates other columns to fulfill largest many key
- Specification of Merge Key
  - `on` taking column name or list of names as key column
  - `left_on` a name matching `right_on` if common value (often used with `.drop('one of the name', axis=1)` to drop duplicate)
  - `left_index` merges on index `True` which equals `join`
- Specifying arithmetic
  - `how='inner'` for intersection of key
- Overlapping Column Names `suffixes` keyword for renaming

```python
# example
merged.isnull().any() # see which column has NaN
merged[merged['population'].isnull()].head() # scout them out
merge.loc[merge['state'].isnull(), 'state/region'].unique() 
# see which 'state/region' also missing
final.query("year == 2010 & ages == 'total'")
```

GROUPBY: Split, Apply, Combine

- Groupby object can be operated by virtually any np-pd aggregation functions
- column indexing with string name of column `groupby('col_name')`
- iteration over groups
  -  `for (method, group) in planets.groupby('method'): print("{0:30s} shape={1}".format(method, group.shape))`
- `planets.groupby('method')['year'].describe().unstack()` produces nice matrix of summary
- `aggregate()` takes string, function or list thereof `['min', np.median, max]`
  - `{'col1': 'min', 'col2': 'max'}`
- `filter()` to drop based on group properties 

```python
def filter_sd(x):
    return x['col2'].std() > 4
df.groupby('key').filter(filter_sd)
```

- `transform()` can return some transformed version of full data to recombine, with same shape as input, e.g. normalisation
  - `.transform(lambda x: x - x.mean())`
- `apply()` takes arbitrary function to group object, with CRITERION that takes DF and ouptut pandas objects or scalar

- Split Key
  - any of list of index, dictionary or series mapping index, pure index 
  - even python function `str.lower`
  - list of valid keys `[str.lower, mapping]`
  - e.g. `decade = decade.astype(str) + 's', decade.name='decade', planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)`

PANEL

- multi-index groupby - to better present 2D groupby 

```python
# pure groupby
df.groupby(['sex', 'class'])['survived'].agg('mean').unstack()
# pivot table
df.pivot_table('survived', index='sex', columns='class')
```

- multi-level pivot table 
  - bin age using `age = pd.cut(df['age'], [0, 18, 80])`
  - `df.pivot_table('survived', ['sex', age], 'class')`
  - adding info on fare sing auto quantile `fare = pd.qcut(df['fare'], 2)` and put into `[fare, 'class']` as extra column
  - resulting 4D aggregation with hierarchical indices !!!
  - full args `fill_value` and `dropna` for handling, `aggfunc` controls type of aggregation same as agg() and `margins=True` gives total

```python
# example of birth in decades comparison
births['decade'] = 10 * (births['year'] // 10)
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')
```

EXTRA EXPLORATION

- robust outliers cutting **sigma-clipping**

```python
quartiles = np.percentile(births['births'] [25, 50, 75])
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[0])

births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')

# set 'day' to int from string
# then combine timestamps to create Date index
births.index = pd.to_datetime(10000 * births.year +
                             100 * births.month + 
                             births.day, format='%Y%m%d')
births['dayofweek'] = births.index.dayofweek
births.pivot_table('births', index='dayofweek',
                  columns='decade', aggfunc='mean').plot()
# another view by day of year
births.pivot_table('births', 
                  [births.index.month, births.index.day])
new_births.index = [pd.datetime(2012, month, day)
                   for (month, day) in births_by_date.index]
```

> robust estimate of sample mean, where 0.74 comes from interquartile range of Gaussian distribution

VECTORISED STRING

- `names.str.captialize()` given names a series of strings
- list of string ops similar to Python `len, lower, ljust, find, center, zfill, strip, swapcase, translate, islower, startswith, isnumeric, split, isspace, partition, istitle`
- regex `method()` calling `re.match()` on each element returning boolean; `extract(), findall(), replace(), contians(), count()`
  - example extracting first name `monte.str.extract('([A-Za-z]+', expand=Flase)`
  - finding all names start-end on consonant `series.str.findall(r'^[^AEIOU].*[^aeiou]$')`

- MISC ops `get()` index each element, `slice()`, `cat` concatenates strings `repeat`, `normalize` unicoding, `pad` adding whitespace, `wrap` split long strings into lines wiht lenght less than a given width, `join` and `get_dummies` 
  - example slicing first-3 char `str.slice(0, 3)`  or `.str.split().str.get(-1)` getting last name
  - `.str.get_dummies('|')` if that separates chars

MESSY DATA

```python
# recipe database to parse !curl -O URL then gunzip
# ValueError Trailing data -> 
with open('some.json') as f:
    line = f.readline()
pd.read_jsn(line).shape # 2, 12
# concat string
# read into array
with ... f:
    data = (line.strip() for line in f)
    data_json = "[{0}]".format(','.join(data))
recipes = pd.read_json(data_json) # shape 123994324, 18

# textcol.str.len().describe() reveals summary
# how many breakfast recipes
recipes.description.str.contains('[Bb]reakfast').sum()
# suggest based on ingredient
spice_df = pd.DataFrame(dict((spice,
                             recipes.ingredients.str.contains(
                             spice, re.IGNORECASE))
                            for spice in spice_list))
selection = spice_df.query('parsley & paprika & tarragon')
# finding them
recipes.name[selection.index] # wow
```

> **unfortunately, the wide variety of formats used makes this time-consuming, pointing to TRUISM in data science, cleaning and munging data often comprises the majority of work**



TIME SERIES

- native python `datetime` and `dateutil` wins on freedom loses on optimality

```python
# hand made date
from datetime import datetime
datetime(year=2015, month=7, day=4)
from dateutil import parser
date = parser.parse("4th of July, 2015")
# same object
# ops on them 
date.strftime('%A') # standard string format codes 
# COOL LIB pytz helps with time-zoning
```

- Typed arrays of times `np.datetime64` encoding dates as 64-bit integers allowing arrays of dates be repr compactly

  - `date = np.array('2015-07-04', dtype=np.datetime64)`
  - now vector-ops !!!
  - `date + np.arange(12)` gives array of following days

- `datetime64, timedelta64` built on a fundamental time unit - range of encodable time is $2^{64}$ times this fundamental unit - imposing trade-off between time resolution and max time span

  - e.g. nanosecond only with 600 years, inferred by input `np.datetime64('2015-07-04 12:00')` or add `12:59:59:50, 'ns'`
  - `Y` on/off 9.2e18 years, `M, W, D, h, m, s, ms, us, ns, ps, fs` and `as` Attosecond on/off 9.2seconds [1969 AD, 1970AD]

- Pandas best of both

  - `date = pd.datetime('4th of July, 2015')` then `date.strftime('%A')` and also numpy vector-ops `date + pd.to_timedelta(np.arange(12), 'D')`

  - shines on timestamp indexing `index = pd.DatetimeIndex(['2015-07-04', ...]` with all ops in indexing pandas and more such as `date['2015']` 

  - time stamps using `Timestamp` type - at core replacing native `datetime` but based on more efficient `numpy.datetime64` type with index structure as `DatetimeIndex`

  - time periods using `Period` type - encoding fixed-freq interval based on numpy 64 indexing with `PeroidIndex`

  - time deltas or durations using `Timedelta` - speedy based on numpy equivalent indexing with `TimedeltaIndex`

  - common to use `to_dateime()` to parse variey of format yielding `Timestamp` with single date and `DatetimeIndex` with a series of dates

    - `pd.to_datetime([dateime(2015, 7, 3), '4th of July, 2015', '2015-Jul-6', '07-07-2015', '20150708'])`

    - convertible to `PeriodIndex` with `dates.to_period('D')` as frequency

    - `TimedeltaIndex` created on subtraction `dates - dates[0]` a duration

    - regular sequences `pd.date_range()` for easy creating timestamps and `period_range()` for periods and `timedelta_range()` etc e.g. `pd.date_range('2015-07-03', '2015-07-10')` auto freq at Daily or spec num `periods=8` and `freq='H'` settings and similar in the other two objects

    - frequencies and offests `D, W, B, M, BM, Q, BQ, A, BA, H, BH, T, S, L, U, N`

      - change month usd to mark `Q-JAN, BQ-FEB` etcs
      - `W-SUN` for weekday
      - also combining with num specifying other freq e.g. 2 hours 30 mins `timedelta_range(0, periods=9, freq='2H30T')` wow!!
      - all stored as object in `pandas.tseries.offsets` and e.g. `import BDay` and do `freq=BDay()`

    - Resampling, shifting and windowing

      - `resample()` is at core data aggregation while `asfreq()` is data selection - `goog.resample('BA').mean()` and `goog.asfreq('BA')`  noting at each point `reample` gives average of previous year while `asfreq` gives value at end of year
      - up-sampling both similar though resample has more options - `data.asfreq('D', method='bfill')`
      - `tshift` is on index instead of data`goog.tshift(900)` and offtset = `pd.timedelta(900, 'D')` 
        - common usage for computing differences overtime such as ROI on last year `ROI = 100 * (goog.tshift(-365) / goog - 1)`
      - `rolling()` similar to groupby and empowered by mixing with `apply` and `agg`

      ```python
      # one-year center rolling mean and st
      rolling = goog.rolling(365, center=True)
      data = pd.DataFrame({'input': goog, 
                          'one-year mean': rolling.mean(),
                          'one-year sd': rolling.std()})
      
      # example merging
      data.columns = ['west', 'east']
      data['total'] = data.eval('west + east')
      # plotting coarser grid
      weekly = data.resample('W').sum()
      weekly.plot(style=[':', '--', '-'])
      # or rolling mean 30 days 
      daily.rolling(30, center=True).sum().plot(...)
      # smoothing by Gaussian window (50days width of 10)
      daily.rolling(50, center=True,
                   win_type='gaussian').sum(std=10).plot(...)
      
      # digging in
      # avg traffic as func of time of day
      by_time = date.groupby(date.index.time).mean()
      hourly_ticks = 4 * 60 * 60 * np.arange(6)
      by_time.plot(xticks=hourly_ticks, style=...)
      # out an strongly bimodal distribution peaking at 8am 5pm
      # as func of day of week
      by_weekday = data.groupby(data.index.dayofweek).mean()
      # compound groupby on hourly trend 
      weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')
      by_time = data.groupby([weekend, data.index.time]).mean()
      
      ```

HPC on PANDAS

- C-speed ops with allocation of intermediate arrays `eval(), query()` basing on Numexpr 
- Compound expression - since pandas **every intermediate step is explicity allocated in memory** leading to large cost for large array, numexpr compute this type of compound expression element by element without allocating full intermediate arrays

```python
# eval test
nrows, ncols = 100000, 100
rng = np.random.RandomState(42)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols)
                                  for i in range(4)))
%timeit df1 + df2 + df3 + df4
%timeit pd.eval('df1 + df2 + df3 + df4')
# 50% faster and much less memory

# supported ops
#arithmetic
('-df1 * df2 / (df3 + df4) - df5')
#boolean
('df1 < df2 <= df3 != df4')
#bitwise & | syntax on logical
# and or

# atti and indices
result1 = df2.T[0] + df3.iloc[1]
reslut2 = pd.eval('df2.T[0] + df3.iloc[1]')
np.allclose(result1, result2) # True

# further complex func NOT in eval but in Numexpr library !!!

# supports dataframe ops
# accessing directly the column names
# assignment possible
df.eval('D = (A + B) / C', inplace=True)
# local python variables 
df.eval('A + @column_mean')


# query (eval cannot expr df[(df.A)]))
df.query('A < 0.5 and B < 0.5')
# faster masking
df.query('A < @Cmean and B < @Cmean')
```

- Performance and when? check size `df.values.nbytes`
  - `eval()` can be faster even when not maxing out system memory due to how temporary df compare to size of L2 or L2 CPU cache (a few MB); if they are much bigger, then eval() can avoid some potentially slow movement of values between the different memory caches - but practically less important than memory saving

**MATPLOT**

- script usage - need `plt.show()` which starts an **event loop** looking for all currently active figure objects, opens one ore more interactive windows to display - doing lots of background calls with system graphical backend
  - ONLY ONCE per python session, often at end of script
-  verify saved figure `from IPython.display import Image` and call `Image()`
- list out save format `fig.canvas.get_supported_filetypes()`
- Two API
  - first create figure `plt.figure()` then partial `plt.subplot(2, 1, 1)` followed by plotting, moving on to next - STATEFUL interface lively tracking "current" figure and axes where all plt cmd applied - get reference to these using `plt.gcf()` and `plt.gca()` routines
  - create a grid of plots and ax will be array of Axes objects `fig, ax = plt.subplots(2)` then operate on `ax` object
- figure or an instance of `plt.Figure` via `fig = plt.figure()` is like single container for all objects of axes, graphics, text, labels; axes is the bounding box with ticks and labels containing the plot elements 
- implicit creation of figure if not set, also overlaying plots by default as current figure in sequence
- color formating `blue, g, 0.75, hex, tupe of 3 RGB, all HTML color names`
  - mixing line style `-g, --c, -.k, :r`

```python
rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8);

plt.plot(x, y, '-p', color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)

# ERRORs
plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0);

# visual GP
from sklearn.gaussian_process import GaussianProcess

# define the model and draw some data
model = lambda x: x * np.sin(x)
xdata = np.array([1, 3, 5, 6, 8])
ydata = model(xdata)

# Compute the Gaussian process fit
gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1,
                     random_start=100)
gp.fit(xdata[:, np.newaxis], ydata)

xfit = np.linspace(0, 10, 1000)
yfit, MSE = gp.predict(xfit[:, np.newaxis], eval_MSE=True)
dyfit = 2 * np.sqrt(MSE)  # 2*sigma ~ 95% confidence region

# Visualize the result
plt.plot(xdata, ydata, 'or')
plt.plot(xfit, yfit, '-', color='gray')

plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                 color='gray', alpha=0.2)
plt.xlim(0, 10);
```

**SKL**

- learning curve checks for given model, the increase in N has on train and validate score

```python
from sklearn.learning_curve import learning_curve

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree),
                                         X, y, cv=7,
                                         train_sizes=np.linspace(0.3, 1, 25))

    ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1],
                 color='gray', linestyle='dashed')

    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title('degree = {0}'.format(degree), size=14)
    ax[i].legend(loc='best')
```

> visual depiction of how model responds to N, especially when learning curve has converged, adding more N will not improve; only to use more complex model



**Naive Bayes**

- group of extremely fast and simple classification algorithms suitable for high-dimensional dataset
- $P(L | features)$ (probability of label given some observed features)
  - $P(L|features) = \frac{P(features|L)P(L)}{P(features)}$ 
  - one way to make decision on binary label is ratio of posterior probabilities $\frac{P(L_1|features)}{P(L_2|features)} = \frac{P(features|L_1)P(L_1)}{P(features|L_2)P(L_2)}$
- **generative model** is such that the Likelihood for each label is computed because it specifies the hypothetical random process that generates the data; specifying this generative model for each label is the main piece of training of such Bayesian classifier; 
  - general version of training step hard but NB simplifies premise of conditional independence given Label - product rule and addition via log form
  - various types of NB rest on different assumptions about data
- Gaussian NB - assuming data from each label drawn from simple Gaussian distribution **sans covariance between dimensions/features**, **fit by simply finding mean and standard deviation of points within each label, or all params needed to define such distribution**
  - with resulting Gaussian generative model for each label, computing likelihood for any data and thus quickly computing posterior ratio and determine which label most probable for a given point
  - natural probabilistic classification with probability of prediction conveying **uncertainty of prediction**
- Multinomial NB - features assumed to be generated from simple multinomial distribution describing counts among categories - hence suitable for features of counts or rates
  - e.g. text classification in which features are word counts within documents using sparse word count features (need convert content of string into numerical vectors - TF-IDF)

**Linear Regression**

- `model = LinearRegression(fit_intercept=True), model.fit(x[:, np.newaxis], y), xfit = np.linspace(0, 10, 1000), yfit = model.predict(xfit[:, np.newaxis])`

- multidimensional linear models akin geometrically to **fitting a plane to points in 3D or hyperplane to points in higher D**

- **basis function** higher D transformation

  - polynomial, Gaussian (customised)

  ```python
  from sklearn.base import BaseEstimator, TransformerMixin
  
  class GaussianFeatures(BaseEstimator, TransformerMixin):
      """Uniformly spaced Gaussian features for one-dimensional input"""
      
      def __init__(self, N, width_factor=2.0):
          self.N = N
          self.width_factor = width_factor
      
      @staticmethod
      def _gauss_basis(x, y, width, axis=None):
          arg = (x - y) / width
          return np.exp(-0.5 * np.sum(arg ** 2, axis))
          
      def fit(self, X, y=None):
          # create N centers spread along the data range
          self.centers_ = np.linspace(X.min(), X.max(), self.N)
          self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
          return self
          
      def transform(self, X):
          return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                   self.width_, axis=1)
      
  gauss_model = make_pipeline(GaussianFeatures(20),
                              LinearRegression())
  gauss_model.fit(x[:, np.newaxis], y)
  yfit = gauss_model.predict(xfit[:, np.newaxis])
  
  plt.scatter(x, y)
  plt.plot(xfit, yfit)
  plt.xlim(0, 10);
  ```

- EXAMPLE: Predicting Traffic

```python
# !curl -o FremontBridge.csv https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD

import pandas as pd
counts = pd.read_csv('FremontBridge.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv('data/BicycleWeather.csv', index_col='DATE', parse_dates=True)

daily = counts.resample('d').sum()
daily['Total'] = daily.sum(axis=1)
daily = daily[['Total']] # remove other columns

# account for binary col indicating day of week
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)
    


from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2016')
daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
daily['holiday'].fillna(0, inplace=True)



def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """Compute the hours of daylight for the given date"""
    days = (date - pd.datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
daily[['daylight_hrs']].plot()
plt.ylim(8, 17)



# temperatures are in 1/10 deg C; convert to C
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp (C)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])

# precip is in 1/10 mm; convert to inches
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)

daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry day']])

daily['annual'] = (daily.index - daily.index[0]).days / 365.

# fit and sans intercept because daily flags essentailly opearte as their own day-specific intercepts:
# Drop any rows with null values
daily.dropna(axis=0, how='any', inplace=True)

column_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday',
                'daylight_hrs', 'PRCP', 'dry day', 'Temp (C)', 'annual']
X = daily[column_names]
y = daily['Total']

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
daily['predicted'] = model.predict(X)

daily[['Total', 'predicted']].plot(alpha=0.5);

# evident that missing some key features esp during summer time
# either features incomplete or nonlinearity unaccounted

# check coeff of linear model to estimate how much each feature contributes to the daily bicyle count:
params = pd.Series(model.coef_, index=X.columns)
params

# these numbers hard to interpret sans uncertainty - compute using bootstrap resamplings 
from sklearn.utils import resample
np.random.seed(1)
err = np.std([model.fit(*resample(X, y)).coef_
              for i in range(1000)], 0)

# check again with errors estimated
print(pd.DataFrame({'effect': params.round(0),
                    'error': err.round(0)}))
```

- missing nonlinearity in many variables; also discarding finer-grained info such as diff rany morning and afternoon or ignored links between days - all potential effects to explore

**SVM**

- discriminative classification via line or curve (2D) or **manifold** (high dimensional) dividing classes
- **max margin** for best cutter and only important points are SV
- **kernels** same idea as basis function in LR for transforming data into higher dimension space - **radial basis function** `r = np.exp(-(X ** 2).sum(1))`
- finding perfect kernel is a problem - strategy could be compute basis function centered at every point and let SVM sift through results - known as kernel transformation as it is based on similarity relationship (or kernel) between each pair of points
- potential problem - projecting N points into N dimensions might be costly as N grows - HOWEVER due to neat little procedure known as **kernel trick** a fit on kernel-transformed data can be done implicitly - without ever building full N-D representation of kernel projection! 

- soft-margin control by C, smaller the softer for encompassing overlapping points
