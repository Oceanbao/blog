---
title: "Cython"
date: 2019-12-18T15:29:39+08:00
showDate: true
draft: false
---

# Cython Book

Cython depends on CPython implementation (C/Python Interface)

[toc]

# Chapter 1

EXAMPLE COMPARED (code)

> WHY FASTER: function call overhead, looping, math ops, stack versus heap memory allocation.

> Dynamic Lookup: all runtime definitions and object constructions -> Python interpreter checks types of object and find dunders -> unboxing objects to extract the underlying C types, and only THEN can the ops occur! C and Cython already compiled types, the addition compiles to one machine code instruction.



**Cython Wrapping C**

```c
// cfib.h

double cfib(int n);
```

```cython
# Cython wrapper

cdef extern from "cfib.h":
    double cfib(int n)
    
def fib(n):
    """Returns the nth Fib"""
    return cfib(n)
```

- `cdef extern` block uses header file and declare function's signature in the block's indented body; after it defines Python wrapper func calling C func and returning its result.
- Compile Cython code into ext module `wrap_fib` and use it in Python

```python
from wrap_fib import fib
help(fib)
fib(90)
```



# Chapter 2: Compiling and Running Cython Code

**Ways of Compilation**

- Interactively from IPython interpreter
- Compiled automatically at IMPORT time
- Separately compiled by build tools like `distutils`
- Integrated into standard build system such as `make, CMake` etc



**The Pipeline**

First - cython compiler transforms Cython source into optimised and platform-independent C or C++ !!

Second - compile the C/C++ source into SHARED LIB with a standard C/C++ compiler (platform dependent!!) `.so` file on Linux or MacOS and dynamic lib `.pyd` on Windows !!

> Flags passed to C/C++ compiler ensure this shared lib a full-fledged Python module - EXTENSION MODULE importable as if pure Python

> Cython compiler a source-to-source compiler with highly optimised code

> Once having C compiler and cython compiler (pip install cython) in place, ready to follow along with `distutils` and `pyximport` methods

## STANDARD - DISTUTILS + CYTHONIZE

One of many features of `distutils` is its ability to compile C source into an extension module, the SECOND stage in the pipeline. It manages ALL platform, architecture, and Python-version details !!

FIRST stage is handled by `cythonize` cmd, compiling Cython source (options) into C/C++ source file !!

EXPLICIT control the pipeline requiring writing small Python script and run it.

**distutils script**

- `fib.pyx` used by `distutils` to create a compiled ext-mod `fib.so` on MacOS/Linux

- control state via Python script named `setup.py`

- ```python
  # setup.py
  from distutils.core import setup
  from Cython.Build import cythonize
  
  setup(ext_modules=cythonize('fib.pyx'))
  ```

- `cythonize` the simplest call to convert Cython to C code by cython compiler (any file(s) is arg)

- `python setup.py build_ext --inplace`

- ```shell
  $ python setup.py build_ext -i
  Compiling ...
  Cythonizing fib.pyx
  running build_ext
  building 'fib' extension
  creating build
  creating build/temp.macosx-10.4-x86_64-2.7
  gcc -fno-strict-aliasing -fno-common -dynamic -g -02
  	-DNDEBUG -g -fwrapv -03 -Wall -Wstrict-prototypes
  	-I/Users/ksmith/Devel/PY64/Python.framework/Versions/2.7/include/python2.7
  	-c fib.c -o build/temp.macosx-10.4-x86_64-2.7/fib.o
  gcc -bundle -undefined dynamic_lookup
  	build/temp.macosx-10.4-x86_64-2.7/fib.o
  	-o /Users/ksmith/fib.so
  ```

  - first gcc call convert C code into an object file `fib.o`
  - second gcc convert object file into Python ext-mod `fib.so`
  - output: `fib.c`, `fib.so`, `build/` with intermediate build products

**Using Ext-Mod**

- Platform-agnostic, once compiled into ext-mod, the Python interpreter can  use

- `import fib` and `fib.fib?` shows fib as `builtin_function_or_method` indicating compiled code rather than straight Python

- When using Cython to wrap C code, must include other source files in the compilation step

- e.g compiling `cfib.c`

- ```python
  # setup_wrap.py
  # skipping same two imports
  
  # First create an Ext object with right name and sources
  ext = Extension(name='wrap_fib', sources=['cfib.c', 'wrap_fib.pyx'])
  
  # Use cythonize on the ext object
  setup(ext_modules=cythonize(ext))
  ```

  - extra step adding all required C and Cython sources

- if passing precompiled dynamic lib `libfib.so`

- ```python
  # skip two imports
  
  ext = Extension(name="wrap_fib",
                  sources=["wrap_fib.pyx"],
                  library_dirs=["/path/to/libfib.so"],
                  libraries=["fib"])
  
  setup(...)
  ```

  - naming only `wrap_fib.pyx` in the sources arg list and adding two more args to Ext object



## Interactive IPython %%cython magic

```python
%%load_ext cythonmagic

%%cython
def...
```

- generated files in `~/.ipython/cython`



## Compiling On-the-Fly with pyximport

```python
import pyximport
pyximport.install()

import fib
fib.__file__ # .../.pyxbld/lib.macos-...
```

- simple case is fine, no more setup.py

### Controlling and Managing Depens

- `.pyxdeps` to control depens of main pyx files
- `.pyxbld` to customise pyximport for diff uses
  - `make_ext(modname, pyxfilename)` - if defined, called with two str args returning `distutils.extension.Extension` instance or result of `Cython.Build.cythonize`, allowing user to customise Ext by adding files to sources
  - `make_setup_args` - if defined, pyximport calls this func with no arg to get extra arg-dcit to `distutils.core.setup` 

### Example with External Depens

- e.g. wrap fib in C, two C files - `_fib.h, _fib.c`

- `cdef extern from  "_fib.h"` in `fib.pyx` and a minimal wrapper call C 

- config by creating `fib.pyxdeps` having one line: `_fib.*`

  - This glob pattern match both C files, so pyximport will recompile `fib.pyx` whenever whichever changes

- instruct compile and link C by `fib.pyxbld`

  - ```python
    def make_ext(modname, pyxfilename):
        from distutils.extension import Extension
        return Extension(modname,
                        sources=[pyxfilename, '_fib.c'],
                         include_dirs=['.'])
    ```

  - key line is sources=[] telling distutils to compile C file and link all together

  - now whichever changes next interpreter session will mod



## Rolling Own and Compiling by Hand

For the sake of completeness, from pyx source to end

- from Cython to C
  - `cython fib.pyx`
- most common args to above:
  - --cplus, -a, -2/-3
- from C to ext_mod:
  - C to object file with proper incl and flags
  - `.o` to dynamic lib with right linking flags

```shell
CFLAGS=$(python-config --cflags)
LDFLAGS=#(python-config --ldflags)
cython fib.pyx # --> fib.c
gcc -c fib.c ${CFLAGS} # --> fib.o
gcc fib.o -o fib.so -shared ${LDFLAGS} # --> fib.so
```

- `-shared` tells gcc to create a shared library, MUST for MACOS
- Strongly recomm using the same compiler that was used to compile the Python interpreter !!!
- `python-config` gives such tailored compiler/Python version combo



## Other Build Systems

Many build tools better than distutils for depens management.

### CMake

folding Cython to standard CMake-compiled project.

### Make

Recomm to query Python interpreter to find right config/flags per above. (same as distutils.sysconfig)

- access `include` dir for `Python.h` where the Python/C API declared,
  - `INCDIR := $(shell python -c "from distutils import sysconfig; print(sysconfig.get_python_inc())")`
- acquire Python dynamic libraries to link against
  - `LIBS := $(shell python -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBS'))")`

Other config available via `get_config_var` 

## Standalone Executables

- nearly always compiled into dynamic lib and imported
- BUT compiler does have optional embedding Python Interpreter inside main 

```python
# irrationals.py

from math import pi, e

print...
```

- `cython --embed irrationals.py`
  - --> `irrational.c` with `main` entry point embedding Python interpreter
- compile it on MACOS or Linux
  - `gcc $(python-config --cflags) $(python-config --ldflags) ./irrationals.c`
  - --> `a.out` executable
  - `./a.out` 



## Compiler Directives

Comments inside code to control compilation configs. Four scopes in total.

- All can be set globally for an ext_module inside a `directive comment` must be atop before code

- e.g. `nonecheck` would be `# cython: nonecheck=True`

- e.g. turn off bounds checking for indexing `# cython: nonecheck=True, boundscheck=False` or separate lines

- CLI setting using `-X` or `-directive` OVERRIDING COMMENT SETTING

  - `cython --directive nonecheck=False source.pyx`

- Some directives support func-context scope control via decorator and ctx-mnger:

  - ```python
    cimport cython
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fast_indexing():
        ...
    ```

- Even more local control with context manager

  - ```python
    cimport cython
    
    def fast_indexing(a):
        with cython.boundscheck(False), cython.wraparound(False):
            for i in range(len(a)):
                sum += a[i]
    ```

  - Neither of above two directives is affected by comment or CLI setting



# Chapter 3: Cython in Depth

**WHY: RUNTIME INTERPRETATION VS. PRE-COMPILATION, & DYNAMIC VS. STATIC TYPING**

## The WHY

### Interpreted vs. Compiled Execution

- Before running, Python code auto-compiled to Python **bytecode**, a fundamental instructions to be executed  or **interpreted** by Python **VM**
- Since *VM* abstracts away all platform-specific details, Python bytecode can be generated anywhere, up to the VM to translate each high-level bytecode into one or more lower-level ops on OS, and ultimately the CPU.
- C has no VM or interpreter, no high-level bytecodes; C code is translated, or **compiled**, directly to **machine code** by a compiler - output executable/compiled-lib platform-architecture specific, directly run by CPU, as low-level as it gets.
- Bridging bytecode-VM and machinecode-executing CPU: Python interpreter can run compiled C code directly and transparently to the end user - must be a specific kind of dynamic library aka **extension module** - full-fledged Python modules but inside precompiled machine-code
  - Python VM no longer interprets bytecodes but runs machine-code directly - removing overhead
- Cython + C compiler convert Cython source code into compiled platform-specific extension module
- Usually 10-30 x faster just converting from Python code into an equivalent extension module
- BUT real speed comes from replacing Python's dynamic dispatch with static typing

### Dynamic vs. Static Typing

- ST requires the **type of a variable be fixed at compile time** 
- Why fast: **besides compile-time type checking**, compilers use static typing to **generate fast machine code tailored to that specific type**!
- DT has no restrictions on type, interpreter spends most time figuring out what **low-level ops to run**, and **extracting the data to give to this low-level ops**

- Python particularly flexible, a variable can be any type at any time - *dynamic dispatch* 
- Example: `a + b`
  - interpreter inspects the Python object referred to by `a` for its type, requiring at least one pointer lookup at C level
  - interpreter asks the type for an implementation of the addition method, may requiring one or more pointer lookups and internal func calls
  - if method found, interpreter has an actual func it can call, implemented either in Python or C
  - interpreter calls the addition func and passes `a, b` as args
  - addition func extractions the necessary internal data from `a, b`, may requiring several more pointer lookups and conversions from Python types to C types - if ok, only then can it run actual ops adding 
  - result then must be placed inside a (perhaps new) Python object and returned
- C compiler knows at compile time what low-level ops to run and what low-level data as args; at runtime, compiled C program skips nearly ALL steps - for numeric types math ops, compiler generates machine-code to load data into **registers, add them, store the result**



## ST Declaration

- DT variables in Cython come for free: simply assign as Python

- ```python
  cdef int i
  cdef int j
  cdef float k
  
  j = 0
  i = j
  k = 12.0
  j = 2 * k
  assert i != j
  ```

- **these ST vars have C semantics, which changes the behaviour of assignment and follow C coercion and casting rules**!

- `i = j` copies the int data at `j` to the mem_loc reserved for `i` - `i` and `j` refer to independent entities, and can evolve separately

- ```python
  def integrate(a, b, f):
      cdef int i
      cdef int N=2000
      cdef float dx, s=0.0
      dx = (b-a)/N
      for i in range(N):
          s += f(a+i*dx)
      return s * dx
  
  # or 
  def integrate(a, b, f):
      cdef:
          int i
          int N=2000
          float dx, s=0.0
          ...
  ```

- **NOTE: C static used to declare var whose lifetime extends to the entire lifetime of a program, invalid Cython key; C const declares an unmodifiable identifier, which Cython supports**

- Common cdef

- ```json
  C type					Cython cdef
  
  Pointers				cdef int *p
  						cdef void **buf
  Stack-alloc arrays		cdef int arr[10]
  						cdef double points[20][30]
  typedef alias			cdef size_t len
  Compound (structs/unions)	cdef tm time_struct
  							cdef int_short_union hi_lo_bytes
  Func Pointers			cdef void (*f)(int, double)
  ```

- tongue twisters: **func one args as func pointers returns another func pointer**

  - `cdef int (*signal)(int (*f)(int))(int)`

- Cython **auto-infer types WHEN semantics cannot be changed** (int would not be typed as C long)

  - `infer_types` directive giving Cython more leeway to infer 

  - ```cython
    cimport cython
    
    @cython.infer_types(True)
    def more_inference():
        i = 1
        d = 2.0
        c = 3+4j
        r = i * d + c
        return r
    ```

  - here `i` is typed as C long, d as double - enabling this directives put responsibility to coder to ensure that int ops **do not overflow and constant semantics**

  - useful to set globally **to test performance difference**!

### C Pointers

```cython
cdef int *p_int
cdef float** pp_float = NULL

cdef int *a, *b
```

- as with C, asterisk can be declared adjacent to the type or to the variable, though **pointerness is linked with variable, not the type**

- Dereferencing pointers is different! Since Python already uses `*args, **kwargs`, Cython **does not support `*a` syntax to dereference a C pointer**, instead **index into the pointer at location 0** to dereference a pointer in Cython!!

- ```cython
  # e.g. two pointers
  
  cdef double golden_ratio
  cdef double *p_double
  
  # assign golden_ratio ADDR to p_double using ADDR-OF operator &
  p_double = &golden_ratio
  
  # assign to golden_ratio via p_double using indexing-at-zero-to-dereference 
  p_double[0] = 1.618 # *p_double in C to dereference
  print golden_ratio # => 1.618
  
  # access p_double's REFERENT the same
  print p_double[0] # => 1.618
  ```

- Another difference arises **when using pointers to STRUCT**

  - in C if `p_st` pointer to struct `typedef`: `st_t *p_st = make_struct()`

  - then to access a struct member using arrow `int a_doubled = p_st->a + p_st->a;`

  - Cython always uses dot access nonpointer struct or not

  - ```cython
    cdef st_t *p_st = make_struct()
    cdef int a_doubled = p_st.a + p_st.a
    ```

### Mixing ST and DT (Power of Cython)

- e.g. several C ints grouping into a dynamic Python tuple

  - C code to create and init this tuple using Python/C API is tedious, which in Cython:

  - ```cython
    cdef int a, b, c
    # ...calcu 
    tuple_of_ints = (a, b, c)
    ```

  - **This works BECAUSE of linkage between C ints and Python ints, NOT if they were C pointers, which would need dereferencing them BEFORE putting into tuple, or other ways**

  - ```json
    Python types 			C types
    
    bool					bint
    int						[unsigned] char
    long					[unsigned] short
    						[unsigned] int
    						[unsigned] long
    						[unsigned] long long
    float					float
    						double
    						long double
    complex					float complex
    						double complex
    bytes					char *
    str						std::string (C++)
    unicode
    dict					struct
    ```

  - **bint**

    - int at C level converted to and fro a Python bool

  - **Integral type conversions and overflow**

    - Python 3 all int are unlimited precision
    - C checks for overflow, raise OverflowError if cannot represent Python integer
    - Directives `overflowcheck` and `overflowcheck.fold` will catch overflow errors - set to True will raise error for overflowing C integer maths ops - may help remove some overhead when enabled

  - **Floating-point**

    - float is double, conversion may truncate to 0.0 or positive/negative infinity

  - **str and unicode**

    - `c_string_type` and `c_string_encoding` directives need to be set to allow conversion

## ST with Python Type!

- possible for built-in and extension types like NumPy arrays and many others

- ```cython
  cdef list particles, mods
  cdef dict names_from_particles
  cdef str pname
  cdef set unique_particles
  ```

- Underneath, Cython **declares them as C pointers to some built-in Python struct type, can be used as is but constrained to their declared type:**

- ```cython
  particles = list(names_from_particles.keys())
  ```

- Dynamic variables can be init from ST

- ```cython
  other_particles = particles
  del other_particles[0]
  ```

- **Note: ST Python object is fixed, cannot be referring to any other Python type**

- BEHAVIOUR

  - add/sub/mul each has own behaviour: Python long coercion for large values and C overflow for limited-precision int
  - Div/Modulus **markedly different modulus with signed int**: **C rounds towards 0 while Python rounds towards infinity**
    - `-1 % 5`  => 4 in Python and => -1 in C
    - Python raise ZeroDivisionError but C has NO SUCH FAILSAFE!
  - Cython use Python semantics even when operands are ST C scalars, to turn off use `# cython: cdivision=True` or `@cython.cdivision(True)` or `with cython.cdivision(True):` context manager
  - `cdivision_warnings=True` directive emits runtime warning whenever division or modulo run on negative operands

### ST for Speed

- General principle: *the more static type info provided, the better Cython can optimize the result*

- e.g. Appending Particle object to a dynamic variable

- ```cython
  dynamic_particles = make_particles(...)
  #...
  dynamic_particles.append(Particle())
  #...
  ```

  - cython compiler generate code handling any Python object and test at runtime if `dynamic_particles` is a `list`, else as long as `append` method taking args the code will run

    - under, the code first looks up `append` attr using `PyObject_GetAttr` then calls it using the completely general `PyObject_Call` Python/C API - essentially what Python interpreter would do when running equivalent Python bytecode!

  - ```cython
    # static typing
    cdef list static_particles = make_particles(...)
    #...
    static_particles.append(Particle())
    #...
    ```

  - now Cython can generate specialised code directly calling either `PyList_SET_ITEM` or `PyList_Append` from the C API - this is what the above `PyObject_Call` ends up calling *anyway*, but ST allows Cython to remove dynamic dispatch on `static_particles`

- currently supported built-in ST Python types

  - `type, object, bool, complex, basestring, str, unicode, bytes, bytearray, list, tuple, dict, set, frozenset, array, slice, date, time, datetime, timedelta,  tzinfo`
  - For Python types with direct C types `int, long, float` turn out not easy, BUT the need to do so is rare - often simply declare regular C `int, long, float, double` and let Cython do auto-convert to and fro Python
    - Python `float` -> C `double`, hence C `double` is preferred whenever conversions to and fro Python used to ensure **no clipping of values or loss of precision**
    - Python 3 at C level, all integers are `PyLongObjects`
    - Cython properly converts between C integral types and these Python types in a language-agnostic way, raising `OverflowError` when impossible

### Reference Counting and Static String Types

- One of Python's major features is **auto MEM management** - CPython uses plain **reference counting** with auto-GC runs periodically to clean up unreachable reference cycles!

- Cython auto manage (ST or DT) Python object ensuring cleaning up

- Implication in mixing types - if having two Python `bytes` objects `b1, b2` to extract underlying `char` pointer after adding:

  - ```cython
    b1 = b"All men are normal"
    b2 = b"Socrates is a man"
    cdef char *buf = b1 + b2
    ```

  - `b1 + b2` is a temporary Python `bytes` object and the assigning attemps to extract its `char` pointer using Cython's auto-convert rules

  - Because the result of the addition is temporary object, the above code cannot work - temporary result of addition is **deleted at once after created**, Cython is able to catch the rror and issue compilation error

  - the correct way - use a temporary Python variable, either ST or DT

    - ```cython
      # DT
      tmp = s1 + s2
      cdef char *buf = tmp
      # ST
      cdef bytes tmp = s1 + s2
      cdef char *buf = tmp
      ```

  - These cases uncommon, only due to C-level object is **referring** to data managed by Python object - since Python object owns the underlying string, the C `char *` buffer has no way to tell Python that it has another (non-Python) reference - creating temporary `bytes` so that Python does not delete the string data, must ensuring it is kept so long as the C `char *` buffer is required! 

  - **The other C types (table above) are all value types, not pointer types - Python data is copied during assignment (C semantics) allowing C variable to evolve separately from Python object used to init it!**



## Cython's 3 kinds of FUNC

- Much of above variable typing mix applies to functions in Cython.
- Python function is **first-class citizens** - meaning they are objects with state and behaviour!
  - created both at import and dynamically at runtime
  - created anonymously with lambda 
  - defined inside another func
  - returned from other func
  - passed as an arg to other func
  - called with positional and keyword args
  - defined with default values
- C func has minimal call overhead, orders of magnitude faster
  - can be passed as arg to other func (but doing so is much more cumbersome than in Python)
  - cannot be defined inside another
  - has statically assigned name unmodifiable
  - takes only positional arg
  - does not support default values for parameters

### Python func with def

- regular Python func

  - takes DT python object only
  - used the same way regardless how it's imported
  - compiled using any method, e.g. as `fact.pyx` with `import pyximport; pyximport.install(); import fact`

- pure Python func compiled by Cython is faster somewhat, the source of speedup is the **removal of interpretation overhead and the reduced function call overhead in Cython**

- different implementation:

  - Python func has type `function`, Cython has type `builtin_function_or_method`
  - Python has several attributes available (`__name__` , etc) and modifiable, Cython unmodifiable
  - **Python execute bytecodes with Python interpreter, Cython runs compiled C code that calls into Python/C API, bypassing bytecode interpretation entirely!**

- One nice feature of Python `int` is that they can represent arbitrarily large values (to MEM), convenient but cost speed

- Cython version with C `long` integral type is faster but with limited-precision possibly overflow!

  - ```cython
    def typed_fact(long n):
        """Compute n!"""
        if n <= 1:
            return 1
       return n * typed_fact(n - 1)
    ```

  - As `n` is func arg, omitting `cdef` 

  - this code is no faster, since it's a Python func with Python return object - lots of code generated to extract the underlying C long from Python integer returned, ops and packing/unpacking essentially ends up the same code paths as pure

- Cython allows ST args to have default, as positional or keyword!

- One way is to do recursive, but later. The idea is to let C do the hard work and trivially convert result to Python object- cdef

### C Funcs with cdef

- `cdef` C-calling semantics taking and returning ST objects, work with C pointers, structs, other C types cannot be auto-coerced to Python types! (think `cdef` as C function defined with Cython's Python-like syntax)

- Note: **can declare and use Python objects, but `cdef` functions are typically used when needing to get close to C without writing C**

- Mixing function types same as variable in the same source file - returning ST can be any of `pointers, structs, C array, static Python list, dict or void` or ommitted defaulted to `object`

- **Cython DOES NOT allow cdef to be called from external Python code**, therefore they are typically used as fast auxiliary functions to help `def` 

- using it outside the extension_module:

  - ```cython
    # need minimal def calling it internally
    def wrap_c_fact(n):
        return c_fact(n)
    ```

  - BUT: limited to C integral types only - depending on how large `unsigned long` is on OS

    - one option to partially address this is to use `double`

### Combining as cpdef

- **hybrid, getting C-only function and a Python wrapper for it, both with the same name**
- calling it from Cython calls C-only version, and wrapper if called in Python
- **inline** key in C/C++ suggests to compiler to replace the so-delcared func with its body whevere it is called, thereby further removing call overhead `cdef inline long c_fact(long a):` , useful for small inlined func called in deeply nested loops !!!
- LIMIT: its args must be compatible with both Python and C types - **any Python object can be repr at C level (DT or ST built-in), but NOT ALL C types can be in Python**, cannot use `void, pointer, array, etc` in `cpdef`

### Exception Handling

- `def` always returns some sort of `PyObject` pointer at C level - this invariant allows Cython to correctly propagate exceptions from `def` without issues

- `cdef, cpdef` may return non-Python type, making other tracing mechanism necessary

- e.g. `ZeroDivisionErro` will be set but has no way for `cpdef` to communicate to caller

- Cython `except` 

  - ```cython
    cpdef int divide_ints(int i, int j) except? -1:
        return i / j
    ```

  - `except? -1` returns `-1` to act as a possible sentinel for exception occurance (`-1` is arbitrary, any integer literal within the range of values for the return type is ok)

  - `?` used here since `-1` might be a valid result, if any return value always indicates an error, it can be omitted; OR use `except *` to check regardless of return value but incur some overhead

### Embedsignature Compiler Directive

- Pure Python has introspection on signature `pure_func?` namely *definition*
- Cython func do have standard docstring but do not include a signature by default
  - `embedsignature` directive set to True enable such

> **Generated C Code**
>
> ```python
> def mult(a, b):
>     return a * b
> 
> # compare with ST 
> ...(int a, int b):
>     ...
> ```
>
> `cython mult.pyx` => `mult.c`
>
> Several thousand lines long! 
>
> ```c
> /* "mult.pyx":3
> *
> * def mult(a, b):
> *	return a * b 		# <<<<<<<<<
> */
> __pyx_t_1 = PyNumer_Multiply(__pyx_v_a, __pyx_v_b);
> if (unlikely(!__pyx_t_1)) {
>     __pyx_filename = __pyx_f[0];
>     __pyx_lineno = 3;
>     __pyx_clineno = __LINE__;
>     goto __pyx_l1_error;
> }
> 
> # compare with ST
> ...
>     __pyx_t_1 = __Pyx_PyInt_From_int((__pyx_v_a * __pyx_v_b));
> ```



## Type Coercion and Casting

- Both C and Python have well-defined coercion between numeric types

- Explicit casting between types is common in C, esp. in pointers

- Cython replaces parentheses with angle brackets

- `cdef int *ptr_i = <int*>v` ===> `int *ptr_i = (int*)v'` in C

- Explicit casting in C is not checked, given total control over to type

  - e.g. emulating `id` function

  - ```cython
    def print_addr(a):
        cdef void *v = <void*>a
        cdef long addr = <long>v
        print("Cython addr:", addr)
        print("Python id  :", id(a))
        
        
    # use it
    import pyximport; pyximport.install()
    
    import casting
    
    casting.print_addr(1) # same
    ```

- Can casting with Python extension types, built-in or defined

  - ```cython
    def cast_to_list(a):
        cdef list cast_list = <list>a
        print(type(a))
        print(type(cast_list))
        cast_list.append(1)
    ```

  - it takes a Python object of any type and cast it to a static `list`!

  - Cython will treat `cast_list` as a list in C, calling either `PyList_SET_ITEM` or `PyList_Append` at last line

  - OK so long as the args is a `list` or a subtype, raising `SystemError` else

  - **such bare casts are appropriate only when we are certain that the object being cast has a compatible type**

  - when less certain let Cython let before casting

    - ```cython
      def safe_cast_to_list(a):
          cdef list cast_list = <list?>a
          ...
      ```

    - raising `TypeError` when `a` is not `list` or subtype at casting

- Casting also comes into play when working with **base and derived classes in extension type hierarchy**



## Structs, Unions, Enums

- for **un-typedef C struct or union declaration**

  - ```c
    struct mycpx {
        int a;
        float b;
    };
    
    union uu {
        int a;
        short b, c;
    };
    ```

  - ```cython
    # same in Cython
    cdef struct mycpx:
        float real
        float imag
        
    cdef union uu:
        int a
        short b, c
    ```

  - another case where Cython blends Python with C: using Python-like blocks to define C constructs!

  - ```cython
    # combine both with ctypedef creating a new type alias for either
    ctypedef struct mycpx:
        float real
        float imag
    ...
    
    # to declare a variable with the struct type, simply use cdef
    cdef mycpx zz
    
    # init in 3 ways:
    
    # 1: struct literals
    cdef mycpx a = mycpx(3.1415, -1.0)
    cdef mycpx b = mycpx(real=2.718, imag=1.618034)
    # another blend of Python and C++ constructs
    
    # 2: struct fields assignment
    cdef mycpx zz
    zz.real = 3.1415
    zz.imag = -1.0
    # for init, 1 is handy, but direct assign can be used to update field
    
    # assigning from a Python dict
    cdef mycpx zz = {'real': 3.1415, 'imag': -1.0}
    # this uses Cython's auto-convert to do each assign auto, but more overhead!!
    ```

- Nested and anonymous inner struct/union ARE NOT SUPPORTED! Need to un-nest and provide dummy names:

  - ```c
    struct nested {
        int outer_a;
        struct _inner {
            int inner_a;
        } inner;
    };
    ```

  - ```cython
    # cython
    cdef struct _inner:
        int inner_a
    cdef struct nested:
        int outer_a
        _inner inner
        
    # can init nested on per-filed basis or assigning to a nested dictionary matching nested
    
    cdef nested n = {'outer_a': 1, 'inner': {'inner_a': 2}}
    ```

- `enum` defined by members on separate lines or comma-limited

  - ```cython
    cdef enum PRIMARIES:
        RED = 1
        YELLOW = 3
        BLUE = 5
    cdef enum SECONDARIES:
        ORANGE, GREEN, PURPLE
    ```

  - can be declared either `ctypedef` or `cdef` as in struct/union

  - anonymous enums are useful to declare **global integer constant**

  - ```cython
    cdef enum:
        GLOBAL_SEED = 37
    ```

- struct/union/enum will be used more when interfacing with external code

### Type Aliasing with ctypedef

- Another C feature is type aliasing with `ctypedef` used similarly as C's `typedef`, essential when interfacing with external code using `typedef` aliases

- ```cython
  ctypedef double real
  ctypedef long integral
  
  def displacement(real d0, real v0, real a, real t):
      """Compute displacement under constant acceleration"""
      cdef real d = d0 + (v0 * t) + (0.5 * a * t**2)
      return d
  ```

  - `ctypedef` aliases allow switching the precision of calcu from double precision to single precision by changing a single line of program - Cython auto-convert Python numeric types and these `ctypedef` aliase
  - more useful in C++ when `typedef` aliases can significantly shorten long templated types
  - must occur at file scope and CANNOT be used inside a function or other local scope 

> **FUSED TYPES AND GENERIC CODING**
>
> - Cython has novel typing **fused types** allowing reference to several reltaed types with a single type definition
>
>   ```cython
>   # max for integral values
>   
>   from cython cimport integral # fused type for C short, int, long scalar types
>   
>   cpdef integral integral_max(integral a, integral b):
>       return a if a >= b else b
>   
>   # Cython creates 3 integral_max: one for both as shorts, as ints, as longs
>   # using long if called from Python
>   # when called from other Cython ode, check arg type at compile time to set which to use
>   
>   # allowed uses
>   cdef allowed():
>       print(integral_max(<short>1, <short>2))
>       print(integral_max(<int>1, <int>2))
>       ...
>   
>   # CANNOT mix for the same fused type from other Cython code
>   
>   # to generalise support floats and doubles, cannot use cython.numeric fused type since complex are not comparable
>   # Can create own fused type to group integral and floating C types
>   
>   cimport cyhton
>   
>   ctypedef fused integral_or_floating:
>       cython.short
>       cython.int
>       cython.long
>       cython.float
>       cython.double
>       
>   cpdef integral_or_floating generic_max(integral_or_floating a,
>                                         integral_or_floating b):
>       return a if a >= b else b
>   
>   # five spec, one each C type in ctypedef fused block
>   ```



## Loop and While

- if index `i` and `range` key `n` are DT, Cython may NOT be able to gen fast C loop; fixed by typing

- ```cython
  cdef unsigned int i, n = 100
  for i in range(n):
      #...
  ```

### Guidelines for Efficient Loops

- TYPE `range` args as C int (Cython auto-gen `i` as int as well, **provided not using index in body expression**)

- if certain index used in body will not cause overflow, ST `... a[i] = i + 1` for example

- **when looping over container (`list, tuple, dict`, etc) ST index may introduce MORE overhead, depending on situation** - **CONVERT TO C++ EQUIVALENT or TYPED MEMORYVIEWS** (more optimising loop bodies later in Cython NumPy support and typed memoryviews)

- `while` loops must make loop condition expression efficient - may use typed variables and `cdef` functions (**simple `while True` loop with an internal `break` are efficiently translated to C automatically**)

- Examples

  - ```cython
    # Python
    n = len(a) - 1
    # "a" list or array of floats
    for i in range(1, n):
        a[i] = (a[i-1] + a[i] a[i+1]) / 3.0
    
    # access `i` indices each iteration prevents direct iteration!
    # almost Cython-friendly, only need to ad some typing for fast loop
    cdef unsigned int i, n = len(a) - 1
    for i in range(1, n):
        ...
    ```

## Cython Preprocessor (C macro)

- `DEF` creates macro, a compile-time symbolic constant akin to `#define` in C
- **useful for giving meaningful names to magic numbers, allowing them to be updated and changed in a single location**
- **textually substituted with their value at compile time**

```cython
DEF E = 2.718281828459045
DEF PI = 3.14159263589

def feymans_jewel():
    """Returns e**(i*pi) + 1. Should be ~0.0"""
    return E ** (1j * PI) + 1.0
```

- DEF constant must resolve at compile time and are restricted to simple types

- can be **literal integrals, floats, strings, predefined DEF variables calls to a set of predefined functions, or expressions involving these types**

- set of predfined compile-time names from `os.uname`

  - `UNAME_SYSNAME, UNAME_RELEASE, UNAME_VERSION, UNAME_MACHINE, UNAME_NODENAME`

- types available for defining a DEF

  - Constants - `None, True, False`
  - Built-in functions - `abs, chr, cmp, divmod, enumerate, hash, hex, len, map, max, min, oct, ord, pow, rnage, reduce, repr, round, sum, xrange, zip`
  - Built-in types - `bool, complex, dict, float, int, list, long, slice, str, tuple`

- **RHS of DEF MUST ultimately eval to `int, float, string` object**

- conditional preprocessors are not restricted like DEF constants

  - e.g. branching on OS

  - ```cython
    IF UNAME_SYSNAME == "Windows":
        # ...Windows-specific code...
    ELIF UNAME_SYSNAME == "Darwin":
        # ...Mac-specific code...
    ...
    ```



# Chapter 4 Practice: N-Body Simulation

- variables: initial positions, velocities in heliocentric coord-system using symplectic integrator (time-stepping scheme good for computing right trajectories)

```python
# Python

def main(n, bodies=BODIES, ref='sun'):
    # takes number of steps n to integrate initial conditions 
    # of bodies with reference body i.e. the Sun
    
    # gets list of all bodies and makes pairs of all for ease of iteration
    system = list(bodies.values()) 
    pairs = combinations(system)

    # correct Sun's momentum to stay at the Cen of Mass
    offset_momentum(bodies[ref], system)
    
    # compute and print system's total energy
    report_energy(system, pairs)
    
    # Symplectic integrators good at conserving energy, used to test accuracy
    # after getting init energy, core compute in time step
    # unit of time is mean solar day, distance is one astro-unit, solar mass 
    advance(0.01, n, system, pairs)
    report_energy(system, pairs) # should be close to pre-advance
```

- `time python nbody.py 500000` runs 13.21s user

- first, profile `ipython --no-banner` and `%run -p nbody.py 500000`

  - `advance()` consumes 99% of runtime, hence need ST and more efficient data structures

- then convert straight to `.pyx` and compile-run to check working

  - ```python
    # setup.py
    from distutils.core import setup
    from Cython.Build import cythonize
    
    setup(name='nbody',
         ext_modules=cythonize('nbody.pyx'))
    ```

  - need a driver script to run the main function inside the extension module

  - ```python
    # run_nbody.py
    import sys
    from nbody import main
    
    main(int(sys.argv[1]))
    ```

  - build `python setup.py build_ext -i`

  - resulting runtime at 4.78s user, 2.8 times faster - **WITHOUT ANY CHANGE, Cython provides this for free!**

## Python Data Structure and Organisation

- In Python, each body is a tuple of 3 elems, e.g. Sun's init condition

- ```python
  ([0.0, 0.0, 0.0], # pos
   [0.0, 0.0, 0.0], # velocity
   SOLAR_MASS # mass
  )
  
  # Jupiter's
  ([ 4.84143144246472090e+00,
  -1.16032004402742839e+00,
  -1.03622044471123109e-01],
  [ 1.66007664274403694e-03 * DAYS_PER_YEAR,
  7.69901118419740425e-03 * DAYS_PER_YEAR,
  -6.90460016972063023e-05 * DAYS_PER_YEAR],
  9.54791938424326609e-04 * SOLAR_MASS),
  ```

- `system` is a list of these tuples, `pairs` a list of all pairs of these tuples

- **simulation accesses and updates the pos/velo of all planets often, so optimizing their representation is essential!**

- `advance` loop over all steps, each looping over all pairs of bodies

  - ```python
    def advance(dt, n, bodies, pairs):
    	for i in range(n):
    		for (([x1, y1, z1], v1, m1),
    			([x2, y2, z2], v2, m2)) in pairs:
    				# ...update velocities...
            for (r, [vx, vy, vz], m) in bodies: # update velocities
                r[0] += dt * vx
                r[1] += dt * vy
                r[2] += dt * vz
                
    ```

  - **`bodies` and `paris` seq are set up to refer to the same objects, so updating both is possible in sequence, even though looping over difference seq**

  ## Converting Python Data to structs

  - Strategy for speed: convert **list-of-tuples-of-lists-of-floats** into **C array of C structs**

  - **C struct will have much better R/W performance, as using fast C iteration and lookups, rather than the general sloow Python interpreter**

  - ```cython
    # body_t, two double arrays for pos and velo, and a single double for mass
    
    cdef struct body_t:
        double x[3]
        double v[3]
        double m
    ```

  - **leave most of Python code intact and ONLY use struct where performance matters !!**

  - `advance` needs to convert Python list-tuples of bodies into C struct using `cdef` function pair for conversion!

  - ```cython
    # first, list to struct
    
    cdef void make_cbodies(list bodies, body_t *cbodies)
    # loops over the bodies list and init prealloc cbodies array with Python's list data
    
    cdef void make_cbodies(list bodies, body_t *cbodies, int num_cbodies):
        cdef body_t *cbody
        for i, body in enumerate(bodies):
            if i >= num_cbodies:
                break
            (x, v, m) = body
            cbody = &cbodies[i]
            cbody.x[0], cbody.x[1], cbody.x[2] = x
            cbody.v[0], cbody.v[1], cbody.v[2] = v
            cbodies[i].m = m
    
    # its complement, converting body_t array into Python list of tuples
    
    cdef list make_pybodies(body_t *cbodies, int num_cbodies):
        pybodies = []
        for i in range(num_cbodies):
            x = [cbodies[i].x[0], cbodies[i].x[1], cbodies[i].x[2]]
            v = [cbodies[i].v[0], cbodies[i].v[1], cbodies[i].v[2]]
    		pybodies.append((x, v, cbodies[i].m))
    	return pybodies
    
    # ready to convert for loops in advance to use ST
    
    def advance(double dt, int n, bodies):
        # Note this does NOT take a `pairs` arg, and returns a new `bodies` list.
        # This is slightly diff from original 
        cdef:
            int i, ii, jj
            double dx, dy, dz, mag, b1m, b2m,
            body_t *body1
            body_t *body2
            body_t cbodies[NBODIES]
    
    	make_cbodies(bodies, cbodies, NBODIES)
    
        for i in range(n):
            for ii in range(NBODIES-1):
                body1 = &cbodies[ii]
                for jj in range(ii+1, NBODIES):
                    body2 = &cbodies[jj]
                    dx = body1.x[0] - body2.x[0]
                    dy = body1.x[1] - body2.x[1]
                    dz = body1.x[2] - body2.x[2]
                    mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
                    b1m = body1.m * mag
                    b2m = body2.m * mag
                    body1.v[0] -= dx * b2m
                    body1.v[1] -= dy * b2m
                    body1.v[2] -= dz * b2m
                    body2.v[0] += dx * b1m
                    body2.v[1] += dy * b1m
                    body2.v[2] += dz * b1m
            for ii in range(NBODIES):
                body2 = &cbodies[ii]
                body2.x[0] += dt * body2.v[0]
                body2.x[1] += dt * body2.v[1]
                body2.x[2] += dt * body2.v[2]
    
    return make_pybodies(cbodies, NBODIES)	
    ```
  
  - **for loop converted to nested for loops over indices into C array of body_t structs**, **using two body_t pointers to refer to the current bodies in the pair**
  
  - removed pairs arg to advance, so need to update main

### Running Cythonized version

- 0.54s user - 25 times faster
- serial hand-written C runs 0.14s user, 4 times faster
- C code `advance` uses `sqrt` to compute distance! the above uses `**` operator and `pow` => straightforwad to use sqrt `ds = dx * dx + dy * dy + dz * dz; mag = dt / (ds * sqrt(ds))` 
- this requires type `ds` as a double and add a cimport `from libc.math cimport sqrt` for C speed
- with this change, the final speed is 0.15s user, 90 x faster than pure Python

## Conversion Summary

1. Profile using `cProfile` or `%run` to see most runtime
2. Inspect hotspots for **nested loops, numeric-heavy ops, nested Python containers**, all can be easily converted with Cython to use more efficient C constructs (above case have all)
3. Use Cython to declare C data structures equivalent to Python's. Create converters if needed to transform Python data to C data. In the N-body simu, `body_t` struct to represent nested that-long-name, getting better data locality and fast access! Two converters, `make_cbodies, make_pybodies` to convert to and fro both
4. Convert hotspots to use C data, removing Python data from nested loops to the extent possible - ensuring all variables used in nested loop (including loop variables) are ST! 
5. Test to check semantics intact



## Detail Code Study

```cython
# cython setup.py
from distutils.core import setup
from Cython.Build import cythonize

setup(name="nbody",
      ext_modules=cythonize("nbody.pyx"))

# global definition the same; maybe Cython could do typed

# Python def BODIES
BODIES = {
    'sun': ([....]),
    'jupiter': ...
}
# Cython need struct
cdef struct body_t:
    double x[3]
    double v[3]
    double m

DEF NBODIES = 5

BODIES = {
    ... # same as python
}

# Python
def advance... # see above

# Python
def report_energy(bodies, pairs):

    e = 0.0

    for (((x1, y1, z1), v1, m1),
            ((x2, y2, z2), v2, m2)) in pairs:
        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
    for (r, [vx, vy, vz], m) in bodies:
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
    print("%.9f" % e)
    
# Cython
def report_energy(bodies):
    # no `pairs` arg but compute internally
    e = 0.0
    paris = combinations(bodies)
    for ...
    # the same
    
# Cython auxiliaries
cdef void make_cbodies(list bodies, body_t *cbodies, int num_cbodies):
    cdef body_t *cbody
    for i, body in enumerate(bodies):
        if i >= num_cbodies:
            break
        (x, v, m) = body
        cbody = &codies[i]
        cbody.x[0], cbody.x[1], cbody.x[2] = x
        cbody.v[0], cbody.v[1], cbody.v[2] = v
        cbodies[i].m = m
        
cdef list make_pybodies(body_t *cbodies, int num_cbodies):
    pybodies = []
    for i in range(num_cbodies):
        x = [cbodies[i].x[0], cbodies[i].x[1], cbodies[i].x[2]]
        v = [cbodies[i].v[0], cbodies[i].v[1], cbodies[i].v[2]]
        pybodies.append((x, v, cbodies[i].m))
    return pybodies

# the same 
def offset_momentum(ref, bodies):

    px = py = pz = 0.0

    for (r, [vx, vy, vz], m) in bodies:
        px -= vx * m
        py -= vy * m
        pz -= vz * m
    (r, v, m) = ref
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m

    
# python
def main(n, bodies=BODIES, ref='sun'):
    system = list(bodies.values())
    pairs = combinations(system)
    offset_momentum(bodies[ref], system)
    report_energy(system, pairs)
    advance(0.01, n, system, pairs)
    report_energy(system, pairs)

if __name__ == '__main__':
    main(int(sys.argv[1]), bodies=BODIES, ref='sun')

# cyhton
def main(n, bodies=BODIES, ref='sun'):
    system = list(bodies.values())
    offset_momentum(bodies[ref], system)
    report_energy(system)
    system = advance(0.01, n, system)
    report_energy(system)

# DIFF: separete run_nbody.py using pyx (after build)
import sys
from nbody import main
main(int(sys.argv[1]))
# Makefile
all: nbody.so
nbody.so: nbody.pyx
    cython -a nbody.pyx
    python setup.py build_ext -fi
clean:
    -rm -r build nbody.so nbody.c nbody.html
```



# Chapter 5 Extension Type

## Comparing Py-Class and Ext Type

- Python all is object - WHAT? 
  - object has **identity, value, type**  - typically value or data is under `__dict__` ; 
  - a type is responsible for creating and destroying its objects ! with `class`
  - how Cython allows low-C access to an object's data and methods, benefit?
  - built-in types in Python all implemented at C level via Python/C API, incorporated into the Python runtime.
  - they behaves just like regular Python classes
- **Extension types** - self-made C-level types via API
  - when using them, running compiled and ST code ! - having fast C-level access to types' methods and instances' data!
  - direct Python/C API creation is not for the uninitiated, hence Cython `cdef class`

```cython
# Example
# when compiled to C, resulting cls just regular Python, NOT ext_type!
# small boost due to pre-compilation skipping interpreter overhead
# real boost comes with ST
cdef class Particle:
    cdef double mass, position, velocity
	# the rest of __init__ and get_momentum(self) INTACT
```

- **the `cdef` type declarations in class body are NOT class-level attributes!! They are C instance attributes!! similar to C++ declaration.** 
- **ALL INSTANCE ATTRIBUTES MUST BE DECLARED WITH `CDEF` AT THE CLASS LEVEL IN THIS WAY FOR EXTENSION TYPES** 
- **CANNOT ACCESS EXT_TYPE CLASS ATTRIBUTES NOR ASSIGN NEW OR MOD!!**
  - when ext_type init, a C `struct` is **allocated and init** requiring the **size and fields** be known at COMPILE TIME, hence need to declare ALL attributes with cdef
  - In contrast, Py-object init creating at interpret time `__dict__` attribute
  - C structs are fixed and not open to new members! **private by default**  - python class is wide open

## Type Attributes and Access Control

- py-class attribute goes through a general lookup process that works for any attribute, instance attribute or method or data attribute inside base cls etc. - several level of indirection to search for target = overhead

- **Methods defined in `cdef class` extension types have FULL ACCESS to ALL instance attributes - cython will translate any accesses like `self.mass` into low-level accesses to C-struct fileds!** bypassing lookup

- WHAT IF wanting access? Make **read-only or readable and writable**

  - ```cython
    # read-only attributes
    cdef class Particle:
        cdef readonly double mass
        cdef double position, velocity
    # need recompile 
    
    # R/W attributes
    cdef class Particle:
        cdef public double mass
        # ...
    	# cy_particle.mass = 12-6
    ```

    - when calling method Cython still uses fast C-level direct access and essentially ignoring readonly and pubic declarations - exist only to allow and control access from Python runtime!

## C-Level INIT and FINAL

- C-struct in ext_type has implications: 

  - when Python calls `__init__`, the `self` arg is required to be a valid instance of that ext_type; `__init__` call typically init attributes on `self` arg

  - at C, before `__init__` called, the instance's struct must be ALLOC, all struct fields must be in a valid state, ready to accept initial values

  - Cython adds a special `__cinit__` performing C ALLOC and INIT

  - **the above `Particle.__init__` can take on this role because the fields are all `double` scalars and require no C ALLOC** , BUT possible, depending on how ext_type is **subclassed or alternative constructors**, for `__init__` to be called multiple times during object creation, and other cases where `__init__`` is bypassed entirely

  - Cython guarantees that `__cinit__` called exactly once and called before `__init__`, `__new__` or alternative Python-level constructors (e.g. `classmethod` constructors); Cython passes any INIT args into `__cinit__`

  - ```cython
    # instances with internal C array, dynamically allocated
    cdef class Matrix:
        cdef:
            unsigned int nrows, ncols
            double *_matrix
    	# the CORRECT place to put `self._matrix` dynamically allocation is `__cinit__`
    	cdef __cinit__(self, nr, nc):
            self.nrows = nr
            self.ncols = nc
            self._matrix = <double*>malloc(nr * nc * sizeof(double))
            if self._matrix == NULL:
                raise MemoryError()
    	# deconstrutor
        def __dealloc__(self):
            if self._matrix != NULL:
                free(self._matrix)
    ```

    - if `self._matrix` allocated inside `__init__` instead, it would never be called - which can occur with an alternate `classmethod` constructor - then any method using `self._matrix` would lead to ugly segmentation faults
    - conversely, if `__init__` called twice - perhaps due to inconsistent use of `super` in a class hierarchy - then a MEM_LEAK would result (particularly hard to track down)
    - CLEANUP: Cython uses C FIN `__dealloc__` taking care of undoing `__cinit__` (see above)

## cdef and cpdef Methods

- concepts in CH3 about `def, cdef, cpdef` **functions** also apply to **extension type methods**; NOTE cannot use `cdef, cpdef` to define methods on non-cdef classes

- **cdef methods has C calling semantics, just as cdef functions do: all args are passed in as is, no type mapping from Python to C!**

- hence cdef methods cannot be accessed from Python

- `cpdef` method is particularly useful - callable from external Python code and other Cython code!

- **HOWEVER, the input/output types have to be automatically convertible to and fro Python objects (no pointer types for example)**

- ```cython
  # cpdef method in Particle
  cpdef double get_momentum(self):
      return self.mass * self.velocity
  
  # a function
  def add_momentums(particles):
      total_mom = 0.0
      for particle in particles:
          total_mom += particle.get_momentum()
  	return total_mom
  
  # faster with typing
  def add_momentums_typed(list particles):
      cdef:
          double total_mom = 0.0
          Particle particle
  	for particle in particles:
          total_mom += particle.get_momentum()
  	return total_mom
  
  # NOTE: arg type is list, double and CRUCIALLY, the loop indexing variable particle as a Particle !!
  # hence when get `get_momentum` called in this typed func (def)!! no Python objects are invovled !!
  # Even the in-place sum is C-only operation !! since total_mom is ST C double
  
  ```

  - this could be defined in interpreted Python, or it could be compiled and run by Cython - either case, the call to `get_momentum` is a fully general Python attribute lookup and call, since Cython does not know that `particles` is list of `Particle` objects
  - **when Python calls get_momentum on a Particle object, the get_momentum Python wrapper is used, and the correct packing and unpacking from Python object to underlying Particle struct occurs auto**

- to see the effect of `cpdef` over `def` method, remove the `Particle particle` declaration, forcing Cython to use Python calling semantics on `particle.get_momentum()` - slower than ALL-PYTHON version! `particle` as loop variable here yields the most speed, typing `particles` and `total_mom` has less of an effect

- WHAT IF `get_momentum` is `cdef` method? 

  - ```cython
    # another method
    cdef class Particle:
        cdef double mass, position, velocity
        # ...
        cdef double get_momentum_c(self):
            return self.mass * self.velocity
    
    # will have to modify add_momentums_typed 
    def add_momentums_typed_c(list particles):
        cdef:
            double total_mom = 0.0
            Particle particle
    	for particle in particles:
            total_mom += particle.get_momentum_c()
    	return total_mom
    ```

    - **fastest! (40%) BUT DOWNSIDE is that `get_momentum_c` is NOT callable from Python, only Cython - since these are trivial, the speedups are skewed heavily toward function call overhead. For more heavy calculations, the speed difference between `cdef` and `cpdef` will be INSIGNIFICANT, while the FLEXIBILITY `cpdef` becomes more relevant**

## Inheritance and Subclassing

- **CAN SUBCLASS a SINGLE BASE TYPE**, which must itself be a type implemented in C - either built-in or another ext_type ! 

- For example, subclass of `Particle, CParticle`, storing particle's momentum rather than computing it on the fly - do not want duplicate work done in Particle, so subclass it

- ```cython
  cdef class CParticle(Particle):
      cdef double momentum
      def __init__(self, m, p, v):
          super(CParticle, self).__init__(m, p, v)
          self.momentum = self.mass * self.velocity
  	cpdef double get_momentum(self):
          return self.momentum
  ```

  - Because `CParticle` is a more specific `Particle`, everywhere using a `Particle`, should be able to sub in `CParticle` without code change - all while we revel in Platonic beauty of polymorphism.

  - **In all above methods/functions can pass in a list of `CParticle` - the `add_momentums` function does everything with dynamic Python variables, so all follows Python semantics there - BUT `add_momentums_typed` expects the elements of the list to be `Particle` instances, when `CParticles` passed in, the RIGHT version of `get_momentum` is RESOLVED, bypassing the Python/C API**

  - ```cython
    # subclass Particle in pure Python
    class PyParticle(Particle):
        def __init__(self, m, p, v):
            super(PyParticle, self).__init__(m, p, v)
    	def get_momentum(self):
            return super(PyParticle, self).get_momentume()
    ```

    - cannot access any private C attributes or `cdef` methods
    - CAN override `def` and `cpdef` methods 
    - CAN pass `PyParicles` to list as well

  - **Crossing Cython/Python boundary polymorphically is nice, but it does have overhead**

    - becaues `cdef` method is inaccessible or not overrideable from Python, it does not have to cross the language boundary, so it has less call overhead than `cpdef` equivalent. Relevant concern only for small functions where call overhead is non-negligible 

## Casting and Subclass

- when working with a DT object, Cython cannot access any C data or methods on it! All attribute lookup must be done via API slow! If knew dynamic variable is or may possibly be an instance of a built-in type or an extension type, THEN it is worth **casting to the ST** - allowing Cython to access C attributes and methods - further Cython can also access Python attributes and `cpdef` methods directly without going through API

- casting: either by creating a ST variable of desired type and assigning the dynamic variable to it, OR using Cython's casting operator

- Example: an object `p` might be instance of `Particle` or one of its subclasses, ALL cython knows about `p` is that it is a Python object, can call `get_momentum` which works if `p` has such a method and fail otherwise; because `p` is a **dynamic variable**, Cython will access `get_momentum` looking up in a Python dictionary, if OK, `PyObject_Call` will exec the method - BUT if casting it to a `Particle` **explicitly**, the call is faster:

- ```cython
  cdef Particle static_p = p
  print static_p.get_momentum()
  print static_p.velocity
  
  # assignment raise TypeError if p is not an instance of Particle or its subcls
  # call to method uses direct access to cpdef method
  # allowed access to private velocity attribute, which is not available to p !!
  
  # NOTE Cython uses general Python method lookups on DT objects, failling with an
  # AttributeError if method is declared cdef; to ensure fast access to cpdef methods, or to 
  # allow any access to cdef methods, must provide ST info for th object
  
  # casting way
  print (<Particle>p).get_momentum()
  
  # this removes need to create a temp_var 
  # () due to precedence rules
  # as using a raw cast to Particle, no type checking is performed for speed
  # unsafe if p not an instance of Particle, leading to segmentation fault
  # if such possible:
  print (<Particle?>p).get_momentum() # TypeError, tradeoff is checked cast calls into API and incurs runtime overhead
  ```



## Extension Type Objects and None

- ```cython
  def dispatch(Particle p):
      print p.get_momentum()
      print p.velocity
      
  # if calling dispath and pass NON-Particle object, then TypeError
  dispatch(Particle(1,2,3)) # OK
  dispatch(CParticle(1,2,3)) # OK
  dispatch(PyParticle(1,2,3)) # OK
  dispatch(object()) # TypeError
  
  # calling dispatch with None does not result in TypeError
  dispatch(None) # Segmentation fault!
  ```

  - However, Cython treats `None` specially - though not an instance of Particle, allows it to be passed in as if it were - analogous to NULL pointer in C:

    - allowed wherever a C pointer is expected, but doing anything other than checking whether it is NULL will result in segmentation fault or worse!

  - Why? because dispatch(unsafely) accesses the cpdef function get_momentum and the private attribute both are C interface; Python's none object essentially has NO C interface, so trying to call a method on it or access an **attribute is not valid - to be safe, check if p is None first:**

    - ```cython
      def dispatch(Particle p):
          if p is None:
              raise TypeError("...")
      
      # so common for special syntax
      def dispatch(Particle p not None):
          # ...
      ```

      - safety at up-front type checking
      - **IF ANY POSSIBILITY THAT A FUNC OR METHOD ARG MIGHT BE NONE, need this cost to guard agasint it if accessing any C attributes or methods on the object**
      - SEGMENTATION FAULT AND DATA CORRUPTION
      - If access only Python methods (def methods) and Python attributes (`public, readonly`) on the object, then exception will be raised, API handles for us

  - Cython also provides `nonecheck` compiler directive - off by default for speed `# cython: nonecheck=True`

  - or `cython --directive nonecheck=True source.pyx`

## Extension Type Properties 

- **Python properties handy and powerful, allowing precise control over attribute access and on-the-fly computation!**

- E.g. Particle ext_type has `get_momentum` method, but Pythonically horrendous a getter method like that; should be either exposing `momentum` directly or make a property instead

- ```cython
  # pure python
  class Particle(object):
      #...
      def _get_momentum(self):
          return self.mass * self.velocity
      momentum = property(_get_momentum)
  
  # CANNOT set or delete p.momentum because no setter or deleter in property definition
  
  # Cython diff syntax but same end
  cdef class Particle:
      # ....
      property momentum:
          __get__(self):
              return self.mass * self.velocity
  ```

  - **knowing ST gives faster access, this is a read-only property**

- ```cython
  # setter
  def __set__(self, m):
      self.velocity = m / self.mass
  ```

  - arbitrarily decide that setting momentum will modify velocity and leave mass intact, allowing assignment
  - `p = cython_particle.Particle(1, 2, 3); p.momentum = 4.0; p.velocity == `
  - **`__del__` property controls deletion - decoupled the three !!**

## Special Methods Are Even More SPECIAL

- when providing support for operator overloading with ext_type, need to define a special method - specifc name with leading and trailling double underscores
- Extension types do not support `__del__`, that's the role of `__dealloc__`

### Arithmetic

- `__add__(self, other)` enables ` c + d` to call `C.__add__(c, d)` ; if not implemented, Python interpreter calls `type(d).__radd__(d, c)` to give d's class a chance to add itself to a C instance

- Applied to ALL extension type (beyond Cython): DO NOT support `__radd__`, instead effectively overload `__add__` to do BOTH regular and class-type - meaning for a Cython-defined extension type E, `__add__` will be called when expression `f + e`  is eval and ALSO call when `f + e` eval.

  - `E.__add__` called with f and e as args, IN THAT ORDER! 

  - so `__add__` may be called with an arbitrary type as first arg, NOT AN INSTANCE of E class; becuase of this, it is misleading to name its first argument self!!

  - ```cython
    cdef class E:
        """Extension type supporting addition"""
        cdef int data
        def __init__(self, d):
            self.data = d
    	def __add__(x, y):
            # Regular __add__ behaviour
            if isinstance(x, E):
                if isinstance(y, int):
                    return (<E>x).data + y
    		# __radd__ behaviour
            elif isinstance(y, E):
                if isintance(x, int):
                    return (<E>y).data + x
    		else:
                return NotImplemented
            
    # shell
    e = special_method.E(100)
    # takes first branch of E.__add__
    e + 1 # 101
    # second branch
    1 + e # 101
    # error: E.__add__ returns last branch, built-in type float tries to do an __radd__ with an E instances as left argument. NOT knowing how to add itself to an E object, it again returns last branch, 
    e + 1.0 # TypeError
    
    # One more case: float's __add__ called, realised it did not know how to handle E instances, returned NotImplemented. Python then called E.__add__(1.0, e) (or equi-else), which also returned NotImplemented, causing raise TypeError
    1.0 + e # TypeError
    
    # Note
    # The in-place operations like __iadd__ always take an instance of the class as the first
    # argument, so self is an appropriate name in these cases. The exception to this is
    # __ipow__ , which may be called with a different order of arguments, like __add__ .
    ```

    - **Cython does not auto-type either arg to `__add__`, making the `isinstance` chcek and cast necessary to access each E instances's internal .data attribute.**

  

  ### Rich Comparisons

  - no `__eq__, __lt__, __le__` but a SINGLE, cryptic `__richcmp__(x, y, op)` taking an integer third arg to specify which comparison ops to perform

  - ```cython
    from cpython.object cimport Py_LT, Py_LE, Py__EQ, Py_GE, Py_GT, Py_NE
    
    cdef class R:
        cdef double data
        def __init__(self, d):
            self.data = d
    	def __richcmp__(x, y, int op):
            cdef:
                R r # interesting, declare its own cls ?
                double data
    	# Make r always refer to R instance
        r, y = (x, y) if isinstance(x, R) else (y, x)
        
        data = r.data
        if op == Py_LT:
            return data < y
        elif op == Py_LE:
        	return data <= y
        elif op == Py_EQ:
        	return data == y
        elif op == Py_NE:
        	return data != y
        elif op == Py_GT:
        	return data > y
        elif op == Py_GE:
        	return data >= y
        else:
        	assert False
            
    # SPECIAL CHAIN COMPARE:
    0 <= r <= 100 # True
    ```

  

  ### Iterator

  - `__iter__` makes object iterable

  - `__next__` makes iterator

  - ```cython
    cdef class I:
        cdef:
            list data
            int i
    	def __init__(self):
            self.data = range(100)
            self.i = 0
    	def __iter__(self):
            return self
        def __next__(self):
            if self.i >= len(self.data):
                raise StopIteration()
    		set = self.data[self.i]
            self.i += 1
            return ret
        
    # I defines __iter__, can be used in for loops
    from special_methods import I
    i = I()
    s = 0
    for x in i:
        s += x
    
    # iterator
    it = iter(I())
    it.next() # 0
    ```

SUMMARY

CYTHON MELDS EXT_TYPE DEFINITION

1. allows easy and fast access to an instance's C data and methods
2. memory efficient
3. allows control over attribute visibility
4. can be subclassed from Python
5. workds with existing built-in types and other extension types

USE: TO WRAP C STRUCTS, FUNCTIONS, AND C++ CLASSES TO PROVIDE NICE OBJECT-ORIENTED INTERFACES TO EXTERNAL LIBRARIES.



# Chapter 6: Organising Cython Code

- Cython `import` allows RUNTIME access Python objects DEFINED in external Python modules or Python-accessible objects DEFINED in other extension modules!
- If that was it: would not allow two Cython modules to access each other's cdef or cpdef functoins, ctypedefs, structs, nor allow C access to other extension types!
- SO: 3-FILE types to organise Cython-specfic and C parts of a project
- IMPLEMENTATION FILE - `.pyx`
- DEFINITION FILE - `.pxd`
- INCLUDE FILE - `.pxi`
- `cimport` for COMPILE-TIME access to C constructs, looking for their declarations inside `.pxd` files

## Declaration File

- need for sharing `pyx` C constructs

- Example of Simulator.pyx:

  - `ctypedef`
  - `cdef` class named State to hold simulation state
  - Two `def`, setup and output, to init and to report or visualise results
  - Two `cpdef`, run and step, to drive the simulation and advance one time step

- ```cython
  # simulator.pyx
  ctypedef double real_t
  
  cdef class State:
      cdef:
          unsigned int n_particles
          real_t *x
          real_t *vx
  	def __cinit__(...):
          # ...
  	def __dealloc__(...):
          # ...
  	cpdef real_t momentum(self):
          # ...
          
  def setup(input_fname):
      # ...
  cpdef run(State st):
      # ... calls step func repeatedly
  cpdef int step(State st, real_t timestep):
      # ... advance st one time step...
  def output(State st):
      # ...
  ```

  - all in one file, calling cpdef run will access fast C bypassing Python wrapper
  - AS SIMULATOR EXT_MODULE GAINS MORE FUNCTIONALITY, BECOMES HARDER TO MAINTAIN
  - To make it modular, need to break it up into logical subcomponents !

- First create `simulator.pxd` definition file holding declaration C constructs to share

- ```cython
  # simulator.pxd
  ctypedef double real_t
  
  cdef class State:
      cdef:
          unsigned int n_particles
          real_t *x
          real_t *vx
          
  	cpdef real_t momentum(self)
      
  cpdef run(State st)
  
  cpdef int step(State st, real_t timestep)
  ```

  - because definition files (header files?) are meant for compile-time access, note that we put ONLY C declarations in it. These functions are accessible at runtime, so they are just declared and defined inside implementation file.
  - implementation files need change, since base name the same `simulator.*`,  they are treated as ONE NAMESPACE by Cython - CANNOT repeat any of declaractions in implementation file - compile error

### Declarations and Definitions

- Syntactically, declaration for a function or method includes everything for the fucntion or method's SIGNATURE:
  -  the declaration type (`cdef, cpdef`); 
  - the function or methods' name; 
  - all in the arg list. 
  - NOT terminating colon. 
- For cdef class, the declaration includes 
  - cdef class line (colon included) and 
  - extension types' name, 
  - all attribute declarations and 
  - all method declarations
- Cython definition is all required for that construct's implementation. The definition for a function or method repeats the declaration as part of the definition (i.e., the implementation); the definition for a `cdef` class does not redeclare the attribute declarations

```cython
# simulator.pyx

cdef class State:
    def __cinit__(...):
        # ...
	def __dealloc__(...):
        # ...
	cpdef real_t momentum(self):
        # ...

def setup(input_fname):
    #...
cpdef run(State st):
	# ...calls step function repeatedly...
cpdef int step(State st, real_t timestep):
	# ...advance st one time step...
def output(State st):
    #...
```

- `ctypedef` and State type's attributes have been moved to the definition file, so they are removed from the implementation file. The definitions of all objects, whether C or Python, go inside the implementation file.
- **Cython compiler will auto-use definition files' declarations!**
- What's in DEFINITION FILE: **anything meant to be publicly accessible to other Cython modules at C level**
  - C type declarations - `ctypedef, struct, union, enum`
  - Declarations for external C or C++ libs (`cdef extern` blocks)
  - Declarations for `cdef, cpdef` module-level functions
  - Declarations for `cdef class` extensions types
  - `cdef` attributes of extension types
  - Declarations for `cdef, cpdef` methods
  - Implementation of C-level `inline` functions and methods
- CANNOT contain
  - implementation of Python or non-inline C functions or methods
  - Python class definition
  - Executables Python code outside of `IF, DEF` macros
- Now an external implementation file can access all C-level constructs inside simulator.pyx via `cimport` statement

## cimport Statement

- suppose another improved `.pyx` need `setup, step` functions but a different `run` need to subclass `State` extension type:

- ```cython
  # imp_simulator.pyx
  from simulator cimport State, step, real_t
  from simulator import setup as sim_setup
  
  cdef class NewState(State):
      cdef:
          # ... extra attributes...
  	def __cinit__(self, ...):
          # ...
  	def __dealloc__(self):
          # ...
  
  def setup(fname):
      # ... call sim_setup and tweak things slightly ...
      
  cpdef run(State st):
      # ... improved run taht uses simulator.step ...
  ```

  - first line uses cimport statement to access the State extension type, the step cpdef function, and the real_t ctypedef

    - **this access is at C level and occurs at compile time**

    - only declarations in definition file are `cimport`able 

    - constrast to second line which uses import to access the setup def from extension module; works at Python level and import occurs at runtime

    - ```cython
      cimport simullator # pxd file
      # ...
      cdef simulator.State st = simulator.State(params)
      cdef simulator.real_t dt = 0.01
      simulator.step(st, dt)
      
      # can provide an alias when cimport definition file
      cimport simulator as sim
      
      # ....
      cdef sim.State st = sim.State(params)
      cdef sim.real_t dt = 0.01
      sim.step(st, dt)
      
      # also provide alias to specific cimported declarations with as 
      from simulator cimport State as sim_state, step as sim_step
      ```

  - compile-time error to cimport a Python-level object like setup function

  - reverse, compile-time error to import C-only declaration like real_t

  - Allowed to import or cimport State extension type or the step cpdef function, although cimport is recommended

  - if import rather than cimport extension types or cpdef functions, would have Python-only access

  - This blocks access to any private attributes or cdef methods, and cpdef methods and functions use the slower Python wrapper

  - Def file can contain `cdef extern` blocks, useful to gropu such declarations inside their own `.pxd` files for use elsewhere, doing so provides a useful namespace to help disambiguate where a function is declared

  - e.g. the Mersenne Twister random-number generator RNG header file has a few funcs that can be declared inside a `_mersenne_twister.pdx` definition file:

  - ```cython
    cdef extern from "mt19937ar.h":
        # init mt[N] with a seed
        void init_genrand(unsigned long s)
        
        # gen a random number on [0, 0xffffffff]-interval
        unsigned long genrand_int32()
        
        # gen a random number on [0, 0x7fffffff]-interval
        long genrand_int31()
    # generates a random number on [0,1]-real-interval
    double genrand_real1()
    # generates a random number on [0,1)-real-interval
    double genrand_real2()
    # generates a random number on (0,1)-real-interval
    double genrand_real3()
    # generates a random number on [0,1) with 53-bit resolution
    double genrand_res53()
    ```

    - now any implementation file can simply cimport the necessary function:
    - `from _mersenne_twister cimport init_genrand, genrand_real3`

## Predefined Definition Files

- Cython comes with several predefined definition files for oft-used C, C++ and Python header files
- `Includes` dir underneath main Cython source dir
- `libc` contains `.pxd` files for the `stdlib, stdio, math, string, stdint` header files
- `libcpp` package with `.pxd` files of STL, `string, vector, list, map, pair, set`
- `numpy` file for Numpy/C API

### Using cimport with a module in a package

```cython
from libc cimport math
math.sin(3.14)

# cimport imports module-like math namespace from libc packages allowing dotted access
# C funcs declared in math.h C standard library
```

```cython
# multiple named cimports

from libc.stdlib cimport rand, srand, qsort, malloc, free
cdef int *a = <int*>malloc(10 * sizeof(int))

from libc.string cimport memcpy as c_memcpy

from libcpp.vector cimport vector
cdef vector[int] *vi = new vector[int](10)

# invalid same namespace
```

- Definition files have some similarities to C/C++ header files
  - declare C-level constructs for use by external code
  - allow us to break up what would be one large file into several components
  - declare the public C-level interface for an implementation
- **C/C++ access header files via the #include preprocessor command, which essentially does a dumb source-level inclusion of the named header file**
- **Cython's cimport statement is more intelligent and less error prone: think of it as a compile-time import statement that works with namespaces!**
- Cython's predecessor, Pyrex, did not have cimport statement, and instead had an include for source-level inclusion of an external include file. Cython also supports the include statement and include files, which are used in several Cython projects

## Include Files and Include Statement

- suppose an extension type that we want available on all major platforms, but it must be implemented differently on different platforms. 

- Goal is to abstract away these differences and to provide a consistent interface in a transparent way - include files and the include statement provide one way to accomplish our nice platform-free design goals

- Place 3 different implementations of the extension type in 3 `.pxi` files

  - `linux.pxi, darwin.pxi, windows.pxi`

  - one of the three will be selected and used at compile time

  - inside `interface.pyx` 

    - ```cython
      IF UNAME_SYSNAME == "Linux":
          include "linux.pxi"
      ELIF ...
      ```

## Organising and Compiling Cython Modules Inside Python Packages

- INCREMENTALLY CONVERT PYTHON CODE TO CYTHON

- allowing external API remain intact while overall performance improves

- example of Python pysimulator 

  - ```shell
    __init__.py
    main.py
    core
    	__ini__.py
    	core.py
    	sim_state.py
    plugins
    	__init__.py
    	plugin0.py
    	plugin1.py
    utils
    	__init__.py
    	config.py
    	output.py
    ```

  - focus for this example is not the internal details of the pysimulator modules; it's how Cython modules can access compile-time declarations and work easily within the framework of a Python project

  - suppose profiling reveals `core.py, sim_state.py, plugin0.py` are slow

  - `sim_state.py` module contains State class that will convert into extension type

  - `core.py` contains two funcs, run and step, that will convert to `cpdef`

  - `plugin0.py` contains run also convert into `cpdef` func

- first to convert .py modules into implementation files and extract their **public Cython declarations into definitions files** ; because components are spread out in diff packages and subpackages, must remember to use proper qualified names for importing

- ```cython
  # sim_state.pxd
  ctypedef double real_t
  
  cdef class State:
      cdef:
          unsigned int n_particles
          real_t *x
          real_t *vx
          
  	cpdef real_t momemtum(self)
  
  # all `cpdef` takes `State` instance, need C-level access - so all modules will have to `cimport` the `State` declarationfrom appropriate definition file

  # core.pxd declares the `run, step` cpdef 
  from simulator.core.sim_state cimport State, real_t
  
  cpdef int run(State, list plugins=None)
  cpdef step(State st, real_t dt)
  
  # `cimport` is ABSOLUTE, using fully qualified name to access for clarity
  
  # plugin0.pxd declares its own `run cpdef` takes State instance
  from simulator.core.sim_state cimport State
  
  cpdef run(State st)
  
  # main.py - pure Python, like all inside `utils` subpackage - pulls all together
  from simulator.utils.config import setup_params
  from simulator.utils.output import output_state
  from simulator.core.sim_state import State
  from simulator.core.core import run
  from simulator.plugins import plugin0
  
  def main(fname):
      params = setup_params(fname)
      state = State(params)
      output_state(state)
      run(state, plugins=[plugin0.run])
      output_state(state)
      
  # main.py module remains unchanged after Cyhon conversions, as do any other pure-Python modules in project - Cython allows surgically replacing individual components with extension modules, and rest intact
  ```
  
- to run, first **compile** Cython source into extension modules - pyximport on-the-fly during dev and testing

- `import pyximport; pyximport.install(); from simulator.main import main`

- imported all extension modules, pyximport compiled them auto

- `main("params.txt")` => output indicating process, steps, 

- create distributable compiled package - `distuils` or other build system 

- minimum `setup.py`

- ```python
  from distutils.core import setup
  from Cython.Build import cythonize
  
  setup(name="simulator",
       	packages=["simulator", "simulator.core",
                    "simulator.utils", "simulator.plugins"],
       	ext_modules=cythonize("**/*.pyx"),
       )
  ```

- cythonize glob pattern to recursively search all dir for `.pyx` implementation files and compile them as needed - this way is flexible and powerful - auto-detect when a `.pyx` file has changed and recompile as needed - also detect interdependencies between implementation and definition files and recompile all dependent implementation files

- In sum, 3-file types, in conjunction with `cimport, include` allows organising code into separate moduels and packages, without sacrificing performance - allowing Cython to expand beyond speeding up isolated extension modules, to scale full-fledged projects



# Chapter 7: Wrapping C

- both way - makes C-level Cython constructs for external C code, useful when embedding Python in C app
- Cython allows full control over all aspects during interfacing
- Final outcome: C-speed, minimal wrapper overhead, Pythonic interface

## Declaring External C code in Cython

- First must declare C interface components in Cython `extern` block

- ```cython
  cdef extern from "header_name":
      indented declarations from header file
  ```

  - cython compiler generates an `#include "header_name"` line inside the generated source file
  - the types, funcs, and other declarations made in block body are accessible from Cython
  - Cython will check at compile time that C declarations are used in `type-correct manner`, will prodce error if not

- Declarations in extern block have C-like syntax for variables/funcs, Cython-specific syntax for declaring `structs, unions`

> **Bare extern Declarations** - Cython supports extern `cdef extern external_declaration`, used in this manner, Cython will place it (func signature, variable, struct, union or other such C declaration) in the generated source code with extern modifier - must match C declaration
>
> This style of external declaration is not recommended, as it has the same drawbacks as using extern in C directly, `extern` block is preferred !!

- **if necessary to have `#include` preprocessor directive for a specific header file, but no declarations are required, the block can be empty**

- ```cython
  cdef extern from "header.h":
      pass
  ```

- conversely, if name of header file not necessary (perhaps it's already included by another header file that has its own extern block), but like to interface with external code, can suppress #include statement generation with from *:

- ```cython
  cdef extern from *:
      declarations...
  ```



## Cython Does NOT Automate Wrapping

- purpose of extern block is simple, but c**an be misleading - for ensuring calling and using the declared C functions, variables, structs in type-correct manner**
- NOT auto-gen wrappers for the declared objects! - only C code generated for the ENTIRE extern block is a single #include "header.h" line - STILL HAVE TO write `def, cpdef` (possibly `cdef`) calling C functions declared in extern block
- Pros of Cython C-wrapper
  - highly optimised generated wrapper code
  - often goal is to customize, improve, simplify or otherwise Pythonize interface
  - high-level, Pythonic and not limited to domain-specific interfacing commands, making complicated wrapping tasks easier

## Declaring External C Func and typedefs

- most common declarations are C func/`typedefs` - almost directly from their C equivalents, only mod:

  - change `typedef` to `ctypedef`
  - rid of keywords such as `restrict, volatile`
  - ensure return type and name are declared on a single line
  - remove line-terminating semicolons

- Possible to break up long func declaration over several lines after opening parenthesis as in Python

- e.g. C declar and macros

- ```c
  #define M_PI 3.1415926
  #define MAX(a, b) ((a) >= (b) ? (a) : (b))
  
  double hypot(double, double);
  
  typedef int integral;
  typedef double real;
  
  void func(integrall, integral, real);
  
  real *func_arrays(integral[], integral[][10], real **);
  ```

- ```cython
  # Cython declarations for them, bar the macros, nearly copy-paste
  
  cdef extern from "header.h":
      
      double M_PI
      float MAX(float a, float b)
      
      # arg naming is helpful for calling and doc DO IT!
      double hypot(double x, double y)
      
      ctypedef int integral
      ctypedef double real
      
      void func(integral a, integral b, real c)
      
      real *func_arrays(integral[] i, integral[][10] j, real **k)
  ```

- NOTE for `M_PI` macro, declared as if it were a global of double - and `MAX` func-lke macro, declared in Cython as if a regular C func named `MAX` taking two float return float

- a more complex header.h

  - ```cython
    # C: func named signal taking func-pointer returns func-pointer
    
    cdef extern from "header.h":
    	void (*signal(void(*)(int)))(int)
    
    # Cython uses extern blocks ONLY to check type correctness, can add helper `ctypedef` for legibility
    
    cdef extern from "header.h":
        ctypedef void (*void_int_fptr)(int)
        void_int_fptr signal(void_int_fptr)
        
    # easier to read, as Cython does NOT declare void_int_ptr typedef in generated code, can use it to help make C simpler - void_int_fptr ctypedefs ONLY a Cython convenience, no corresponding typedef in header file!
    ```



## Declaring and Wrapping C structs, unions, enums

- same syntax as before, but omit `cdef` as it's implied

  - ```cython
    cdef extern from "header_name":
        
        struct struct_name:
            struct_members
    	union ...
        enum ...
    ```

  - ```c
    // above translate to C
    
    struct struct_name {
        struct_members
    };
    
    // for typedef verions in C
    
    typedef struct struct_name {
        struct_members
    } struct_alias;
    
    // same for union and enum
    ```

  - ```cython
    cdef extern from "header_name":
        
        ctypedef struct struct_alias:
            struct_members
    ```

  - Cython uses just the alias type names and NOT generate struct, union, enum as part of declaration as is proper

  - to STATICALLY declare a struct in Cython, use `cdef` with struct name or `typedef` alias name; 

  - ONLY necessary to declare fields ACUTALLY used in preceding struct, etc. if no fields are used but it is necessary to use struct as opaque type, then body should be `pass`



## Wrapping C Functions

- **after declaring external func, MUST wrap in `def` , `cpdef` or `cdef` class to access from Python**

- e.g. wrap a RNG, two exposure, 1) `init_genrand` and `genrand_real1`

- ```cython
  # declare them in Cython
  
  cdef extern from "mt19937ar.h":
      void init_genrand(unsigned long s)
      double genrand_real1()
      
  # MUST provide def / cpdef so that these callable from Python
  
  def init_state(unsigned long s):
      init_genrand(s)
      
  def rand():
      return genrand_real1()
  
  # to compile all, write setup.py (make sure to include header in sources)
  
  from distutils.core import setup, Extension
  from Cython.Build import cythonize
  
  ext = Extension("mt_random",
                 	sources=["mt_random.pyx", "mt19937ar.c"])
  
  setup(name="mersenne_random",
        ext_modules = cythonize([ext]))
  
  ```

- `python setup.py build_ext --inplace`

- output: `mt_random.so` or `mt_random.pyd` depending on OS

- ```shell
  # use it in IPython
  
  mt_random.init_state(42)
  
  mt_random.rand() # 0.37....
  ```

- CONS: uses a STATIC GLOBAL array to store RNG state, allowing only ONE RNG a time

- next version supports concurrent generators

## Wrapping C structs with Extension Types

- ```c
  // mt19937ar-struct.h
  // improved API first forward-decalres a struct typedef in header file
  
  typedef struct _mt_state mt_state;
  
  // declares creation and destruction
  
  mt_state *make_mt(unsigned long s);
  void free_mt(mt_state *state);
  
  // RNG func take a pointer to a heap-alloc mt_state struct as arg, wrap just one of them
  
  double genrand_real1(mt_state *state);
  ```

- ```cython
  # cython declaration for new API
  
  cdef extern from "mt19937ar-struct.h":
      ctypedef struct mt_state
      mt_state *make_mt(unsigned long s)
      void free_mt(mt_state *state)
      double genrand_real1(mt_state *state)
  ```

- because mt_state struct is opaque and Cython does not need to access any of its internal fields, the preceding ctypedef is sufficient ! essentially a named placeholder

- cython exposes none of these C extern declarations to Python - it's nice to wrap this imp-version in an extension type named MT, the only attribute will hold is a private pointer to an mt_state struct:

  - ```cython
    cdef class MT:
        cdef mt_state *_thisptr
        # creating mt_state heap-alloc struct MUST in C-level before MT object init
        # proper place to do it is in __cinit__
        def __cinit__(self, unsigned long s):
            self._thisptr = make_mt(s)
            if self._thisptr == NULL:
                msg = "Insufficient memory."
                raise MemoryError(msg)
    	# deconstructor
        def __dealloc__(self):
            if self._thisptr != NULL:
                free_mt(self._thisptr)
                
    	# these methods allow proper create, init, free MT object
        # to RNG, simply define def / cpdef calling the C fucntions
        cpdef double rand(self):
            return genrand_real1(self._thisptr)
    ```

  - declaring and interfacing the remaining funcs is left as exercise

  - ```python
    # setup_mt_type.py
    
    from distuils.core import setup, Extension
    from Cython.Build import cythonize
    
    ext_type = Extension("mt_random_type",
                         sources=["mt_random_type.pyx",
                                  "mt19937ar-struct.c"])
    
    setup(name="mersenne_random",
          ext_modules = cythonize([ext_type]))
    ```

  - ```shell
    python setup_mt_type.py build_ext --inplace
    
    from mt_random_type import MT
    
    mt1, mt2 = MT(0), MT(0)
    
    mt1.rand() == mt2.rand() # True
    ```

  

  > For wrapping C structs in Cython, the pattern used in this example is common and
  > recommended. The internal struct pointer is kept private and used only internally. The
  > struct is allocated and initialized in __cinit__ and automatically deallocated in
  > __dealloc__ . Declaring methods cpdef when possible allows them to be called by ex‐
  > ternal Python code, and efficiently from other Cython code. It also allows these methods
  > to be overridden in Python subclasses.



## More Control - constant, modifier, output

- `const` not useful in `cdef` but IS in specific instances within `cdef extern` to ensure Cython generates const-correct code

- NOT necessary for declaring func arg, may be required when declaring a `typedef` using `const` or return one

- ```c
  typedef const int * const_int_ptr;
  const double *returns_ptr_to_const(const_int_ptr);
  ```

  ```cython
  # forward to .pyd
  cdef extern from "header.h":
      ctypedef const int * const_int_ptr
      const double *returns_ptr_to_const(const_int_ptr)
  ```

- modifiers `volatile, restrict` should be removed in Cython 

- naming C level objects differently

  - ```cython
    # wrap C print - cannot use print as reserved - aliasing
    
    cdef extern from "printer.h":
        void _print "print"(fmt_str, arg)
        
    # named _print in Cython, but print in generated C, also works for typedef/struct/etc
    
    cdef extern from "pathological.h":
        # typedef void * class
        ctypedef void * kclass "class"
        
        # int finally(void) function
        int _finally "finally"()
        
        # struct del { int a, b; };
        struct _del "del":
            int a, b
            
    	# enum yield { ALOT; SOME; ALITTLE; };
        enum _yield "yield":
            ALOT
            SOME
            ALITTLE
    ```

  - **string in quotes is the name of object in generated C code, Cython does no checking on content of this string**



## Error Exception - Callback

- `except` clause can be used in conjunction with `cdef` callback

- Cython supports C func-pointer - allow wrapping C func taking func-pointer callbacks

- Callback can be pure-C, or arbitrary Python

- **Empower passing in a Python func created at runtime to control behaviour of the underlying C function**

- e.g., wrap `qsort` from C STL

  - ```cython
    # declared
    cdef extern from "stdlib.h":
        void qsort(void *array, size_t count, size_t size,
                   int (*compare)(const void *, const void *))
    ```

  - features

    1. alloc C array of integers of proper size 
    2. convert list of Python integers into C int array
    3. call qsort with proper compare func
    4. convert sorted values back to Python and return

  - ```cython
    # func declaration
    
    cdef extern from "stdlib.h":
        void *malloc(size_t size)
        void free(void *ptr)
        
    def pyqsort(list x):
        cdef:
            int *array
            int i, N
            
    	# Alloc C array
        N = len(x)
        array = <int*>malloc(sizeof(int) * N)
        if array == NULL:
            raise MemoryEffor("Unable to allocate array.")
            
    	# Fill C array with Python int
        for i in range(N):
            array[i] = x[i]
            
    	# qsort array....
        
        # Convert back to Python and free C array
        for i in range(N):
            x[i] = array[i]
    	free(array)
        
    # set up compare callback
    # standard sort
    
    cdef int int_compare(const void *a, const void *b):
        # convert void pointer args into C int
        cdef int ia, ib
        # dereference pointer via indexing 0
        ia = (<int*>a)[0]
        ib = (<int*>b)[0]
        # return signed value
        return ia - ib
    
    # put in qsort array...
    
    qsort(<void*>array, <size_t>N, sizeof(int), int_compare)
    
    # works but static, expand by allowing reverse-sorting array by negating return value of int_compare:
    
    cdef int reverse_int_compare(const void *a, const void *b):
        return -int_compare(a, b)
    
    # add a ctypedef to make working with callback easier
    
    ctypedef int (*qsort_cmp)(const void *, const void *)
    
    def pyqsort(list x, reverse=False):
        # ...
        cdef qsort_cmp cmp_callback
        
        # Select appropriate callback
        if reverse:
            cmp_callback = reverse_int_compare
    	else:
            cmp_callback = int_compare
            
    	# qsort the array ....
        qsort(<void*>array, <size_t>N, sizeof(int), cmp_callback)
        
        # ...
    ```

  - ```shell
    # use
    import pyximport; pyximport.install()
    
    from pyqsort improt pyqsort
    
    from random import suffle
    inlist = range(10); shuffle(inlist)
    
    pyqsort(inlist, reverse=True)
    ```

  - ```cython
    # more control, allow passing own Python comparison func
    # C callback has to call Python callback, converting arg between C types and Python types
    # use a module-global Python object, py_cmp, to store Python func, allow for setting Python callback at runtime, and C callback wrapper can access it when needed
    
    cdef object py_cmp = None
    
    # qsort expects C func, have to create a callback wrapper cdef matching compare func pointer signature and calling py_cmp 
    
    cdef int py_cmp_wrapper(const void *a, const void *b):
        cdef int ia, ib
        ia = (<int*>a)[0]
        ib = (<int*>b)[0]
        return py_cmp(ia, ib)
    
    # inside py_cmp_wrapper, must cast void pointer arg to int pointers, dereference them to extract value integers, pass to py_cmp
    # cython will auto-convert C integers to Python int, hence return value will be converted to C int
    
    # reverse version
    cdef int reverse_py_cmp_wrapper(const void *a, const void *b):
        return -py_cmp_wrapper(a, b)
    
    # final logic over 4 callbacks
    
    def pyqsort(list x, cmp=None, reverse=False):
        global py_cmp
        # ...
        
        # set up cmp callback
        if cmp and reverse:
            py_cmp = cmp
            cmp_callback = reverse_py_cmp_wrapper
    	elif cmp and not reverse:
            py_cmp = cmp
            cmp_callback = py_cmp_wrapper
    	elif reverse:
            cmp_callback = reverse_int_compare
    	else:
            cmp_callback = int_compare
            
    	# qsort the array ...
        qsort(<void*>array, <size_t>N, sizeof(int), cmp_callback)
    ```

  - ```shell
    # user defined sort
    def cmp(a, b):
    	return abs(a) - abs(b)
    	
    pyqsort(a, cmp=cmp)
    ```



### Callbacks and Exception Propagation

- ```cython
  # update qsort declaration to allow exception callback
  
  cdef extern from "stdlib.h":
      void qsort(void *array, size_t count, size_t size,
                 int (*compare)(const void *, const void *) except *)
      
  # also adding except * clause to ctypedef and all callbacks
  
  ctypedef int (*qsort_cmp)(const void *, const void *) except *
  
  cdef int int_compare(const void *a, const void *b) except *:
  	# ...
  cdef int reverse_int_compare(const void *a, const void *b) except *:
  	# ...
  cdef int py_cmp_wrapper(const void *a, const void *b) except *:
  	# ...
  cdef int reverse_py_cmp_wrapper(const void *a, const void *b) except *:
  	# ...
  ```



# Wrapping C++



# Chapter 9: Profiling Tools

