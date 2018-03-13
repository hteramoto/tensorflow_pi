# Super Simple 𝜋 Approximation Using TensorFlow

Practice exercise for understanding TensorFlow low level APIs.  Runs Gregory Leibniz series and BPP digit extraction algorithm to approximate 𝜋 at 64-bit floating point accuracy.

## Prerequisite

* TensorFlow


## Command line usage

```
usage: pi.py [-h] [-d] [-l LOGPATH] 
 optional arguments: 
   -h, --help            show this help message and exit
   -d, --debug           Enables TensorFlow debugger.   
   -l LOGPATH, --logPath LOGPATH
                         Saves TensorBoard logs at path.
```

## Sample Output

```
Running pi approximation algorithm Gregory Leibniz series:
iteration       1000:  Pi = 3.140592653839794
iteration       2000:  Pi = 3.1410926536210413
iteration       3000:  Pi = 3.1412593202657186
iteration       4000:  Pi = 3.1413426535937043
iteration       5000:  Pi = 3.141392653591791
iteration       6000:  Pi = 3.141425986924278
iteration       7000:  Pi = 3.1414497964476573
iteration       8000:  Pi = 3.141467653590268
iteration       9000:  Pi = 3.1414815424790095
iteration      10000:  Pi = 3.1414926535900345
iteration      11000:  Pi = 3.1415017444990587
iteration      12000:  Pi = 3.1415093202565925
iteration      13000:  Pi = 3.1415157305129755
iteration      14000:  Pi = 3.141521225018448
iteration      15000:  Pi = 3.1415259869231935

Running pi approximation algorithm BBP digit extraction:
iteration          1:  Pi = 3.1333333333333333
iteration          2:  Pi = 3.1414224664224664
iteration          3:  Pi = 3.1415873903465816
iteration          4:  Pi = 3.1415924575674357
iteration          5:  Pi = 3.1415926454603365
iteration          6:  Pi = 3.141592653228088
iteration          7:  Pi = 3.141592653572881
iteration          8:  Pi = 3.141592653588973
iteration          9:  Pi = 3.1415926535897523
iteration         10:  Pi = 3.1415926535897913
iteration         11:  Pi = 3.141592653589793
iteration         12:  Pi = 3.141592653589793
iteration         13:  Pi = 3.141592653589793
iteration         14:  Pi = 3.141592653589793
iteration         15:  Pi = 3.141592653589793
```
