# CuStabilizer


In this repository, I implement a cuda package for stabilizer simulation and compare the performance with qiskit, stim, and cuquantum


# Usecases of tableau simulation 

## My tableau simulator

To initialize the tableau stabilizer, import the class from tableau.py file:

```python
from tableau import tableau
Tb=tableau(3)
Tb.init_tableau()
stablist=Tb.get_stabilizers()
print(stablist)
```

User can add some Clifford gates by the provided interface:


```python
from tableau import tableau
Tb=tableau(3)
Tb.H(0)
Tb.CNOT(0,1)
Tb.CZ(0,2)
Tb.X(0,2)
Tb.Y(0,2)
stablist=Tb.get_stabilizers()
print(stablist)
```


User can also read the instructions from other file:


```python
from tableau import tableau
Tb=tableau(3)
Tb.read_instructions_from_file("testcases/example1.stab")
Tb.calculate()
stablist=Tb.get_stabilizers()
print(stablist)
```


## Tableau simulator in c++

The tableau simulation is also implented in C++. To compile and run, excute the following command:


```console
g++ -o tableau tableau.cpp
./tableau
```



## Stim


To compare with the simulation speed 


```python
from stimsimulator import stimsimulator
Stimtb=stimsimulator(3)
Tb.read_instructions_from_file("testcases/example1.stab")
Tb.calculate()
stablist=Tb.get_stabilizers()
print(stablist)
```



