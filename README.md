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


## Tableau simulator in c++ and using cuda

The tableau simulation is also implented in C++. To compile and run, excute the following command:


```console
cd custabilizer
./compile.sh
./test
```

## Tableau simulator using cuda


First, modify the code in test.cpp to make sure you start the cuda mode of simulation

```c++
int main() {
    std::cout << "Hello, World!" << std::endl;
    Tableau* tb=new Tableau(5,true);
    tb->init_tableau();
    tb->print_tableau();
    tb->calculate_stabilizers();
    tb->print_stabilizers();
    tb->read_instructions_from_file("../testcases/example1.stab");
    tb->print_instructions();
    tb->calculate();
    //tb->calculate_stabilizers();
    //tb->print_stabilizers();
    
    tb->show_tableau_bit();
    tb->show_tableau_char();
    return 0;
}
```

Then, compile the code by running the following script:

```console
cd custabilizer
./compileCu.sh
./test
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


# Low rank stabilizer decomposistion



# Noisy circuit simulation


## Trajectory method