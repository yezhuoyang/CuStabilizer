import numpy as np





class tableau:
    
    
    def __init__(num__qubit,self) -> None:
        self._num_qubit = num__qubit
        #Store instructions in tuples
        self._instructions=[]
        
        
    #Read instructions from a file
    def read_instructions_from_file(self, filepath):
        pass    
        
        
    def init_tableau(self):
        self._tableau = np.zeros((2*self._num_qubit, 2*self._num_qubit+1), dtype=int)
        self._tableau[0:self._num_qubit, 0] = 1        
        diagonal_indices=np.arange(2*self._num_qubit)
        self._tableau[diagonal_indices, diagonal_indices] = 1
        
           
    '''
    Return the list of stabilizers in the string format
    For example:
    ["XXI","XYI",...]
    '''    
    def get_stabilizers():   
        pass
    
    
    def get_destabilizer():
        pass
    
    
    def CNOT(self, control, target):
        pass
    
    def Hadamard(self, target):
        pass
    
    def Phase(self, target):
        pass
    
    def measure(self, target):
        pass
    