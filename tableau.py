import numpy as np




class tableau:
    
    
    def __init__(self,num_qubit) -> None:
        self._num_qubit = num_qubit
        #Store instructions in tuples
        self._instructions=[]
        
        
    #Read instructions from a file
    def read_instructions_from_file(self, filepath):
        pass    
    
    
    
    def calculate(self):
        pass
    
    
    
    #Execute one instruction. Instruction is in tuple form: (gate, target)
    def execute_step(self, instruction):
        pass
    
        
        
    def init_tableau(self):
        self._tableau = np.zeros((2*self._num_qubit, 2*self._num_qubit+1), dtype=int)   
        diagonal_indices=np.arange(2*self._num_qubit)
        self._tableau[diagonal_indices, diagonal_indices] = 1
        
           
    '''
    Return the list of stabilizers in the string format
    For example:
    ["XXI","XYI",...]
    '''    
    def get_stabilizers(self):   
        stabilizer_list=[]
        for i in range(0,self._num_qubit):
            
            stabilizer=""
            for j in range(0,self._num_qubit):
                if self._tableau[i,j]==1 and self._tableau[i,j+self._num_qubit]==1:
                    stabilizer+="Y"
                elif self._tableau[i,j]==1 and self._tableau[i,j+self._num_qubit]==0:
                    stabilizer+="Z"
                elif self._tableau[i,j]==0 and self._tableau[i,j+self._num_qubit]==1:
                    stabilizer+="X"                   
                else:
                    stabilizer+="I"
            if self._tableau[i,2*self._num_qubit]==1:
                stabilizer="-"+stabilizer
            stabilizer_list.append(stabilizer)
        return stabilizer_list
    
    
    def get_destabilizer(self):
        destabilizer_list=[]
        for i in range(self._num_qubit,self._num_qubit*2):
            
            destabilizer=""
            for j in range(0,self._num_qubit):
                if self._tableau[i,j]==1 and self._tableau[i,j+self._num_qubit]==1:
                    destabilizer+="Y"
                elif self._tableau[i,j]==1 and self._tableau[i,j+self._num_qubit]==0:
                    destabilizer+="Z"
                elif self._tableau[i,j]==0 and self._tableau[i,j+self._num_qubit]==1:
                    destabilizer+="X"                   
                else:
                    destabilizer+="I"
            if self._tableau[i,2*self._num_qubit]==1:
                destabilizer="-"+destabilizer
            destabilizer_list.append(destabilizer)
        return destabilizer_list
    
    
    def commute(self,row1,row2):
        symplectic_inner_prod=0
        for i in range(0,self._num_qubit):
            symplectic_inner_prod+=self._tableau[row1,i]*self._tableau[row2+self._num_qubit,i]
        for i in range(0,self._num_qubit):
            symplectic_inner_prod+=self._tableau[row1+self._num_qubit,i]*self._tableau[row2,i]      
        return symplectic_inner_prod%2==0 
    
    
    def CNOT(self, control, target):
         for k in range(0,2*self._num_qubit):
            #Swap xki and zki
            multi=self._tableau[k,control+self._num_qubit]*self._tableau[k,target]
            xorsum=(self._tableau[k,target+self._num_qubit]+self._tableau[k,control]+1)%2
            #Set new rk
            self._tableau[k,2*self._num_qubit]=(self._tableau[k,2*self._num_qubit]+multi+xorsum)%2
            #Update zki and xki
            self._tableau[k,target+self._num_qubit]=(self._tableau[k,target+self._num_qubit]+self._tableau[k,control+self._num_qubit])%2
            self._tableau[k,control]=(self._tableau[k,control]+self._tableau[k,target])%2        
    
    
    #Hadamard gate
    def H(self, target):
        for k in range(0,2*self._num_qubit):
            #Set new rk
            self._tableau[k,2*self._num_qubit]=(self._tableau[k,2*self._num_qubit]+(self._tableau[k,target]*self._tableau[k,target+self._num_qubit]))%2
            #Swap xki and zki
            tmp=self._tableau[k,target]
            self._tableau[k,target]=self._tableau[k,target+self._num_qubit]
            self._tableau[k,target+self._num_qubit]=tmp   
    
    
    #Phase gate
    def P(self, target):
        for k in range(0,2*self._num_qubit):
            #Set new rk
            self._tableau[k,2*self._num_qubit]=(self._tableau[k,2*self._num_qubit]+(self._tableau[k,target]*self._tableau[k,target+self._num_qubit]))%2
            #Update zki
            self._tableau[k,target]=(self._tableau[k,target]+self._tableau[k,target+self._num_qubit])%2

            
    #Pauli X gate           
    def X(self, target):
        pass
    
    #Pauli Y gate          
    def Y(self, target):
        pass
    
    #Pauli Z gate     
    def Z(self, target):
        pass 
 
    #Pauli CZ gate      
    def CZ(self, control, target):
        pass
    
    #Do gaussian elimination to reduce the tableau to the normal form
    def Gaussian_elimination(self):
        pass
    
    
    
    def measure(self, target):
        pass
    