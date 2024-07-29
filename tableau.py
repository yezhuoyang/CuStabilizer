import numpy as np




class tableau:
    
    
    def __init__(self,num_qubit) -> None:
        self._num_qubit = num_qubit
        #Store instructions in tuples
        self._instructions=[]
        self._measured_result=[]
        
    #Read instructions from a file
    def read_instructions_from_file(self, filepath):
        # Open the file and read lines
        with open(filepath, 'r') as file:
            for line in file:
                # Strip leading/trailing whitespace and split by space
                parts = line.strip().split()
                
                if len(parts)==2:                
                    gate_type = parts[0]
                    qubit_index = int(parts[1])
                    if gate_type == 'h':
                        self._instructions.append(('h', qubit_index))
                    elif gate_type =='p':
                        self._instructions.append(('p', qubit_index))                        
                    elif gate_type =='x':
                        self._instructions.append(('x', qubit_index))                        
                    elif gate_type =='y':
                        self._instructions.append(('y', qubit_index)) 
                    elif gate_type =='z':
                        self._instructions.append(('z', qubit_index))     
                    elif gate_type =='m':
                        self._instructions.append(('m', qubit_index))
                elif len(parts)==3:
                    gate_type = parts[0]
                    control_index = int(parts[1])     
                    target_index = int(parts[2])     
                    if gate_type == 'cz':
                        self._instructions.append(('cz', control_index, target_index))     
                    elif gate_type == 'cnot':
                        self._instructions.append(('cnot', control_index, target_index))                                   
    
    
    def calculate(self):
        self.init_tableau()
        for instruction in self._instructions:
            self.execute_step(instruction)
        
    
    #Execute one instruction. Instruction is in tuple form: (gate, target)
    def execute_step(self, instruction):
        if instruction[0] == 'h':
            self.H(instruction[1])
        elif instruction[0] == 'p':
            self.P(instruction[1])
        elif instruction[0] == 'x':
            self.X(instruction[1])
        elif instruction[0] == 'y':
            self.Y(instruction[1])
        elif instruction[0] == 'z':
            self.Z(instruction[1])
        elif instruction[0] == 'cz':
            self.CZ(instruction[1], instruction[2])
        elif instruction[0] == 'cnot':
            self.CNOT(instruction[1], instruction[2])
        elif instruction[0] == 'm':
            self.measure(instruction[1])
    
        
        
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
        self.H(target)
        self.Z(target)
        self.H(target)
    
    #Pauli Y gate          
    def Y(self, target):
        self.X(target)
        self.Z(target)
    
    #Pauli Z gate     
    def Z(self, target):
        self.P(target)
        self.P(target)
 
    #Pauli CZ gate      
    def CZ(self, control, target):
        self.H(target)
        self.CNOT(control,target)
        self.H(target)
    
    
    
    #Calculate the inner product of two tableau
    def inner_product(self,other):
        pass
    
    
    #Do gaussian elimination to reduce the tableau to the normal form
    def Gaussian_elimination(self):
        pass
    
    
    #Measure qubit target in the computational basis
    def measure(self, target):
        pass
    