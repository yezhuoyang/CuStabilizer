import stim




class stimsimulator:
    

    def __init__(self,num_qubit) -> None:
        self._num_qubit = num_qubit
        #Store instructions in tuples
        self._instructions=[]
        self._measured_result=[]
        self._simulator = stim.TableauSimulator()
    
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
            
            
    def CNOT(self, control, target):
        self._simulator.cnot(control, target)
    
    
    #Hadamard gate
    def H(self, target):
        self._simulator.h(target)
    
    
    #Phase gate
    def P(self, target):
        self._simulator.s(target)


    #Pauli X gate           
    def X(self, target):
         self._simulator.x(target)
    
    #Pauli Y gate          
    def Y(self, target):
         self._simulator.y(target)

    #Pauli Z gate     
    def Z(self, target):
         self._simulator.z(target)
 
 
    #Pauli CZ gate      
    def CZ(self, control, target):
         self._simulator.cz(control,target)
    
    
    def measure(self, target):
        if(self._simulator.measure(target)):
            self._measured_result.append(1)
        else:
            self._measured_result.append(0)
            
            
        