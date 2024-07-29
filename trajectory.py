




#Trajectory method to simulate noise channel
class trajectory:
        
    def __init__(self,num_qubit) -> None:
        self._num_qubit = num_qubit
        #Store instructions in tuples
        self._instructions=[]
        self._measured_result=[]