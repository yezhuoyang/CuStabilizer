#ifndef TABLEAU_H
#define TABLEAU_H

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

#define _CONTROLNOT 0
#define _HADAMARD 1
#define _PHASE 2
#define _PAULIX 3
#define _PAULIY 4
#define _PAULIZ 5
#define _CONTROLZ 6

//Macro function used to set the bitvalue of tableau for cuda calculation
#define getTableauElement(tableauMatrix,rowsize, row, col) (tableauMatrix[row*rowsize+col/8]&(0b10000000>>(col%8)))>>(7-(col%8))


#define setTableauValue(tableauMatrix,rowsize, row, col,value)       \
    do {                             \
        tableauMatrix[row*rowsize+col/8]=(value==1?(tableauMatrix[row*rowsize+col/8]|(0b00000000^(1<<(7-(col%8))))):(tableauMatrix[row*rowsize+col/8]&(0b11111111^(1<<(7-(col%8)))))); \
    } while (0)



struct Instruction {
    int type;
    int target;
    int control;

    Instruction(const int& tp,const int& contr,const int& targ):type(tp),target(targ),control(contr){}

};


// Overload the << operator outside the struct without friend
std::ostream& operator<<(std::ostream& os, const  Instruction& inst);


class Tableau {

private:
         size_t num_qubits;
         int threadNum;
         int rowsize;
         int charsize;
         int threadsPerBlock;
         int blocksPerGrid;
         bool cudaMode;
         vector<vector<bool>> tableauMatrix;
         unsigned char* char_tableauMatrix;
         unsigned char* cutableauMatrix;         
         vector<string>* stablizerList;
         vector<Instruction>* instructionSet; 

public:

         Tableau(const size_t& num_qubits,const bool& _cudaMode);
         ~Tableau();
         void calculate();
         void read_instructions_from_file(const string &  filepath);
         void print_instructions();
         void execute_step(const Instruction& inst);
         void print_tableau();
         void print_stabilizers();
         void calculate_stabilizers();
         void init_tableau();
         void X(const size_t& target);
         void Y(const size_t& target);
         void Z(const size_t& target);
         void P(const size_t& target);
         void H(const size_t& target);
         void CNOT(const size_t& control,const size_t& target);
         void CZ(const size_t& control,const size_t& target);
         void checkCudaError(const char* msg);
         void show_tableau_bit();
         void show_tableau_char();
};


#endif 