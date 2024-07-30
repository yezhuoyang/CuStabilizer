#ifndef TABLEAU_H
#define TABLEAU_H

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cstdlib>


using namespace std;

#define _CONTROLNOT 0
#define _HADAMARD 1
#define _PHASE 2
#define _PAULIX 3
#define _PAULIY 4
#define _PAULIZ 5
#define _CONTROLZ 6



struct Instruction {
    int type;
    int target;
    int control;

    Instruction(const int& tp,const int& contr,const int& targ):type(tp),target(targ),control(contr){

    }

};


// Overload the << operator outside the struct without friend
std::ostream& operator<<(std::ostream& os, const  Instruction& inst);



class Tableau {

private:
         size_t num_qubits;
         vector<vector<bool>> tableauMatrix;
         vector<string>* stablizerList;
         vector<Instruction>* instructionSet; 

public:

         Tableau(const size_t& num_qubits);
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
};


#endif 