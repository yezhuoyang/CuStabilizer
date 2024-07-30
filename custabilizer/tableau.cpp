
#include <iostream>
#include <vector>
using namespace std;

#define CNOT 0
#define HADAMARD 1
#define PHASE 2
#define PAULIX 3
#define PAULIY 4
#define PAULIZ 5
#define CZ 6



struct Instruction {
    int type;
    int target;
    int control;
};


class Tableau {

private:
         size_t num_qubits;
         vector<vector<bool>> tableauMatrix;
         vector<string>* stablizerList;
         vector<Instruction>* instructionSet; 

public:

         Tableau(size_t num_qubits) : num_qubits(num_qubits){
             tableauMatrix.resize(num_qubits*2);
             for (size_t i = 0; i < num_qubits*2; i++) {
                 tableauMatrix[i].resize(num_qubits*2+1);
             }
             instructionSet=new vector<Instruction>();
         }

         ~Tableau(){
            delete instructionSet;
            if(stablizerList!=nullptr){
                delete stablizerList;
            }
         }

         void calculate(){
            
         }


         void read_instructions_from_file(){

         }

         void execute_step(Instruction inst){

         }



         void print_tableau() {
             for (size_t i = 0; i < num_qubits*2; i++) {
                 for (size_t j = 0; j < num_qubits*2+1; j++) {
                     std::cout << tableauMatrix[i][j] << " ";
                 }
                 std::cout << std::endl;
             }
         }


        void print_stabilizers(){
            if(stablizerList==nullptr){
                stablizerList=new vector<string>;
            }
            for (auto it = stablizerList->begin(); it != stablizerList->end(); ++it) {
                std::cout << *it << std::endl;
            }
        }


         void calculate_stabilizers(){
            string tmpstr;
            if(stablizerList!=nullptr){
                  delete stablizerList;
            }
            stablizerList=new vector<string>;
            for (size_t i = 0; i < num_qubits; i++) {
                 tmpstr="";
                 for (size_t j = 0; j < num_qubits; j++) {
                     if(tableauMatrix[i][j]&&tableauMatrix[i][j+num_qubits]){
                            tmpstr+="Y";
                     }
                     else if(tableauMatrix[i][j]&&(!tableauMatrix[i][j+num_qubits])){
                            tmpstr+="Z";
                     }
                     else if((!tableauMatrix[i][j])&&tableauMatrix[i][j+num_qubits]){
                            tmpstr+="X";
                     }
                     else{
                            tmpstr+="I";
                     }
                 }
                 if(tableauMatrix[i][2*num_qubits]){
                       tmpstr="-"+tmpstr;
                 }
                 stablizerList->push_back(tmpstr);
            }
         }


        void init_tableau(){
            for(size_t i=0; i< num_qubits*2;i++){
                tableauMatrix[i][i]=1;
            }
        }

        void X(size_t target){

        }

         void Y(size_t target){

         }

        void Z(size_t target){

        }


        void P(size_t target){

        }


        void H(size_t target){

        }


        void CNOT(size_t control,size_t target){

        }

        void CZ(size_t control,size_t target){

        }





         


};




int main() {
    std::cout << "Hello, World!" << std::endl;
    Tableau* tb=new Tableau(5);
    tb->init_tableau();
    tb->print_tableau();
    tb->calculate_stabilizers();
    tb->print_stabilizers();
    return 0;
}