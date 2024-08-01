#include "tableau.h"
#include <cuda_runtime.h>


void P_cuda(int blocksPerGrid,int threadsPerBlock,unsigned char* tableauMatrix,size_t target,int qubit_num,int rowsize,int N);

void H_cuda(int blocksPerGrid,int threadsPerBlock,unsigned char* tableauMatrix,size_t target,int qubit_num,int rowsize,int N);

void CNOT_cuda(int blocksPerGrid,int threadsPerBlock,unsigned char* tableauMatrix,size_t control,size_t target,int qubit_num,int rowsize,int N);


// Overload the << operator outside the struct without friend
std::ostream& operator<<(std::ostream& os, const  Instruction& inst) {
    switch (inst.type)
    {
    case _CONTROLNOT:
        os << "CNOT " << inst.control << "," << inst.target;
        break;
    case _CONTROLZ:
        os << "CZ " << inst.control << "," << inst.target;       
         break; 
    case _HADAMARD:
        os << "H " <<inst.target;      
        break;
    case _PHASE:
        os << "P " <<inst.target;      
        break;
    case _PAULIX:
        os << "X " <<inst.target;      
        break;    
    case _PAULIY:
        os << "Y " <<inst.target;      
        break;  
    case _PAULIZ:
        os << "Z " <<inst.target;      
        break;               
    default:
        break;
    }
    return os;
}



void Tableau::show_tableau_bit(){
    int rowsize=((2*num_qubits+1)+7)/8; 
    int tmpindex;
    int showint;
    for(int row=0;row<2*num_qubits;row++){
          for(int col=0;col<2*num_qubits+1;col++){
                  //tmpindex=col-8*(col/8);
                  //showint=(tableauMatrix[row*rowsize+col/8]&(0b10000000>>tmpindex));
                  //showint=showint>>(7-tmpindex);
                  showint=getTableauElement(char_tableauMatrix,rowsize, row, col);
                  std::cout<<showint<<" ";
          }
          std::cout<<"\n";
    }
}



void Tableau::show_tableau_char(){
    int rowsize=((2*num_qubits+1)+7)/8; 
    int tmpindex;
    int zstabint;
    int xstabint;
    int phaseint;
    std::string tmpstr;
    for(int row=0;row<num_qubits;row++){
          tmpstr="";
          for(int col=0;col<num_qubits;col++){
                  zstabint=getTableauElement(char_tableauMatrix,rowsize, row, col);
                  xstabint=getTableauElement(char_tableauMatrix,rowsize, row, (col+num_qubits));
                  if((xstabint==0)&&(zstabint==0)){
                        tmpstr=tmpstr+"I";
                  }
                  else if((xstabint==1)&&(zstabint==0)){
                        tmpstr=tmpstr+"X";
                  }
                  else if((xstabint==0)&&(zstabint==1)){
                        tmpstr=tmpstr+"Z";
                  }
                  else{
                         tmpstr=tmpstr+"Y";
                  }
          }
          phaseint=getTableauElement(char_tableauMatrix,rowsize, row, (2*num_qubits));
          if(phaseint==1){
                tmpstr="-"+tmpstr;
          }
          std::cout<<tmpstr<<"\n";
    }
}



void Tableau::checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


Tableau::Tableau(const size_t& num_qubits,const bool& _cudaMode) : num_qubits(num_qubits),cudaMode(_cudaMode){
    tableauMatrix.resize(num_qubits*2);
    for (size_t i = 0; i < num_qubits*2; i++) {
        tableauMatrix[i].resize(num_qubits*2+1);
    }
    instructionSet=new vector<Instruction>();

    //Initialize the cuda memory
    if(cudaMode){
            threadNum=2*num_qubits;
            threadsPerBlock = 2*num_qubits;
            blocksPerGrid =1;
            // Every row is processed in a single thread, every thread is exactly one row of the tableau
            // Every tableau is process in a block, every block is exactly processed in one block
            rowsize=((2*num_qubits+1)+7)/8; 
            charsize=rowsize*(2*num_qubits);
            char_tableauMatrix =(unsigned char*) malloc(charsize);
            cudaMalloc(&cutableauMatrix,charsize);
            checkCudaError("cudaMalloc");
    }

}

Tableau::~Tableau(){
    delete instructionSet;
    if(stablizerList!=nullptr){
        delete stablizerList;
    }
    if(char_tableauMatrix!=nullptr){
        delete char_tableauMatrix;
    }
    if(cudaMode){
          cudaFree(cutableauMatrix);
          checkCudaError("cudaFree");
    }
}


void  Tableau::calculate(){
    for(auto it=instructionSet->begin();it!=instructionSet->end();it++){
        execute_step(*it);
    }
    if(cudaMode){
          cudaMemcpy(char_tableauMatrix,cutableauMatrix,charsize,cudaMemcpyDeviceToHost); 
          checkCudaError("cudaMemcpy to host");
    }
}


void  Tableau::read_instructions_from_file(const string &  filepath){
    ifstream file(filepath);
    vector<std::string> words;
    string word;
    if(!file.is_open()){
        cerr<<"File not exists!"<<endl;
    }
    string line;
    size_t count;
    int control;
    int target;
    Instruction* instpointer;
    while(getline(file,line)){
        words.clear();
        count=0;
        istringstream iss(line);
        while(iss>>word){
            words.push_back(word);
            count++;
        }
        if(count==2){
            if(words[0]=="h"){
                    target  = std::atoi(words[1].c_str());
                    instpointer=new Instruction(_HADAMARD,-1,target);
                    instructionSet->push_back(*instpointer);
            }
            else if(words[0]=="p"){
                    target  = std::atoi(words[1].c_str());
                    instpointer=new Instruction(_PHASE,-1,target);
                    instructionSet->push_back(*instpointer);
            }
            else if(words[0]=="x"){
                    target  = std::atoi(words[1].c_str());
                    instpointer=new Instruction(_PAULIX,-1,target);
                    instructionSet->push_back(*instpointer);
            }
            else if(words[0]=="y"){
                    target  = std::atoi(words[1].c_str());
                    instpointer=new Instruction(_PAULIY,-1,target);
                    instructionSet->push_back(*instpointer);
            }
            else if(words[0]=="z"){
                    target  = std::atoi(words[1].c_str());
                    instpointer=new Instruction(_PAULIZ,-1,target);
                    instructionSet->push_back(*instpointer);
            }
        }
        else if(count==3){
            if(words[0]=="c"){
                    control = std::atoi(words[1].c_str());
                    target  = std::atoi(words[2].c_str());
                    instpointer=new Instruction(_CONTROLNOT,control,target);
                    instructionSet->push_back(*instpointer);
            }
            else if(words[0]=="cz"){
                    control = std::atoi(words[1].c_str());
                    target  = std::atoi(words[2].c_str());
                    instpointer=new Instruction(_CONTROLZ,control,target);
                    instructionSet->push_back(*instpointer);
            }                    
        }
    }
    file.close();
}

void  Tableau::print_instructions(){
    if(instructionSet==nullptr){
        return;
    }
    for(auto it=instructionSet->begin();it!=instructionSet->end();it++){
        cout<<*it<<endl;
    }
}



void  Tableau::execute_step(const Instruction& inst){
    switch (inst.type)
    {
    case _CONTROLNOT:
        CNOT(inst.control,inst.target);
        break;
    case _HADAMARD:
        H(inst.target);
        break;
    case _PHASE:
        P(inst.target);
        break;
    case _PAULIX:
        X(inst.target);
        break;
    case _PAULIY:
        Y(inst.target);
        break;            
    case _PAULIZ:
        Z(inst.target);
        break;
    case _CONTROLZ:
        CZ(inst.control,inst.target);
    default:
        break;
    }
    if(cudaMode){
            cudaDeviceSynchronize();
            checkCudaError("Kernel execution");
    }
}



void  Tableau::print_tableau() {
    for (size_t i = 0; i < num_qubits*2; i++) {
        for (size_t j = 0; j < num_qubits*2+1; j++) {
            std::cout << tableauMatrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}


void  Tableau::print_stabilizers(){
    if(stablizerList==nullptr){
        stablizerList=new vector<string>;
    }
    for (auto it = stablizerList->begin(); it != stablizerList->end(); ++it) {
        std::cout << *it << std::endl;
    }
}


void  Tableau::calculate_stabilizers(){
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


void  Tableau::init_tableau(){
    if(cudaMode){
          //Initialize the tableau
        for(int i=0;i<charsize;++i){
            char_tableauMatrix[i]=0;
        }
        for(int k=0;k<2*num_qubits;k++){
            setTableauValue(char_tableauMatrix,rowsize, k, k,1);
        }
        cudaMemcpy(cutableauMatrix,char_tableauMatrix,charsize,cudaMemcpyHostToDevice); 
        checkCudaError("cudaMemcpy to device");  
        return;
    }
    for(size_t i=0; i< num_qubits*2;i++){
        tableauMatrix[i][i]=1;
    }
}

void  Tableau::X(const size_t& target){
    H(target);
    Z(target);
    H(target);
}

void  Tableau::Y(const size_t& target){
    X(target);
    Z(target);
}

void  Tableau::Z(const size_t& target){
    P(target);
    P(target);
}

void Tableau::P(const size_t& target){
    if(cudaMode){
        P_cuda(blocksPerGrid, threadsPerBlock,cutableauMatrix,target,num_qubits,rowsize,2*num_qubits);
        return;
    }
    for(size_t k=0;k<2*num_qubits;k++){
        tableauMatrix[k][2*num_qubits]=((tableauMatrix[k][2*num_qubits])!=(tableauMatrix[k][target]&&tableauMatrix[k][target+num_qubits]));
        tableauMatrix[k][target]=(tableauMatrix[k][target]!=tableauMatrix[k][target+num_qubits]);
    }
}


void Tableau::H(const size_t& target){
    if(cudaMode){
         H_cuda(blocksPerGrid, threadsPerBlock,cutableauMatrix,target,num_qubits,rowsize,2*num_qubits);
         return;
    }
    bool tmp;
    for(size_t k=0;k<2*num_qubits;k++){
        tableauMatrix[k][2*num_qubits]=((tableauMatrix[k][2*num_qubits])!=(tableauMatrix[k][target]&&tableauMatrix[k][target+num_qubits]));
        tmp=tableauMatrix[k][target];
        tableauMatrix[k][target]=tableauMatrix[k][target+num_qubits];
        tableauMatrix[k][target+num_qubits]=tmp;
    }
}

void Tableau::CNOT(const size_t& control,const size_t& target){
    if(cudaMode){
        CNOT_cuda(blocksPerGrid, threadsPerBlock,cutableauMatrix,control,target,num_qubits,rowsize,2*num_qubits);
        return;
    }
    bool multi;
    bool xorsum;
    for(size_t k=0;k<2*num_qubits;k++){
        multi=(tableauMatrix[k][control+num_qubits]&&tableauMatrix[k][target]);
        xorsum=(tableauMatrix[k][target+num_qubits]==tableauMatrix[k][control]);
        tableauMatrix[k][2*num_qubits]=(tableauMatrix[k][2*num_qubits]!=multi);
        tableauMatrix[k][2*num_qubits]=(tableauMatrix[k][2*num_qubits]!=xorsum);
        tableauMatrix[k][target+num_qubits]=(tableauMatrix[k][target+num_qubits]!=tableauMatrix[k][control+num_qubits]);
        tableauMatrix[k][control]=(tableauMatrix[k][control]!=tableauMatrix[k][target]);
    } 
}

void Tableau::CZ(const size_t& control,const size_t& target){
    H(target);
    CNOT(control,target);
    H(target);          
}


