#include "tableau.h"






int main() {
    std::cout << "Hello, World!" << std::endl;
    Tableau* tb=new Tableau(5);
    tb->init_tableau();
    tb->print_tableau();
    tb->calculate_stabilizers();
    tb->print_stabilizers();
    tb->read_instructions_from_file("../testcases/example1.stab");
    tb->print_instructions();
    tb->calculate();
    tb->calculate_stabilizers();
    tb->print_stabilizers();
    return 0;
}