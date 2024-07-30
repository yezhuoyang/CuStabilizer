
#include <iostream>
#include <vector>
using namespace std;


class Tableau {

private:
         size_t rows;
         size_t cols;
         std::vector<std::vector<bool>> tableauMatrix;
public:

         Tableau(size_t rows, size_t cols) : rows(rows), cols(cols) {
             tableauMatrix.resize(rows);
             for (size_t i = 0; i < rows; i++) {
                 tableauMatrix[i].resize(cols);
             }
         }

        void read_instructions_from_file(){

         }

         void print_tableau() {
             for (size_t i = 0; i < rows; i++) {
                 for (size_t j = 0; j < cols; j++) {
                     std::cout << tableauMatrix[i][j] << " ";
                 }
                 std::cout << std::endl;
             }
         }

         


};




int main() {
    std::cout << "Hello, World!" << std::endl;
    Tableau* tb=new Tableau(5,5);
    tb->print_tableau();
    return 0;
}