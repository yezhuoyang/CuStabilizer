
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
                string line;
                ifstream file("instructions.txt");
                if (file.is_open()) {
                    while (getline(file, line)) {
                        std::cout << line << std::endl;
                    }
                    file.close();
                }
         }

         void print_tableau() {
             for (size_t i = 0; i < rows; i++) {
                 for (size_t j = 0; j < cols; j++) {
                     std::cout << tableauMatrix[i][j] << " ";
                 }
                 std::cout << std::endl;
             }
         }

         


}




int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}