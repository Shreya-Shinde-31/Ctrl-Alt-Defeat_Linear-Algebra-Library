/*Linear algebra library includes the following operations/functions:
1.Matrix-vector operations:
These operations include matrix-vector multiplication, vector addition and subtraction, and multiplication of a vector by a scalar.
2.Matrix-matrix operations:
These operations include matrix addition and subtraction, matrix multiplication, and matrix inversion.
3.Linear equation solving:
These operations include solving systems of linear equations and finding the eigenvalues of a matrix.
4.Norms and inner products:
These operations include calculating the norm of a vector or matrix, and calculating the inner product.
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace std;

template <typename T>
class Matrix;

template <typename T>
T determinant(const Matrix<T>& mat);

template <typename T>
class Matrix {
private:
    vector<vector<T>> data;

public:
    Matrix(size_t rows, size_t cols) : data(rows, vector<T>(cols)) {}

    void setElement(size_t row, size_t col, T value) {
        data[row][col] = value;
    }

    T getElement(size_t row, size_t col) const {
        return data[row][col];
    }

    size_t numRows() const {
        return data.size();
    }

    size_t numCols() const {
        return data.empty() ? 0 : data[0].size();
    }

    void display() const {
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < numCols(); ++j) {
                cout << data[i][j] << " ";
            }
            cout << endl;
        }
    }
//Matrix Addition
    Matrix<T> operator+(const Matrix<T>& other) const {
        if (numRows() != other.numRows() || numCols() != other.numCols()) {
            throw invalid_argument("Matrix dimensions must match for addition");
        }
        Matrix<T> result(numRows(), numCols());
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < numCols(); ++j) {
                result.setElement(i, j, data[i][j] + other.getElement(i, j));
            }
        }
        return result;
    }
//Matrix Subtraction
    Matrix<T> operator-(const Matrix<T>& other) const {
        if (numRows() != other.numRows() || numCols() != other.numCols()) {
            throw invalid_argument("Matrix dimensions must match for subtraction");
        }
        Matrix<T> result(numRows(), numCols());
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < numCols(); ++j) {
                result.setElement(i, j, data[i][j] - other.getElement(i, j));
            }
        }
        return result;
    }
//Scalar Multiplication
    Matrix<T> operator*(T scalar) const {
        Matrix<T> result(numRows(), numCols());
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < numCols(); ++j) {
                result.setElement(i, j, data[i][j] * scalar);
            }
        }
        return result;
    }
//Matrix Multiplication
    Matrix<T> operator*(const Matrix<T>& other) const {
        if (numCols() != other.numRows()) {
            throw invalid_argument("Number of columns in first matrix must match number of rows in second matrix for multiplication");
        }
        Matrix<T> result(numRows(), other.numCols());
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < other.numCols(); ++j) {
                T sum = 0;
                for (size_t k = 0; k < numCols(); ++k) {
                    sum += data[i][k] * other.getElement(k, j);
                }
                result.setElement(i, j, sum);
            }
        }
        return result;
    }
//Transpose
    Matrix<T> transpose() const {
        Matrix<T> result(numCols(), numRows());
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < numCols(); ++j) {
                result.setElement(j, i, data[i][j]);
            }
        }
        return result;
    }
//Norm
    T norm() const {
        T sum = 0;
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < numCols(); ++j) {
                sum += data[i][j] * data[i][j];
            }
        }
        return sqrt(sum);
    }
//Inner Product
    T innerProduct(const Matrix<T>& other) const {
        if (numRows() != other.numRows() || numCols() != 1 || other.numCols() != 1) {
            throw invalid_argument("Both inputs must be column vectors of the same size for inner product");
        }
        T sum = 0;
        for (size_t i = 0; i < numRows(); ++i) {
            sum += data[i][0] * other.getElement(i, 0);
        }
        return sum;
    }
//inverse
    Matrix<T> inverse() const {
        // Only implemented for 2x2 matrices for demonstration purposes
        if (numRows() != numCols() || numRows() != 2) {
            throw invalid_argument("Matrix inversion is only supported for 2x2 matrices");
        }
        T det = determinant(*this);
        if (det == 0) {
            throw invalid_argument("Matrix is singular, cannot be inverted");
        }
        Matrix<T> result(2, 2);
        result.setElement(0, 0, data[1][1] / det);
        result.setElement(0, 1, -data[0][1] / det);
        result.setElement(1, 0, -data[1][0] / det);
        result.setElement(1, 1, data[0][0] / det);
        return result;
    }
//Eigen Values
    vector<T> eigenValues() const {
        if (numRows() != numRows()) {
            throw invalid_argument("Eigenvalues can only be calculated for square matrices");
        }

        if (numRows() != 2) {
            throw invalid_argument("Eigenvalues calculation is only supported for 2x2 matrices");
        }
        T a = data[0][0];
        T b = data[0][1];
        T c = data[1][0];
        T d = data[1][1];
        T discriminant = sqrt((a + d) * (a + d) - 4 * (a * d - b * c));
        T lambda1 = (a + d + discriminant) / 2;
        T lambda2 = (a + d - discriminant) / 2;
        return {lambda1, lambda2};
    }
};

// determinant function as a friend of the Matrix class template
template<typename T>
T determinant(const Matrix<T>& mat) {
    // Implementation for 2x2 matrix determinant
    if (mat.numRows() != 2 || mat.numCols() != 2) {
        throw invalid_argument("Determinant calculation is only supported for 2x2 matrices");
    }
    return mat.getElement(0, 0) * mat.getElement(1, 1) - mat.getElement(0, 1) * mat.getElement(1, 0);
}

template <typename T>
void performAddition(const Matrix<T>& mat1, const Matrix<T>& mat2, bool isMatrix1, bool isMatrix2) {
    Matrix<T> result_add = mat1 + mat2;
    cout << "Addition Result:" << endl;
    result_add.display();
}

template <typename T>
void performSubtraction(const Matrix<T>& mat1, const Matrix<T>& mat2, bool isMatrix1, bool isMatrix2) {
    Matrix<T> result_sub = mat1 - mat2;
    cout << "Subtraction Result:" << endl;
    result_sub.display();
}

template <typename T>
void performScalarMultiplication(const Matrix<T>& mat, bool isMatrix) {
    T scalar;
    cout << "Enter the scalar value: ";
    cin >> scalar;
    Matrix<T> result_mult = mat * scalar;
    cout << "Scalar Multiplication Result:" << endl;
    result_mult.display();
}

template <typename T>
void performMatrixMultiplication(const Matrix<T>& mat1, const Matrix<T>& mat2, bool isMatrix1, bool isMatrix2) {
    Matrix<T> result_mult = mat1 * mat2;
    cout << "Matrix Multiplication Result:" << endl;
    result_mult.display();
}

template <typename T>
void performTranspose(const Matrix<T>& mat, bool isMatrix) {
    Matrix<T> result_transpose = mat.transpose();
    cout << "Transpose Result:" << endl;
    result_transpose.display();
}

template <typename T>
void performNorm(const Matrix<T>& mat, bool isMatrix) {
    T matrixNorm = mat.norm();
    cout << "Norm Result:" << endl;
    cout << matrixNorm << endl;
}

template <typename T>
void performInnerProduct(const Matrix<T>& mat1, const Matrix<T>& mat2, bool isMatrix1, bool isMatrix2) {
    T innerProd = mat1.innerProduct(mat2);
    cout << "Inner Product Result:" << endl;
    cout << innerProd << endl;
}

template <typename T>
void performInverse(const Matrix<T>& mat, bool isMatrix) {
    try {
        Matrix<T> result_inverse = mat.inverse();
        cout << "Inverse Result:" << endl;
        result_inverse.display();
    } catch (const invalid_argument& e) {
        cerr << e.what() << endl;
    }
}

template <typename T>
void performEigenValues(const Matrix<T>& mat, bool isMatrix) {
    try {
        vector<T> eigenVals = mat.eigenValues();
        cout << "Eigenvalues:" << endl;
        cout << "λ1: " << eigenVals[0] << ", λ2: " << eigenVals[1] << endl;
    } catch (const invalid_argument& e) {
        cerr << e.what() << endl;
    }
}

int main() {
    size_t rows1, cols1, rows2, cols2;

    cout << "Enter the number of rows for matrix 1: ";
    cin >> rows1;
    cout << "Enter the number of columns for matrix 1: ";
    cin >> cols1;

    Matrix<int> mat1(rows1, cols1);
    cout << "Enter elements for matrix 1:" << endl;
    for (size_t i = 0; i < rows1; ++i) {
        for (size_t j = 0; j < cols1; ++j) {
            int value;
            cout << "Enter element at position (" << i << ", " << j << "): ";
            cin >> value;
            mat1.setElement(i, j, value);
        }
    }

    cout << "Matrix 1:" << endl;
    mat1.display();

    cout << "Enter the number of rows for matrix 2: ";
    cin >> rows2;
    cout << "Enter the number of columns for matrix 2: ";
    cin >> cols2;

    Matrix<int> mat2(rows2, cols2);
    cout << "Enter elements for matrix 2:" << endl;
    for (size_t i = 0; i < rows2; ++i) {
        for (size_t j = 0; j < cols2; ++j) {
            int value;
            cout << "Enter element at position (" << i << ", " << j << "): ";
            cin >> value;
            mat2.setElement(i, j, value);
        }
    }

    cout << "Matrix 2:" << endl;
    mat2.display();

    int choice;
    do {
        cout << "Menu:" << endl;
        cout << "1. Matrix Addition" << endl;
        cout << "2. Matrix Subtraction" << endl;
        cout << "3. Scalar Multiplication" << endl;
        cout << "4. Matrix Multiplication" << endl;
        cout << "5. Transpose" << endl;
        cout << "6. Norm" << endl;
        cout << "7. Inner Product" << endl;
        cout << "8. Inverse" << endl;
        cout << "9. Eigenvalues" << endl;
        cout << "10. Exit" << endl;
        cout << "Enter your choice: ";
        cin >> choice;

        switch (choice) {
            case 1:
                performAddition(mat1, mat2, true, true);
                break;
            case 2:
                performSubtraction(mat1, mat2, true, true);
                break;
            case 3:
                performScalarMultiplication(mat1, true);
                break;
            case 4:
                performMatrixMultiplication(mat1, mat2, true, true);
                break;
            case 5:
                cout << "Select the target for transpose:" << endl;
                cout << "1. Matrix 1" << endl;
                cout << "2. Matrix 2" << endl;
                cout << "Enter your choice: ";
                int targetTranspose;
                cin >> targetTranspose;
                switch (targetTranspose) {
                    case 1:
                        performTranspose(mat1, true);
                        break;
                    case 2:
                        performTranspose(mat2, true);
                        break;
                    default:
                        cerr << "Invalid choice for transpose" << endl;
                }
                break;
            case 6:
                cout << "Select the target for norm:" << endl;
                cout << "1. Matrix 1" << endl;
                cout << "2. Matrix 2" << endl;
                cout << "Enter your choice: ";
                int targetNorm;
                cin >> targetNorm;
                switch (targetNorm) {
                    case 1:
                        performNorm(mat1, true);
                        break;
                    case 2:
                        performNorm(mat2, true);
                        break;
                    default:
                        cerr << "Invalid choice for norm" << endl;
                }
                break;
            case 7:
                performInnerProduct(mat1, mat2, true, true);
                break;
            case 8:
                cout << "Select the target for inverse:" << endl;
                cout << "1. Matrix 1" << endl;
                cout << "2. Matrix 2" << endl;
                cout << "Enter your choice: ";
                int targetInverse;
                cin >> targetInverse;
                switch (targetInverse) {
                    case 1:
                        performInverse(mat1, true);
                        break;
                    case 2:
                        performInverse(mat2, true);
                        break;
                    default:
                        cerr << "Invalid choice for inverse" << endl;
                }
                break;
            case 9:
                cout << "Select the target for eigenvalues:" << endl;
                cout << "1. Matrix 1" << endl;
                cout << "2. Matrix 2" << endl;
                cout << "Enter your choice: ";
                int targetEigen;
                cin >> targetEigen;
                switch (targetEigen) {
                    case 1:
                        performEigenValues(mat1, true);
                        break;
                    case 2:
                        performEigenValues(mat2, true);
                        break;
                    default:
                        cerr << "Invalid choice for eigenvalues" << endl;
                }
                break;
            case 10:
                cout << "Exiting..." << endl;
                break;
            default:
                cerr << "Invalid choice" << endl;
        }
    } while (choice != 10);

    return 0;
}
