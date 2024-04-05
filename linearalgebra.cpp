#include <iostream>
#include <array>
#include <cmath>

// Type traits to validate matrix size
template <class T, size_t rows, size_t cols>
struct MatrixTraits {
    static constexpr bool valid = true;
};

// Class representing a matrix
template <class T, size_t rows, size_t cols>
class Matrix {
    static_assert(MatrixTraits<T, rows, cols>::valid, "Invalid matrix size");

    std::array<std::array<T, cols>, rows> data; // Data storage for the matrix

public:
    // Constructor for initializing the matrix with values
    template<typename... Args>
    Matrix(Args... args) : data{std::array<T, cols>{args...}...} {
        static_assert(sizeof...(args) == rows * cols, "Wrong number of arguments");
    }

    // Matrix-vector multiplication
    auto operator*(const std::array<T, cols>& vec) const {
        std::array<T, rows> result = {};
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i] += data[i][j] * vec[j];
            }
        }
        return result;
    }

    // Vector addition
    auto operator+(const std::array<T, cols>& vec) const {
        auto result = data;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] += vec[j];
            }
        }
        return result;
    }

    // Vector subtraction
    auto operator-(const std::array<T, cols>& vec) const {
        auto result = data;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] -= vec[j];
            }
        }
        return result;
    }

    // Scalar multiplication for a matrix
    auto operator*(const T& scalar) const {
        auto result = data;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] *= scalar;
            }
        }
        return result;
    }

    // Matrix multiplication
    template <size_t other_cols>
    auto operator*(const Matrix<T, cols, other_cols>& other) const {
        Matrix<T, rows, other_cols> result;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other_cols; ++j) {
                result.data[i][j] = 0;
                for (size_t k = 0; k < cols; ++k) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    // Print matrix
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Trigonometric functions
    auto sin() const {
        Matrix<T, rows, cols> result;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = std::sin(data[i][j]);
            }
        }
        return result;
    }

    auto cos() const {
        Matrix<T, rows, cols> result;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = std::cos(data[i][j]);
            }
        }
        return result;
    }

    auto tan() const {
        Matrix<T, rows, cols> result;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = std::tan(data[i][j]);
            }
        }
        return result;
    }

    auto asin() const {
        Matrix<T, rows, cols> result;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = std::asin(data[i][j]);
            }
        }
        return result;
    }

    auto acos() const {
        Matrix<T, rows, cols> result;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = std::acos(data[i][j]);
            }
        }
        return result;
    }

    auto atan() const {
        Matrix<T, rows, cols> result;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = std::atan(data[i][j]);
            }
        }
        return result;
    }
};

int main() {
    // Create matrices and vector
    Matrix<double, 2, 2> A(1, 2, 3, 4);
    Matrix<double, 2, 2> B(5, 6, 7, 8);
    std::array<double, 2> vec = {5, 6};

    // Perform operations
    auto matrixVectorResult = A * vec;
    auto vectorAdditionResult = A + vec;
    auto vectorSubtractionResult = A - vec;
    auto scalarMultiplicationResult = A * 2.0;
    auto matrixMultiplicationResult = A * B;

    auto sineResult = A.sin();
    auto cosineResult = A.cos();
    auto tangentResult = A.tan();
    auto arcsineResult = A.asin();
    auto arccosineResult = A.acos();
    auto arctangentResult = A.atan();

    // Output results
    std::cout << "Matrix-Vector Multiplication Result: ";
    for (size_t i = 0; i < 2; ++i) {
        std::cout << matrixVectorResult[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Vector Addition Result: ";
    for (size_t i = 0; i < 2; ++i) {
        std::cout << vectorAdditionResult.data[0][i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Vector Subtraction Result: ";
    for (size_t i = 0; i < 2; ++i) {
        std::cout << vectorSubtractionResult.data[0][i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Scalar Multiplication for Vector Result: ";
    for (size_t i = 0; i < 2; ++i) {
        std::cout << scalarMultiplicationResult.data[0][i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Matrix Multiplication Result:" << std::endl;
    matrixMultiplicationResult.print();

    std::cout << "Sine of Matrix A:" << std::endl;
    sineResult.print();

    std::cout << "Cosine of Matrix A:" << std::endl;
    cosineResult.print();

    std::cout << "Tangent of Matrix A:" << std::endl;
    tangentResult.print();

    std::cout << "Arcsine of Matrix A:" << std::endl;
    arcsineResult.print();

    std::cout << "Arccosine of Matrix A:" << std::endl;
    arccosineResult.print();

    std::cout << "Arctangent of Matrix A:" << std::endl;
    arctangentResult.print();

    return 0;
}
