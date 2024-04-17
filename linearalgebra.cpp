/*Linear algebra library includes the following operations/functions:
1.Matrix-vector operations:
These operations include matrix-vector multiplication, vector addition and subtraction, and multiplication of a vector by a scalar.
2.Matrix-matrix operations:
These operations include matrix addition and subtraction, matrix multiplication, and matrix inversion.
3.Linear equation solving:
These operations include solving systems of linear equations and finding the eigenvalues of a matrix.
4.Decompositions:
These operations include LU decomposition.
5.Norms and inner products:
These operations include calculating the norm of a vector or matrix, and calculating the inner product of two vectors.
*/

#include <iostream>
#include <type_traits>
#include <cmath>
#include <stdexcept>
#include <vector>

template <typename T>
struct Numeric {
    static constexpr bool value = std::is_arithmetic<T>::value;
};

template <typename T, size_t Rows, size_t Cols>
class Matrix {
    template <typename U, size_t R, size_t C>
    friend class Matrix;

    template <typename U, size_t R, size_t C1, size_t C2>
    friend auto operator*(const Matrix<U, R, C1>&, const Matrix<U, C1, C2>&);

    template <typename U, size_t R>
    friend auto operator*(const Matrix<U, R, 1>&, const Matrix<U, 1, R>&);

private:
    T data[Rows][Cols];

public:
    Matrix() {}

    explicit Matrix(const T& scalar) {
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                data[i][j] = scalar;
    }

    Matrix(const T(&arr)[Rows][Cols]) {
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                data[i][j] = arr[i][j];
    }

    template <typename U>
    auto operator*(const U& vec) const {
        static_assert(Cols == Rows, "Invalid vector size");
        using ResultType = std::decay_t<decltype(data[0][0] * vec.data[0][0])>;
        Matrix<ResultType, Rows, 1> result;

        for (size_t i = 0; i < Rows; ++i) {
            ResultType sum = 0;
            for (size_t j = 0; j < Cols; ++j)
                sum += data[i][j] * vec.data[j][0];
            result(i, 0) = sum;
        }
        return result;
    }

    auto operator+(const Matrix<T, Rows, Cols>& other) const {
        Matrix<T, Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    auto operator-(const Matrix<T, Rows, Cols>& other) const {
        Matrix<T, Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                result.data[i][j] = data[i][j] - other.data[i][j];
        return result;
    }

    auto operator*(const T& scalar) const {
        Matrix<T, Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                result.data[i][j] = data[i][j] * scalar;
        return result;
    }

    auto operator*(const Matrix<T, Rows, Cols>& other) const {
        Matrix<T, Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                result.data[i][j] = data[i][j] * other.data[i][j];
        return result;
    }

    T& operator()(size_t row, size_t col) {
        return data[row][col];
    }

    const T& operator()(size_t row, size_t col) const {
        return data[row][col];
    }

    void display() const {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    Matrix<T, Rows, Cols> invert() const {
        if (Rows != Cols) {
            throw std::runtime_error("Matrix must be square for inversion");
        } else {
            Matrix<T, Rows, Cols> identity;
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = 0; j < Cols; ++j) {
                    identity.data[i][j] = (i == j) ? 1 : 0;
                }
            }

            Matrix<T, Rows, Cols> augmented(*this);
            for (size_t i = 0; i < Rows; ++i) {
                T pivot = augmented.data[i][i];
                if (pivot == 0) {
                    throw std::runtime_error("Matrix is singular");
                }

                for (size_t j = 0; j < Cols; ++j) {
                    augmented.data[i][j] /= pivot;
                    identity.data[i][j] /= pivot;
                }

                for (size_t k = 0; k < Rows; ++k) {
                    if (k != i) {
                        T factor = augmented.data[k][i];
                        for (size_t j = 0; j < Cols; ++j) {
                            augmented.data[k][j] -= augmented.data[i][j] * factor;
                            identity.data[k][j] -= identity.data[i][j] * factor;
                        }
                    }
                }
            }

            return identity;
        }
    }

    Matrix<T, Cols, Rows> transpose() const {
        Matrix<T, Cols, Rows> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    std::vector<T> eigenvalues(size_t iterations = 1000, double epsilon = 1e-6) const {
        if (Rows != Cols) {
            throw std::runtime_error("Matrix must be square to compute eigenvalues");
        }

        std::vector<T> eigenvalues;
        Matrix<T, Rows, 1> b;

        // Start with a random vector
        for (size_t i = 0; i < Rows; ++i) {
            b(i, 0) = static_cast<T>(rand()) / RAND_MAX; // Random initialization
        }

        for (size_t iter = 0; iter < iterations; ++iter) {
            Matrix<T, Rows, 1> b_next = (*this) * b;
            T norm = 0;
            T diff = 0;

            // Calculate norm and diff
            for (size_t i = 0; i < Rows; ++i) {
                norm += b_next(i, 0) * b_next(i, 0);
                diff += std::abs(b_next(i, 0) - b(i, 0));
            }

            // Normalize b_next
            norm = std::sqrt(norm);
            for (size_t i = 0; i < Rows; ++i) {
                b_next(i, 0) /= norm;
            }

            // Check convergence
            if (diff < epsilon) {
                break;
            }

            b = b_next;
        }

        // Rayleigh quotient
        Matrix<T, 1, Rows> b_transpose = b.transpose();
        Matrix<T, 1, 1> eigenvalue = b_transpose * (*this * b);

        eigenvalues.push_back(eigenvalue(0, 0));
        return eigenvalues;
    }
    //LU Decomposition
    std::pair<Matrix<T, Rows, Rows>, Matrix<T, Rows, Cols>> luDecomposition() const {
        if (Rows != Cols) {
            throw std::runtime_error("LU decomposition requires a square matrix");
        }

        Matrix<T, Rows, Rows> L, U;
        U = *this;

        for (size_t i = 0; i < Rows; ++i) {
            L(i, i) = 1;

            for (size_t k = i + 1; k < Rows; ++k) {
                if (U(i, i) == 0) {
                    throw std::runtime_error("LU decomposition failed: zero pivot encountered");
                }

                T factor = U(k, i) / U(i, i);
                L(k, i) = factor;

                for (size_t j = i; j < Rows; ++j) {
                    U(k, j) -= factor * U(i, j);
                }
            }
        }

        return std::make_pair(L, U);
    }
    //Norm 
    T norm() const {
        T sum = 0;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                sum += data[i][j] * data[i][j];
            }
        }
        return std::sqrt(sum);
    }

    T innerProduct(const Matrix<T, Rows, 1>& other) const {
        T sum = 0;
        for (size_t i = 0; i < Rows; ++i) {
            sum += data[i][0] * other(i, 0);
        }
        return sum;
    }
};

template <typename T, size_t Rows, size_t Cols1, size_t Cols2>
auto operator*(const Matrix<T, Rows, Cols1>& mat1, const Matrix<T, Cols1, Cols2>& mat2) {
    using ResultType = std::decay_t<decltype(T() * T())>;
    Matrix<ResultType, Rows, Cols2> result;

    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < Cols2; ++j) {
            ResultType sum = 0;
            for (size_t k = 0; k < Cols1; ++k) {
                sum += mat1(i, k) * mat2(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

template <typename T, size_t Rows>
auto operator*(const Matrix<T, Rows, 1>& mat, const Matrix<T, 1, Rows>& vec) {
    T result = 0;
    for (size_t i = 0; i < Rows; ++i)
        result += mat(i, 0) * vec(0, i);
    return result;
}

int main() {
    double arr1[2][2] = {{4, 7}, {2, 6}};
    double arr2[2][2] = {{1, 2}, {3, 4}};
    double arr3[2][1] = {{1}, {2}};

    Matrix<double, 2, 2> mat1(arr1);
    Matrix<double, 2, 2> mat2(arr2);
    Matrix<double, 2, 1> mat3(arr3);

    std::cout << "Matrix 1:" << std::endl;
    mat1.display();

    std::cout << "Matrix 2:" << std::endl;
    mat2.display();

    std::cout << "Vector:" << std::endl;
    mat3.display();

    auto result1 = mat1 * mat2;
    std::cout << "Matrix-matrix multiplication result:" << std::endl;
    result1.display();

    auto result2 = mat1 * mat3;
    std::cout << "Matrix-vector multiplication result:" << std::endl;
    result2.display();

    auto result3 = mat1 + mat2;
    std::cout << "Matrix addition result:" << std::endl;
    result3.display();

    auto result4 = mat1 - mat2;
    std::cout << "Matrix subtraction result:" << std::endl;
    result4.display();

    auto result5 = mat3 * 2.0; // Multiplication of vector by a scalar
    std::cout << "Vector multiplication by scalar result:" << std::endl;
    result5.display();

    try {
        Matrix<double, 2, 2> inverse = mat1.invert();
        std::cout << "Inverse Matrix:" << std::endl;
        inverse.display();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    try {
        // Calculate eigenvalues
        auto eigenvalues = mat1.eigenvalues();
        std::cout << "Eigenvalues:" << std::endl;
        for (const auto& value : eigenvalues) {
            std::cout << value << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    try {
        double arr1[3][3] = {{4, 3, 2}, {2, 2, 3}, {3, 1, 2}};
        Matrix<double, 3, 3> mat1(arr1);

        std::cout << "Matrix for LU decomposition:" << std::endl;
        mat1.display();

        // Perform LU decomposition
        auto luDecomp = mat1.luDecomposition();
        Matrix<double, 3, 3> L = luDecomp.first;
        Matrix<double, 3, 3> U = luDecomp.second;

        std::cout << "Lower Triangular Matrix (L):" << std::endl;
        L.display();

        std::cout << "Upper Triangular Matrix (U):" << std::endl;
        U.display();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // Define vectors
    Matrix<double, 2, 1> vec1;
    vec1(0, 0) = 1;
    vec1(1, 0) = 2;

    Matrix<double, 2, 1> vec2;
    vec2(0, 0) = 3;
    vec2(1, 0) = 4;

    // Calculate norm of vectors
    std::cout << "Norm of vec1: " << vec1.norm() << std::endl;
    std::cout << "Norm of vec2: " << vec2.norm() << std::endl;

    // Calculate inner product of vectors
    std::cout << "Inner product of vec1 and vec2: " << vec1.innerProduct(vec2) << std::endl;

    return 0;
}
