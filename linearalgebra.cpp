#include <iostream>
#include <type_traits>
#include <cmath>

template <typename T>
struct Numeric {
    static constexpr bool value = std::is_arithmetic<T>::value;
};

template <typename T, size_t Rows, size_t Cols>
class Matrix {
    template <typename U, size_t R, size_t C>
    friend class Matrix;

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
            result.data[i][0] = sum;
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

    auto operator*(const Matrix<T, Cols, Rows>& other) const {
        Matrix<T, Rows, Rows> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Rows; ++j) {
                T sum = 0;
                for (size_t k = 0; k < Cols; ++k)
                    sum += data[i][k] * other.data[k][j];
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    auto inverse() const {
        static_assert(Rows == Cols, "Matrix must be square for inversion");
        Matrix<T, Rows, Rows> result(*this);
        Matrix<T, Rows, Rows> identity;
        identity.makeIdentity();

        // Gauss-Jordan elimination
        for (size_t i = 0; i < Rows; ++i) {
            // Swap rows if necessary to get a non-zero pivot
            if (result.data[i][i] == 0) {
                size_t swapRow = i + 1;
                while (swapRow < Rows && result.data[swapRow][i] == 0)
                    ++swapRow;
                if (swapRow == Rows) {
                    // Matrix is singular, return an identity matrix
                    return identity;
                }
                result.swapRows(i, swapRow);
                identity.swapRows(i, swapRow);
            }

            // Scale row to have a pivot of 1
            T pivot = result.data[i][i];
            for (size_t j = 0; j < Rows; ++j) {
                result.data[i][j] /= pivot;
                identity.data[i][j] /= pivot;
            }

            // Zero out other entries in the column
            for (size_t k = 0; k < Rows; ++k) {
                if (k != i) {
                    T factor = result.data[k][i];
                    for (size_t j = 0; j < Rows; ++j) {
                        result.data[k][j] -= factor * result.data[i][j];
                        identity.data[k][j] -= factor * identity.data[i][j];
                    }
                }
            }
        }

        return identity;
    }

    auto luDecomposition() const {
        Matrix<T, Rows, Rows> lower, upper;

        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = i; j < Rows; ++j) {
                T sum = 0;
                for (size_t k = 0; k < i; ++k)
                    sum += lower.data[i][k] * upper.data[k][j];
                upper.data[i][j] = data[i][j] - sum;
            }
            lower.data[i][i] = 1;
            for (size_t j = i + 1; j < Rows; ++j) {
                T sum = 0;
                for (size_t k = 0; k < i; ++k)
                    sum += lower.data[j][k] * upper.data[k][i];
                lower.data[j][i] = (data[j][i] - sum) / upper.data[i][i];
            }
        }

        return std::make_pair(lower, upper);
    }

    auto qrDecomposition() const {
        Matrix<T, Rows, Rows> q, r;

        for (size_t j = 0; j < Rows; ++j) {
            Matrix<T, Rows, 1> v = getColumn(j);
            for (size_t i = 0; i < j; ++i) {
                T dotProduct = q.getColumn(i) * v;
                v = v - dotProduct * q.getColumn(i);
            }
            T norm = v.norm();
            q.setColumn(j, v / norm);
            for (size_t i = 0; i < Rows; ++i) {
                T dotProduct = q.getColumn(j) * getColumn(i);
                r.data[j][i] = dotProduct;
            }
        }

        return std::make_pair(q, r);
    }

    Matrix<T, Rows, 1> getColumn(size_t col) const {
        Matrix<T, Rows, 1> column;
        for (size_t i = 0; i < Rows; ++i)
            column.data[i][0] = data[i][col];
        return column;
    }

    void setColumn(size_t col, const Matrix<T, Rows, 1>& column) {
        for (size_t i = 0; i < Rows; ++i)
            data[i][col] = column.data[i][0];
    }

    void makeIdentity() {
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                data[i][j] = (i == j) ? 1 : 0;
    }

    void swapRows(size_t row1, size_t row2) {
        for (size_t j = 0; j < Cols; ++j) {
            T temp = data[row1][j];
            data[row1][j] = data[row2][j];
            data[row2][j] = temp;
        }
    }

    void display() const {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    T norm() const {
        T sum = 0;
        for (size_t i = 0; i < Rows; ++i)
            sum += data[i][0] * data[i][0];
        return std::sqrt(sum);
    }
};

int main() {
    double arr1[2][2] = {{1, 2}, {3, 4}};
    double arr2[2][2] = {{1, 0}, {0, 1}};

    Matrix<double, 2, 2> mat1(arr1);
    Matrix<double, 2, 2> mat2(arr2);

    std::cout << "Matrix 1:" << std::endl;
    mat1.display();

    std::cout << "Matrix 2:" << std::endl;
    mat2.display();

    auto result_addition = mat1 + mat2;
    std::cout << "Matrix addition result:" << std::endl;
    result_addition.display();

    auto result_subtraction = mat1 - mat2;
    std::cout << "Matrix subtraction result:" << std::endl;
    result_subtraction.display();

    auto result_multiplication = mat1 * mat2;
    std::cout << "Matrix multiplication result:" << std::endl;
    result_multiplication.display();

    auto result_inverse = mat1.inverse();
    std::cout << "Matrix inversion result:" << std::endl;
    result_inverse.display();

    auto lu = mat1.luDecomposition();
    std::cout << "LU Decomposition:" << std::endl;
    std::cout << "Lower Matrix:" << std::endl;
    lu.first.display();
    std::cout << "Upper Matrix:" << std::endl;
    lu.second.display();

    auto qr = mat1.qrDecomposition();
    std::cout << "QR Decomposition:" << std::endl;
    std::cout << "Q Matrix:" << std::endl;
    qr.first.display();
    std::cout << "R Matrix:" << std::endl;
    qr.second.display();

    return 0;
}
