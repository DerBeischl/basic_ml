#include "matrix.hpp"
#include <cassert>

namespace ml
{

    Matrix::Matrix(const size_t width, const size_t height)
        : is_transposed(false), width_(width), height_(height),
          storage_(width * height)
    {
    }

    Matrix::Matrix(const std::initializer_list<double> &data)
        : is_transposed(false), width_(1), height_(data.size()),
          storage_(data)
    {
    }

    size_t Matrix::width() const
    {
        return is_transposed ? height_ : width_;
    }

    size_t Matrix::height() const
    {
        return is_transposed ? width_ : height_;
    }

    Matrix &Matrix::outer_product(const Matrix &rhs)
    {
        assert((rhs.height() == height() && rhs.width() == width()));

        for (size_t i = 0; i < height() * width(); i++)
            storage_[i] *= rhs.storage_[i];

        return *this;
    }

    double Matrix::operator()(const size_t x, const size_t y) const
    {
        return is_transposed ? storage_[y + x * width_] : storage_[x + y * width_];
    }

    double &Matrix::operator()(const size_t x, const size_t y)
    {
        return is_transposed ? storage_[y + x * width_] : storage_[x + y * width_];
    }

    double Matrix::operator[](const size_t i) const
    {
        return storage_[i];
    }

    double &Matrix::operator[](const size_t i)
    {
        return storage_[i];
    }

    Matrix &Matrix::operator+=(const Matrix &rhs)
    {
        assert((rhs.height() == height() && rhs.width() == width()));

        for (size_t i = 0; i < width() * height(); ++i)
            storage_[i] += rhs.storage_[i];

        return *this;
    }

    Matrix &Matrix::operator*=(const Matrix &rhs)
    {
        assert((rhs.height() == width()));

        Matrix result(rhs.width(), height());

        for (size_t x = 0; x < rhs.width(); ++x)
            for (size_t y = 0; y < height(); ++y)
                for (size_t s = 0; s < width(); ++s)
                    result(x, y) += (*this)(s, y) * rhs(x, s);

        this->storage_ = std::move(result.storage_);
        this->width_ = result.width_;
        this->height_ = result.height_;
        this->is_transposed = result.is_transposed;

        return (*this);
    }

    Matrix &Matrix::operator*=(const double rhs)
    {
        for (size_t i = 0; i < width() * height(); ++i)
            storage_[i] *= rhs;

        return (*this);
    }

    Matrix &Matrix::operator-=(const Matrix &rhs)
    {
        assert((rhs.height() == height() && rhs.width() == width()));

        for (size_t i = 0; i < width() * height(); ++i)
            storage_[i] -= rhs.storage_[i];

        return *this;
    }

    Matrix operator*(Matrix lhs, const Matrix &rhs)
    {
        lhs *= rhs;
        return lhs;
    }

    Matrix operator*(Matrix lhs, const double rhs)
    {
        lhs *= rhs;
        return lhs;
    }

    Matrix operator+(Matrix lhs, const Matrix &rhs)
    {
        lhs += rhs;
        return lhs;
    }

    Matrix operator-(Matrix lhs, const Matrix &rhs)
    {
        lhs -= rhs;
        return lhs;
    }

    std::ostream &operator<<(std::ostream &out, const Matrix &rhs)
    {
        for (size_t x = 0; x < rhs.height(); ++x)
        {
            out << "[\t";
            for (size_t y = 0; y < rhs.width(); ++y)
            {
                out << rhs(y, x) << "\t";
            }
            out << "]\n";
        }
        out << "\n";
        return out;
    }

}
