package com.afconsult.math;

import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

public class Matrix {

    private final int rows;
    private final int cols;

    private double[][] m;

    private Matrix(int rows, int cols) {
        if (rows < 1 || cols < 1) {
            throw new IllegalArgumentException("Illegal matrix dimensions");
        }

        this.rows = rows;
        this.cols = cols;

        m = new double[rows][];
        for (int r = 0; r < rows; r++) {
            m[r] = new double[cols];
        }
    }

    public Matrix(int rows, int cols, BiFunction<Integer, Integer, Double> initFunction) {
        if (rows < 1 || cols < 1) {
            throw new IllegalArgumentException("Illegal matrix dimensions");
        }

        this.rows = rows;
        this.cols = cols;

        m = new double[rows][];
        for (int r = 0; r < rows; r++) {
            m[r] = new double[cols];

            for (int c = 0; c < cols; c++) {
                m[r][c] = initFunction.apply(r, c);
            }
        }
    }

    public static Matrix Identity(int dim) {
        return new Matrix(dim, dim, (r, c) -> Objects.equals(r, c) ? 1.0 : 0.0);
    }

    public static Matrix Zeros(int rows, int cols) {
        return new Matrix(rows, cols);
    }

    public double get(int row, int col) {
        return m[row][col];
    }

    public int getRows() { return rows; }
    public int getCols() { return cols; }

    public Matrix scale(double s) {

        Matrix v = new Matrix(rows, cols);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                v.m[r][c] = s * this.m[r][c];
            }
        }

        return v;
    }

    public Matrix apply(Function<Double, Double> function) {
        return new Matrix(this.rows, this.cols, (r, c) -> function.apply(this.m[r][c]));
    }

    public Matrix add(Matrix o) {
        if (rows != o.rows || cols != o.cols) {
            throw new IllegalArgumentException("Matrix dimensions do not agree");
        }

        Matrix v = new Matrix(rows, cols);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                v.m[r][c] = this.m[r][c] + o.m[r][c];
            }
        }

        return v;
    }

    public Matrix subtract(Matrix o) {
        if (rows != o.rows || cols != o.cols) {
            throw new IllegalArgumentException("Matrix dimensions do not agree");
        }

        Matrix v = new Matrix(rows, cols);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                v.m[r][c] = this.m[r][c] - o.m[r][c];
            }
        }

        return v;
    }

    public Matrix multiply(Matrix o) {
        if (cols != o.rows) {
            throw new IllegalArgumentException("Matrix dimensions do not agree");
        }

        Matrix v = new Matrix(rows, o.cols);

        for (int r = 0; r < rows; r++) {
            Matrix row = this.row(r);

            for (int c = 0; c < o.cols; c++) {
                Matrix col = o.column(c);

                v.m[r][c] = dotProd(row, col);
            }
        }

        return v;
    }

    public Matrix transpose() {
        return new Matrix(cols, rows, (r, c) -> this.m[c][r]);
    }

    private Matrix row(int r) {
        if (r < 0 || r > rows - 1) {
            throw new IllegalArgumentException("Row exceeds matrix dimensions");
        }

        Matrix v = new Matrix(1, cols);
        System.arraycopy(this.m[r], 0, v.m[0], 0, cols);
        return v;
    }

    private Matrix column(int c) {
        if (c < 0 || c > cols - 1) {
            throw new IllegalArgumentException("Columns exceeds matrix dimensions");
        }

        Matrix v = new Matrix(rows, 1);

        for (int r = 0; r < rows; r++) {
            v.m[r][0] = this.m[r][c];
        }

        return v;
    }

    private static double dotProd(Matrix a, Matrix b) {
        if (a.rows != 1 || b.cols != 1 || a.cols != b.rows) {
            throw new IllegalArgumentException("Dimensions do not agree");
        }

        double prod = 0.0;

        for (int x = 0; x < a.cols; x++) {
            prod += a.m[0][x] * b.m[x][0];
        }

        return prod;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof  Matrix)) {
            return false;
        }

        Matrix o = (Matrix)obj;

        if (rows != o.rows || cols != o.cols) {
            return false;
        }

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (Double.compare(this.m[r][c], o.m[r][c]) != 0) {
                    return false;
                }
            }
        }

        return true;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        for (int r = 0; r < rows; r++) {
            builder.append(r == 0 ? "[" : " ");

            for (int c = 0; c < cols; c++) {
                builder.append(c > 0 ? String.format(", %-2.2f", m[r][c]) : String.format("%-2.2f", m[r][c]));
            }

            builder.append(r == rows - 1 ? "]" : "\n");
        }

        return builder.toString();
    }
}
