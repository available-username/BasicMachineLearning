package se.phi.math;

import junit.framework.TestCase;

import static org.junit.Assert.*;

public class MatrixTest extends TestCase {

    public void testAdd() {
        int rows = 3;
        int cols = 4;
        Matrix a = new Matrix(rows, cols, (r, c) -> (double)r * cols + c);
        Matrix b = new Matrix(rows, cols, (r, c) -> (double)r * cols + c);

        Matrix v = a.add(b);

        assertEquals(rows, v.getRows());
        assertEquals(cols, v.getCols());

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(0, Double.compare(a.get(r, c) + b.get(r, c), v.get(r, c)));
            }
        }
    }

    public void testSubtract() {
        int rows = 3;
        int cols = 4;
        Matrix a = new Matrix(rows, cols, (r, c) -> (double)r * cols + c);
        Matrix b = new Matrix(rows, cols, (r, c) -> (double)r * cols + c);

        Matrix v = a.subtract(b);

        assertEquals(rows, v.getRows());
        assertEquals(cols, v.getCols());

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(0, Double.compare(a.get(r, c) - b.get(r, c), v.get(r, c)));
            }
        }
    }

    public void testEquals() {
        int rows = 3;
        int cols = 4;
        Matrix a = new Matrix(rows, cols, (r, c) -> (double)r * cols + c);
        Matrix b = new Matrix(rows, cols, (r, c) -> (double)r * cols + c);

        assertTrue(a.equals(b));
        assertFalse(a.equals(null));
        assertFalse(a.equals(1));
        assertFalse(a.equals(Matrix.Zeros(rows, cols)));
        assertFalse(a.equals(Matrix.Identity(rows)));
        assertTrue(a.equals(a));
    }

    public void testScale() {

    }

    public void testDotProduct() {
        Matrix a = new Matrix(1, 4, (r, c) -> (double)c);
        Matrix b = new Matrix(4, 1, (r, c) -> (double)r);

        Matrix v = a.multiply(b);

        assertEquals(1, v.getRows());
        assertEquals(1, v.getCols());
        assertEquals(0, Double.compare(1.0*1.0 + 2.0*2.0 + 3.0*3.0, v.get(0, 0)));
    }

    public void testMultiply() {
        int rows = 3;
        int cols = 4;

        Matrix b = new Matrix(cols, rows, (r, c) -> (double)c * rows + r);
        Matrix a = new Matrix(rows, cols, (r, c) -> (double)r * cols + c);

        // B * A
        Matrix v = b.multiply(a);

        assertEquals(b.getRows(), v.getRows());
        assertEquals(a.getCols(), v.getCols());

        int dim = 10;
        Matrix square = new Matrix(dim, dim, (r, c) -> (double)r * dim + c);
        Matrix identity = Matrix.Identity(dim);
        Matrix squareIdentity = identity.multiply(square);
        Matrix identitySquare = square.multiply(identity);

        assertEquals(square, squareIdentity);
        assertEquals(squareIdentity, square);
        assertEquals(squareIdentity, identitySquare);
    }

    public void testTranspose() {
        int rows = 3;
        int cols = 7;

        Matrix a = new Matrix(rows, cols, (r, c) -> (double)r * cols + c);
        Matrix at = a.transpose();
        Matrix att = at.transpose();

        assertNotEquals(a, at);
        assertEquals(a, att);

        Matrix identity = Matrix.Identity(cols);
        assertEquals(identity, identity.transpose());
    }
}
