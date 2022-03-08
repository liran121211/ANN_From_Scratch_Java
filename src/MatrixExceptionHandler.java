public class MatrixExceptionHandler extends Exception {

    public MatrixExceptionHandler() {
        super();
    }

    public MatrixExceptionHandler(String errorMessage) {
        super(errorMessage);
    }

}

class InvalidMatrixDimension extends MatrixExceptionHandler {
    public InvalidMatrixDimension(int n_rows, int n_columns, double[][] B) {
        super(String.format("the provided dimensions (%s, %s) does not match the given matrix (%s, %s)", n_rows, n_columns, B.length, B[0].length));
    }

    public InvalidMatrixDimension(int n_rows, int n_columns) {
        super(String.format("one or more of the given dimensions are invalid (%s, %s) ", n_rows, n_columns));
    }

    public InvalidMatrixDimension(Matrix A, Matrix B) {
        super(String.format("the provided dimensions (%s, %s) does not match the given matrix (%s, %s)", A.getRows(), A.getColumns(), B.getRows(), B.getColumns()));
    }
}

class MatrixIndexesOutOfBounds extends MatrixExceptionHandler {
    public MatrixIndexesOutOfBounds(int row, int column) {
        super(String.format("the provided dimensions (%s, %s) are out of bounds", row, column));
    }
}

class InvalidMatrixOperation extends MatrixExceptionHandler {
    public InvalidMatrixOperation(Matrix A, Matrix B, String operand) {
        super(String.format("cannot %s (%s,%s) by (%s,%s)", operand, A.getRows(), A.getColumns(), B.getRows(), B.getColumns()));
    }
}

class InvalidMatrixAxis extends MatrixExceptionHandler {
    public InvalidMatrixAxis(int axis) {
        super(String.format("axis %s is invalid, only 0/1 axis are valid", axis));
    }
}