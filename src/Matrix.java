import java.util.Random;

public class Matrix {
    private final int rows;
    private final int columns;
    private final double[][] matrix;

    /**
     * Default Matrix Constructor.
     *
     * @param n_rows    (Matrix number of rows).
     * @param n_columns (Matrix number of columns).
     */
    protected Matrix(int n_rows, int n_columns) {
        this.rows = Math.abs(n_rows);
        this.columns = Math.abs(n_columns);
        this.matrix = new double[rows][columns];

        for (int i = 0; i < this.rows; i++) //Init (n) Lists.
            for (int j = 0; j < this.columns; j++) {
                this.matrix[i][j] = 0.0;
            }
    }

    /**
     * Matrix Constructor.
     *
     * @param n_rows    (Matrix number of rows).
     * @param n_columns (Matrix number of columns).
     * @param B         (2D double array)
     */
    protected Matrix(int n_rows, int n_columns, double[][] B) {
        this.rows = Math.abs(n_rows);
        this.columns = Math.abs(n_columns);
        this.matrix = new double[rows][columns];

        for (int i = 0; i < this.rows; i++) //Init (n) Lists.
            System.arraycopy(B[i], 0, this.matrix[i], 0, this.columns);
    }

    /**
     * Vector Constructor.
     *
     * @param n_rows    (Matrix number of rows).
     * @param n_columns (Matrix number of columns).
     * @param V         (1D double array)
     */
    protected Matrix(int n_rows, int n_columns, double[] V) {
        this.rows = Math.abs(n_rows);
        this.columns = Math.abs(n_columns);
        this.matrix = new double[rows][columns];

        for (int i = 0; i < this.rows; i++) { //Init (n) Lists.
            for (int j = 0; j < this.columns; j++) {
                if (this.rows > this.columns)
                    this.matrix[i][j] = V[i];
                else if (this.columns > this.rows)
                    this.matrix[i][j] = V[j];
                else
                    this.matrix[i][j] = V[i];
            }
        }
    }

    /**
     * Copy Constructor.
     *
     * @param B (Matrix object.)
     */
    protected Matrix(Matrix B) {
        this.rows = B.rows;
        this.columns = B.columns;
        this.matrix = new double[rows][columns];
        for (int i = 0; i < B.rows; i++) {
            for (int j = 0; j < B.columns; j++) {
                this.setValue(i, j, B.getValue(i, j));
            }
        }
    }

    /**
     * Get number of rows of Matrix.
     *
     * @return (int) number of rows.
     */
    public int getRows() {
        return rows;
    }

    /**
     * Get number of columns of Matrix.
     *
     * @return (int) number of columns.
     */
    public int getColumns() {
        return columns;
    }

    /**
     * Set value into specific cell in the Matrix.
     *
     * @param row    (Matrix row number).
     * @param column (Matrix column number).
     * @param value  (double) value to insert.
     */
    protected void setValue(int row, int column, double value) throws IndexOutOfBoundsException {
        if (!(row > this.rows) && !(column > this.columns))
            this.matrix[row][column] = value;
        throw new IndexOutOfBoundsException("row/column values are out of range.");

    }

    /**
     * Retrive value from specific cell.
     *
     * @param row    (Matrix row number).
     * @param column (Matrix column number).
     * @return (double) value of specific Matrix cell.
     */
    protected double getValue(int row, int column) throws IndexOutOfBoundsException {
        if (!(row > this.rows) && !(column > this.columns))
            return this.matrix[row][column];
        throw new IndexOutOfBoundsException("row/column values are out of range.");
    }

    /**
     * Multiply current Matrix with given Matrix (B).
     *
     * @param B (Matrix object).
     * @return new Matrix after multiplication.
     */
    protected Matrix product(Matrix B) throws IndexOutOfBoundsException {
        if (this.columns != B.rows)
            throw new IndexOutOfBoundsException(String.format("Cannot Multiply (%s,%s) By (%s,%s)", this.rows, this.columns, B.rows, B.columns));

        Matrix temp = new Matrix(this.rows, B.columns);
        double cell;
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < B.columns; j++) {
                cell = 0.0;
                for (int k = 0; k < this.columns; k++) {
                    cell += this.getValue(i, k) * B.getValue(k, j);
                }
                temp.setValue(i, j, cell);
            }
        }
        return temp;
    }

    /**
     * Multiply (num) value to each cell of the Matrix.
     *
     * @param num (double value)
     * @return Matrix with updated values.
     */
    protected Matrix product(double num) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                this.matrix[i][j] *= num;
        }
        return this;
    }

    /**
     * Add values from given Matrix (B) to the current Matrix.
     *
     * @param B (Matrix object).
     * @return Modified Matrix after addition.
     */
    protected Matrix add(Matrix B) {
        if (this.rows != B.rows || this.columns != B.columns)
            throw new IndexOutOfBoundsException(String.format("Matrices has different dimensions (%s,%s) By (%s,%s)", this.rows, this.columns, B.rows, B.columns));

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                this.matrix[i][j] += B.matrix[i][j];
        }
        return this;
    }

    /**
     * add (num) value to each cell of the Matrix.
     *
     * @param num (double value)
     * @return Matrix with updated values.
     */
    protected Matrix add(double num) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                this.matrix[i][j] += num;
        }
        return this;
    }

    /**
     * Transpose matrix from nxm to mxn.
     *
     * @return transposed Matrix.
     */
    protected Matrix transpose() {
        Matrix temp = new Matrix(this.columns, this.rows);
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.columns; j++)
                temp.setValue(i, j, this.getValue(j, i));
        }
        return temp;
    }

    /**
     * Prints Matrix in 2D shape
     *
     * @return (Prettified Matrix String)
     */
    @Override
    public String toString() {
        StringBuilder matrix_output = new StringBuilder();
        matrix_output.append("[");
        for (double[] arr : this.matrix) {
            for (Double number : arr) {
                matrix_output.append(number).append(", ");
            }
            matrix_output.append("\n");
        }
        matrix_output.replace(matrix_output.length() - 3, matrix_output.length() - 1, "]");
        return matrix_output.toString();
    }

    /**
     * Compare values of current Matrix to the given one.
     *
     * @param obj (Matrix object)
     * @return True if equals else false.
     */
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Matrix) { // Validate matrices size
            if (!(((Matrix) obj).columns == this.columns))
                return false;
            if (!(((Matrix) obj).rows == this.rows))
                return false;
        }

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++) {
                if (this.matrix[i][j] != ((Matrix) obj).matrix[i][j])
                    return false;
            }
        }
        return true;
    }

    /**
     * @param n_rows    (number of rows in matrix)
     * @param n_columns (number of columns in matrix)
     * @return (nxm Matrix filled with normal_distribution values)
     */
    protected static Matrix random(int n_rows, int n_columns) {
        Matrix temp = new Matrix(Math.abs(n_rows), Math.abs(n_columns));
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.columns; j++)
                temp.matrix[i][j] = new Random().nextGaussian();
        }
        return temp;
    }
}

