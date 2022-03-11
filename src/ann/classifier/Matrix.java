package ann.classifier;
import java.util.Random;

public class Matrix {
    private final int rows;
    private final int columns;
    private final double[][] matrix;

    /**
     * Default ann.classifier.Matrix Constructor.
     *
     * @param n_rows    (ann.classifier.Matrix number of rows).
     * @param n_columns (ann.classifier.Matrix number of columns).
     */
    protected Matrix(int n_rows, int n_columns) throws InvalidMatrixDimension {
        if (n_rows <= 0 || n_columns <= 0)
            throw new InvalidMatrixDimension(n_rows, n_columns);

        this.rows = n_rows;
        this.columns = n_columns;
        this.matrix = new double[rows][columns];

        for (int i = 0; i < this.rows; i++) //Init (n) Lists.
            for (int j = 0; j < this.columns; j++) {
                this.matrix[i][j] = 0.0;
            }
    }

    /**
     * ann.classifier.Matrix Constructor.
     *
     * @param n_rows    (ann.classifier.Matrix number of rows).
     * @param n_columns (ann.classifier.Matrix number of columns).
     * @param B         (2D double array)
     */
    protected Matrix(int n_rows, int n_columns, double[][] B) throws InvalidMatrixDimension {
        if (n_rows <= 0 || n_columns <= 0)
            throw new InvalidMatrixDimension(n_rows, n_columns);

        if (B.length != n_rows || B[0].length != n_columns)
            throw new InvalidMatrixDimension(n_rows, n_columns, B);

        this.rows = n_rows;
        this.columns = n_columns;
        this.matrix = new double[rows][columns];
        for (int i = 0; i < this.rows; i++) //Init (n) Lists.
            System.arraycopy(B[i], 0, this.matrix[i], 0, this.columns);
    }


    /**
     * Copy Constructor.
     *
     * @param B (ann.classifier.Matrix object.)
     */
    protected Matrix(Matrix B) throws MatrixIndexesOutOfBounds {
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
     * Get number of rows of ann.classifier.Matrix.
     *
     * @return (int) number of rows.
     */
    public int getRows() {
        return rows;
    }

    /**
     * Get number of columns of ann.classifier.Matrix.
     *
     * @return (int) number of columns.
     */
    public int getColumns() {
        return columns;
    }

    /**
     * Set value into specific cell in the ann.classifier.Matrix.
     *
     * @param row    (ann.classifier.Matrix row number).
     * @param column (ann.classifier.Matrix column number).
     * @param value  (double) value to insert.
     */
    protected void setValue(int row, int column, double value) throws MatrixIndexesOutOfBounds {
        if (!(row > this.rows) && !(column > this.columns))
            this.matrix[row][column] = value;
        else
            throw new MatrixIndexesOutOfBounds(row, column);
    }

    /**
     * Retrieve value from specific cell.
     *
     * @param row    (ann.classifier.Matrix row number).
     * @param column (ann.classifier.Matrix column number).
     * @return (double) value of specific ann.classifier.Matrix cell.
     */
    protected double getValue(int row, int column) throws MatrixIndexesOutOfBounds {
        if (!(row > this.rows) && !(column > this.columns))
            return this.matrix[row][column];
        throw new MatrixIndexesOutOfBounds(row, column);
    }

    /**
     * Multiply current ann.classifier.Matrix with given ann.classifier.Matrix (B).
     *
     * @param B (ann.classifier.Matrix object).
     * @return new ann.classifier.Matrix after multiplication.
     */
    protected Matrix dot(Matrix B) throws InvalidMatrixOperation, InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        if (vectorsMultiplication(this, B) != -1)// return multiplication value of 2 vectors.
            return new Matrix(1, 1, new double[][]{{vectorsMultiplication(this, B)}});

        if (this.columns != B.rows)
            throw new InvalidMatrixOperation(this, B, "product");

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
     * Multiply (num) value to each cell of the ann.classifier.Matrix.
     *
     * @param num (double value)
     * @return ann.classifier.Matrix with updated values.
     */
    protected Matrix multiply(double num) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                this.matrix[i][j] *= num;
        }
        return this;
    }

    /**
     * Multiply each value to each cell of the Matrices.
     *
     * @param B (ann.classifier.Matrix object)
     * @return ann.classifier.Matrix with updated values.
     */
    protected Matrix multiply(Matrix B) throws InvalidMatrixOperation {
        if (B.rows == this.rows && B.columns == 1) { // if 1D array
            for (int i = 0; i < this.rows; i++) {
                for (int j = 0; j < this.columns; j++)
                    this.matrix[i][j] *= B.matrix[i][0];
            }
            return this;

        } else if (B.columns == this.columns && B.rows == 1) { // if 1D array
            for (int i = 0; i < this.rows; i++) {
                for (int j = 0; j < this.columns; j++)
                    this.matrix[i][j] *= B.matrix[0][j];
            }
            return this;
        }

        if (this.rows != B.rows || this.columns != B.columns)
            throw new InvalidMatrixOperation(this, B, "multiply");

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                this.matrix[i][j] *= B.matrix[i][j];
        }
        return this;
    }

    /**
     * Add values from given ann.classifier.Matrix (B) to the current ann.classifier.Matrix.
     *
     * @param B (ann.classifier.Matrix object).
     * @return Modified ann.classifier.Matrix after addition.
     */
    protected Matrix add(Matrix B) throws InvalidMatrixOperation {
        if (B.rows == this.rows && B.columns == 1) { // if 1D array
            for (int i = 0; i < this.rows; i++) {
                for (int j = 0; j < this.columns; j++)
                    this.matrix[i][j] += B.matrix[i][0];
            }
            return this;

        } else if (B.columns == this.columns && B.rows == 1) { // if 1D array
            for (int i = 0; i < this.rows; i++) {
                for (int j = 0; j < this.columns; j++)
                    this.matrix[i][j] += B.matrix[0][j];
            }
            return this;
        }

        if (this.rows != B.rows || this.columns != B.columns)
            throw new InvalidMatrixOperation(this, B, "add");

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                this.matrix[i][j] += B.matrix[i][j];
        }
        return this;
    }

    /**
     * Add (num) value to each cell of the ann.classifier.Matrix.
     *
     * @param num (double value)
     * @return ann.classifier.Matrix with updated values.
     */
    protected Matrix add(double num) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                this.matrix[i][j] += num;
        }
        return this;
    }

    /**
     * Subtract (num) value to each cell of the ann.classifier.Matrix.
     *
     * @param num (double value)
     * @return ann.classifier.Matrix with updated values.
     */
    protected Matrix subtract(double num) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                this.matrix[i][j] -= num;
        }
        return this;
    }

    /**
     * Subtract values from given ann.classifier.Matrix (B) to the current ann.classifier.Matrix.
     *
     * @param B (ann.classifier.Matrix object).
     * @return Modified ann.classifier.Matrix after subtraction.
     */
    protected Matrix subtract(Matrix B) throws InvalidMatrixOperation {
        if (B.rows == this.rows && B.columns == 1) { // if 1D array
            for (int i = 0; i < this.rows; i++) {
                for (int j = 0; j < this.columns; j++)
                    this.matrix[i][j] -= B.matrix[i][0];
            }
            return this;
        }
        if (this.rows != B.rows || this.columns != B.columns)
            throw new InvalidMatrixOperation(this, B, "subtract");

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                this.matrix[i][j] -= B.matrix[i][j];
        }
        return this;
    }

    /**
     * Divide (num) value to each cell of the ann.classifier.Matrix.
     *
     * @param num (double value)
     * @return ann.classifier.Matrix with updated values.
     */
    protected Matrix divide(double num) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                this.matrix[i][j] /= num;
        }
        return this;
    }

    /**
     * Divide values from given ann.classifier.Matrix (B) to the current ann.classifier.Matrix.
     *
     * @param B (ann.classifier.Matrix object).
     * @return Modified ann.classifier.Matrix after Division.
     */
    protected Matrix divide(Matrix B) throws InvalidMatrixOperation {
        if (B.rows == this.rows && B.columns == 1) { // if 1D array
            for (int i = 0; i < this.rows; i++) {
                for (int j = 0; j < this.columns; j++) {
                    if (Double.isInfinite(this.matrix[i][j] / B.matrix[i][0]))
                        this.matrix[i][j] = 1e-7;
                    else
                        this.matrix[i][j] /= B.matrix[i][0];
                }
            }
            return this;
        }
        if (this.rows != B.rows || this.columns != B.columns)
            throw new InvalidMatrixOperation(this, B, "divide");

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++) {
                if (Double.isInfinite(this.matrix[i][j] / B.matrix[i][j]))
                    this.matrix[i][j] = 1e-7;
                else
                    this.matrix[i][j] /= B.matrix[i][j];
            }
        }
        return this;
    }

    /**
     * Transofrm ann.classifier.Matrix into other shape.
     *
     * @param n_rows    (number of rows).
     * @param n_columns (number of columns).
     * @return Reshaped ann.classifier.Matrix.
     */
    protected Matrix reshape(int n_rows, int n_columns) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        Matrix flatten_matrix = Matrix.flat(this);
        int index = 0;

        if (Math.abs(n_rows * n_columns) == Math.abs(this.rows * this.columns)) {
            Matrix temp = new Matrix(n_rows, n_columns);
            for (int i = 0; i < temp.rows; i++) {
                for (int j = 0; j < temp.columns; j++) {
                    temp.setValue(i, j, flatten_matrix.getValue(0, index));
                    index++;
                }
            }
            return temp;

        } else if (n_rows == -1 && Math.abs(n_columns) <= Math.abs(this.rows * this.columns)) { // number of rows is arbitrary.
            int rows = Math.abs((this.rows * this.columns) / n_columns);
            Matrix temp = new Matrix(rows, n_columns);
            for (int i = 0; i < temp.rows; i++) {
                for (int j = 0; j < temp.columns; j++) {
                    temp.setValue(i, j, flatten_matrix.getValue(0, index));
                    index++;
                }
            }
            return temp;

        } else if (n_columns == -1 && Math.abs(n_rows) <= Math.abs(this.rows * this.columns)) { // number of columns is arbitrary.
            int columns = Math.abs((this.rows * this.columns) / n_rows);
            Matrix temp = new Matrix(n_rows, columns);
            for (int i = 0; i < temp.rows; i++) {
                for (int j = 0; j < temp.columns; j++) {
                    temp.setValue(i, j, flatten_matrix.getValue(0, index));
                    index++;
                }
            }
            return temp;
        }
        throw new InvalidMatrixDimension(n_rows, n_columns);
    }

    /**
     * Raise all values in ann.classifier.Matrix by (E) exponent.
     *
     * @return Modified ann.classifier.Matrix after subtraction.
     */
    protected Matrix exp() {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                this.matrix[i][j] = Math.exp(this.matrix[i][j]);
        }
        return this;
    }

    /**
     * Transpose matrix from nxm to mxn.
     *
     * @return transposed ann.classifier.Matrix.
     */
    protected Matrix transpose() throws InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        Matrix temp = new Matrix(this.columns, this.rows);
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.columns; j++)
                temp.setValue(i, j, this.getValue(j, i));
        }
        return temp;
    }

    /**
     * Get specific row of the ann.classifier.Matrix.
     *
     * @param row (row number).
     * @return Vector of specific row.
     */
    protected Matrix getRow(int row) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        Matrix temp = new Matrix(1, this.columns);
        for (int i = 0; i < this.columns; i++) {
            temp.setValue(0, i, this.getValue(row, i));
        }
        return temp;
    }

    /**
     * Get specific column of the ann.classifier.Matrix.
     *
     * @param column (column number).
     * @return Vector of specific column.
     */
    protected Matrix getColumn(int column) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        Matrix temp = new Matrix(this.rows, 1);
        for (int i = 0; i < this.rows; i++) {
            temp.setValue(i, 0, this.getValue(i, column));
        }
        return temp;
    }

    /**
     * Get shape of ann.classifier.Matrix
     *
     * @return 1 if ann.classifier.Matrix is Vector else 2.
     */
    protected int shape() {
        if (this.rows == 1 && this.columns >= 1)
            return 1;
        else if (this.columns == 1 && this.rows >= 1)
            return 1;
        else
            return 2;
    }

    /**
     * Return max value in the entire matrix.
     *
     * @return (double) max value.
     */
    protected double max() {
        double max_value = 0.0;
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                if (this.matrix[i][j] > max_value)
                    max_value = this.matrix[i][j];
        }
        return max_value;
    }

    /**
     * Returns the indices of the maximum values of each column/row.
     *
     * @param axis (0 or 1).
     * @return (ann.classifier.Matrix object) vector contains indices of the maximum value of each column/row.
     */
    protected Matrix argmax(int axis) throws InvalidMatrixAxis, InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        if (this.rows == 1 || this.columns == 1) // if vector return (this).
            return this;

        int max_index;
        double max_val;

        if (axis == 0) {
            Matrix temp = new Matrix(1, this.columns);

            for (int i = 0; i < this.getColumns(); i++) {
                max_val = this.getValue(0, i); //reset max val to cell (0,i)
                max_index = 0; //reset max index to cell (0,i)
                for (int j = 1; j < this.getRows(); j++)
                    if (this.getValue(j, i) > max_val) {
                        max_val = this.getValue(j, i);
                        max_index = j;
                    }
                temp.setValue(0, i, max_index);
            }
            return temp;
        }
        if (axis == 1) {
            Matrix temp = new Matrix(this.rows, 1);

            for (int i = 0; i < this.getRows(); i++) {
                max_val = this.getValue(i, 0); //reset max val to cell (i,0)
                max_index = 0; //reset max index to cell (i,0)
                for (int j = 1; j < this.getColumns(); j++)
                    if (this.getValue(i, j) > max_val) {
                        max_val = this.getValue(i, j);
                        max_index = j;
                    }
                temp.setValue(i, 0, max_index);
            }
            return temp;
        } else
            throw new InvalidMatrixAxis(axis);
    }


    /**
     * Return maximum value in each column/row of matrix.
     *
     * @param axis (0 or 1).
     * @return (ann.classifier.Matrix object) vector with max values of each column/row.
     */
    protected Matrix max(int axis) throws InvalidMatrixAxis, InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        if (this.rows == 1)
            return new Matrix(1, 1, new double[][]{{this.max()}});

        if (axis == 0) {
            Matrix temp = new Matrix(1, this.columns);
            for (int i = 0; i < this.getColumns(); i++) {
                double max_val = this.getValue(0, i);
                for (int j = 1; j < this.getRows(); j++)
                    if (this.getValue(j, i) > max_val)
                        max_val = this.getValue(j, i);
                temp.setValue(0, i, max_val);
            }
            return temp;
        }
        if (axis == 1) {
            Matrix temp = new Matrix(this.rows, 1);
            for (int i = 0; i < this.getRows(); i++) {
                double max_val = this.getValue(i, 0);
                for (int j = 1; j < this.getColumns(); j++)
                    if (this.getValue(i, j) > max_val)
                        max_val = this.getValue(i, j);
                temp.setValue(i, 0, max_val);
            }
            return temp;

        } else
            throw new InvalidMatrixAxis(axis);
    }

    /**
     * square root (num) value to each cell of the ann.classifier.Matrix.
     *
     * @return ann.classifier.Matrix with squared root values.
     */
    protected Matrix sqrt() {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                this.matrix[i][j] = Math.sqrt(this.matrix[i][j]);
        }
        return this;
    }

    /**
     * pow value by (num) value to each cell of the ann.classifier.Matrix.
     * @param num (pow number).
     * @return ann.classifier.Matrix with pow values.
     */
    protected Matrix pow(double num) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++)
                this.matrix[i][j] = Math.pow(this.matrix[i][j], num);
        }
        return this;
    }

    /**
     * Return summed values in each column/row of matrix.
     *
     * @param axis (0 or 1).
     * @return (ann.classifier.Matrix object) vector with summed values of each column/row.
     */
    protected Matrix sum(int axis) throws InvalidMatrixAxis, InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        if (axis == 0) {
            Matrix temp = new Matrix(1, this.columns);
            for (int i = 0; i < this.getColumns(); i++) {
                double sum = 0.0;
                for (int j = 0; j < this.getRows(); j++)
                    sum += this.getValue(j, i);
                temp.setValue(0, i, sum);
            }
            return temp;
        }
        if (axis == 1) {
            Matrix temp = new Matrix(this.rows, 1);
            for (int i = 0; i < this.getRows(); i++) {
                double sum = 0.0;
                for (int j = 0; j < this.getColumns(); j++)
                    sum += this.getValue(i, j);
                temp.setValue(i, 0, sum);
            }
            return temp;

        } else
            throw new InvalidMatrixAxis(axis);
    }

    /**
     * Return the log (e) value of each cell of the ann.classifier.Matrix.
     *
     * @return ann.classifier.Matrix with updated values.
     */
    protected Matrix log() {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (this.matrix[i][j] <= 0.0) // Avoid log(0).
                    this.matrix[i][j] = 1 - 1e-7;
                else
                    this.matrix[i][j] = Math.log(this.matrix[i][j]);
            }
        }
        return this;
    }

    /**
     * Return the mean value of the ann.classifier.Matrix.
     *
     * @return (double value).
     */
    protected double mean() {
        double mean = 0.0;
        for (int i = 0; i < this.rows; i++)
            for (int j = 0; j < columns; j++)
                mean += this.matrix[i][j];
        try {
            return mean / (this.rows * this.columns);
        } catch (ArithmeticException e) {
            return 0.0;
        }
    }

    /**
     * Prints ann.classifier.Matrix in 2D shape
     *
     * @return (Prettified ann.classifier.Matrix String)
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
     * Compare values of current ann.classifier.Matrix to the given one.
     *
     * @param obj (ann.classifier.Matrix object)
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
                assert obj instanceof Matrix;
                if (this.matrix[i][j] != ((Matrix) obj).matrix[i][j])
                    return false;
            }
        }
        return true;
    }

    /**
     * @param n_rows    (number of rows in matrix)
     * @param n_columns (number of columns in matrix)
     * @return (nxm ann.classifier.Matrix filled with normal_distribution values)
     */
    protected static Matrix random(int n_rows, int n_columns) throws InvalidMatrixDimension {
        Matrix temp = new Matrix(Math.abs(n_rows), Math.abs(n_columns));
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.columns; j++)
                temp.matrix[i][j] = new Random().nextGaussian();
        }
        return temp;
    }

    /**
     * Given an interval, values outside the interval are clipped to the interval edges.
     * For example, if an interval of [0, 1] is specified,
     * values smaller than 0 become 0, and values larger than 1 become 1.
     *
     * @param B   (ann.classifier.Matrix object).
     * @param min (minimal double value).
     * @param max (maximal double value).
     * @return ann.classifier.Matrix with clipped values.
     */
    public static Matrix clip(Matrix B, double min, double max) throws InvalidMatrixDimension {
        Matrix temp = new Matrix(B.rows, B.columns);
        for (int i = 0; i < B.rows; i++) {
            for (int j = 0; j < B.columns; j++) {
                if (B.matrix[i][j] < min)
                    temp.matrix[i][j] = min;
                else if (B.matrix[i][j] > max)
                    temp.matrix[i][j] = max;
                else
                    temp.matrix[i][j] = B.matrix[i][j];
            }
        }
        return temp;
    }

    /**
     * Change every value of ann.classifier.Matrix to given (value) if it is less 0.
     *
     * @param B     (ann.classifier.Matrix object).
     * @param value (double number).
     * @return ann.classifier.Matrix modified with positive numbers.
     */
    protected static Matrix maximum(Matrix B, double value) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        Matrix temp = new Matrix(B.rows, B.columns);

        for (int i = 0; i < B.rows; i++) {
            for (int j = 0; j < B.columns; j++) {
                temp.setValue(i, j, Math.max(0, value));
            }
        }
        return temp;
    }

    /**
     * Compare each values of ann.classifier.Matrix (A) to ann.classifier.Matrix (B)
     * Set value 1 if equal else 0.
     *
     * @param A (ann.classifier.Matrix object).
     * @param B (ann.classifier.Matrix object).
     * @return Boolean ann.classifier.Matrix of 1/0.
     */
    public static Matrix bitwiseCompare(Matrix A, Matrix B) throws InvalidMatrixDimension {
        if (A.rows != B.rows || A.columns != B.columns)
            throw new InvalidMatrixDimension(A, B);

        Matrix temp = new Matrix(A.rows, A.columns);
        for (int i = 0; i < A.rows; i++) {
            for (int j = 0; j < A.columns; j++) {
                if (A.matrix[i][j] == B.matrix[i][j])
                    temp.matrix[i][j] = 1.0;
                else
                    temp.matrix[i][j] = 0.0;
            }
        }

        return temp;
    }

    /**
     * Create Diagonal (ones) ann.classifier.Matrix.
     *
     * @param n_rows (number of rows).
     * @return Diagnal ann.classifier.Matrix.
     */
    public static Matrix eye(int n_rows) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        if (n_rows <= 0)
            throw new InvalidMatrixDimension(n_rows, n_rows);

        Matrix temp = new Matrix(n_rows, n_rows);
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.columns; j++) {
                if (i == j)
                    temp.setValue(i, j, 1.0);
            }
        }
        return temp;
    }

    /**
     * Create ann.classifier.Matrix that contains OneHot Vectos each row by given Vector (V).
     *
     * @param V (ann.classifier.Matrix object).
     * @return ann.classifier.Matrix of OneHot Vectors.
     */
    public static Matrix oneHotVector(Matrix V) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        Matrix temp = new Matrix(V.columns, V.columns); //(m, m) ann.classifier.Matrix

        if (V.getRows() == 1)
            for (int i = 0; i < V.getColumns(); i++)
                temp.setValue(i, (int) (V.getValue(0, i)), 1.0);
        else
            throw new InvalidMatrixDimension(V, V);

        return temp;
    }

    /**
     * Create a 2D ann.classifier.Matrix with the array_like input as a diagonal to the new output array.
     *
     * @param B (ann.classifier.Matrix object).
     * @return Diagonal ann.classifier.Matrix with the given Vector values.
     */
    public static Matrix diagflat(Matrix B) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        if (B.rows == 1) {
            Matrix temp = new Matrix(B.columns, B.columns);
            for (int i = 0; i < temp.rows; i++) {
                for (int j = 0; j < temp.columns; j++) {
                    if (i == j)
                        temp.setValue(i, j, B.getValue(0, i));
                }
            }
            return temp;

        } else if (B.columns == 1) {
            Matrix temp = new Matrix(B.rows, B.rows);
            B = B.transpose();
            for (int i = 0; i < temp.rows; i++) {
                for (int j = 0; j < temp.columns; j++) {
                    if (i == j)
                        temp.setValue(i, j, B.getValue(0, i));
                }
            }
            return temp;
        } else
            throw new InvalidMatrixDimension(B.rows, B.columns);
    }

    /**
     * Create Diagonal (ones) ann.classifier.Matrix.
     *
     * @param n_rows    (number of rows).
     * @param n_columns (number of columns)
     * @return Diagnal ann.classifier.Matrix.
     */
    public static Matrix eye(int n_rows, int n_columns) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        if (n_rows <= 0)
            throw new InvalidMatrixDimension(n_rows, n_columns);

        Matrix temp = new Matrix(n_rows, n_columns);
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.columns; j++) {
                if (i == j)
                    temp.setValue(i, j, 1.0);
            }
        }
        return temp;
    }

    /**
     * Flat ann.classifier.Matrix from 2D to 1D array.
     *
     * @param B (ann.classifier.Matrix object).
     * @return flatted ann.classifier.Matrix.
     */
    protected static Matrix flat(Matrix B) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        Matrix temp = new Matrix(1, B.rows * B.columns);
        int index = 0;
        for (int i = 0; i < B.rows; i++) {
            for (int j = 0; j < B.columns; j++) {
                temp.setValue(0, index, B.getValue(i, j));
                index++;
            }
        }
        return temp;
    }

    /**
     * Returns ann.classifier.Matrix of zeros with the same shape and type as a given ann.classifier.Matrix.
     *
     * @param B (ann.classifier.Matrix object).
     * @return Zero Values ann.classifier.Matrix.
     */
    protected static Matrix zeros_like(Matrix B) throws InvalidMatrixDimension {
        return new Matrix(B.rows, B.columns);
    }

    /**
     * Multiply 2 Vectors.
     *
     * @param V1 (ann.classifier.Matrix object).
     * @param V2 (ann.classifier.Matrix object).
     * @return ann.classifier.Matrix of (V1xV2) size.
     */
    private static double vectorsMultiplication(Matrix V1, Matrix V2) {
        double sum_values = 0.0;
        if (V1.columns == 1 && V2.columns == 1 && V1.rows == V2.rows) {
            for (int i = 0; i < V1.rows; i++) {
                sum_values += V1.matrix[i][0] * V2.matrix[i][0];
            }
        } else if (V1.rows == 1 && V2.rows == 1 && V1.columns == V2.columns) {
            for (int j = 0; j < V1.columns; j++) {
                sum_values += V1.matrix[0][j] * V2.matrix[0][j];
            }
        } else
            return -1;
        return sum_values;
    }

    /**
     * Sum all values in specific row of the ann.classifier.Matrix.
     *
     * @param V   (ann.classifier.Matrix object).
     * @param row (row number).
     * @return (double) values of sum.
     */
    protected static double sumRow(Matrix V, int row) {
        double sum_values = 0.0;
        for (int i = 0; i < V.columns; i++)
            sum_values += V.matrix[row][i];
        return sum_values;
    }

    /**
     * Sum all values in specific column of the ann.classifier.Matrix.
     *
     * @param V      (ann.classifier.Matrix object).
     * @param column (row number).
     * @return (double) values of sum.
     */
    protected static double sumColumn(Matrix V, int column) {
        double sum_values = 0.0;
        for (int i = 0; i < V.rows; i++)
            sum_values += V.matrix[i][column];
        return sum_values;
    }
}

