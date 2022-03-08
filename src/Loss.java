public interface Loss {
    public Matrix forward(Matrix output, Matrix y_true) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds, InvalidMatrixOperation;

    public void backward(Matrix output, Matrix y_true) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation;

    public Matrix d_inputs();

    public double calculate(Matrix output);
}
