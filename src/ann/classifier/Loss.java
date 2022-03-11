package ann.classifier;

public interface Loss {
    //Calculates the data and regularization losses
    //Given model output and ground truth values

    Matrix forward(Matrix output, Matrix y_true) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds, InvalidMatrixOperation;

    void backward(Matrix output, Matrix y_true) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation;

    Matrix d_inputs();

    double calculate(Matrix output);
}
