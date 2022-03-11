package ann.classifier;

public interface Activation {
    void forward(Matrix inputs) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation, InvalidMatrixAxis;

    void backward(Matrix d_values) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation;

    double forward(Matrix inputs, Matrix y_true) throws InvalidMatrixOperation, InvalidMatrixAxis, MatrixIndexesOutOfBounds, InvalidMatrixDimension; //ann.classifier.Activation_Softmax_Loss_CategoricalCrossEntropy

    void backward(Matrix d_values, Matrix y_true) throws InvalidMatrixAxis, MatrixIndexesOutOfBounds, InvalidMatrixDimension; //ann.classifier.Activation_Softmax_Loss_CategoricalCrossEntropy

    Matrix output();

    Matrix d_inputs();
}
