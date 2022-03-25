package ann.classifier;

import java.io.Serial;
import java.io.Serializable;

public class Activation_Sigmoid_Loss_BinaryCrossEntropy  implements Activation, Serializable {
    private Activation activation;
    private Loss loss;
    private Matrix output;
    private Matrix d_inputs;

    @Serial
    private static final long serialVersionUID = 6529685098267757616L;

    //Creates activation and loss function objects
    protected Activation_Sigmoid_Loss_BinaryCrossEntropy() {
        this.activation = new Activation_Sigmoid();
        this.loss = new Loss_BinaryCrossEntropy();
    }

    @Override
    public void forward(Matrix inputs) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation, InvalidMatrixAxis {
        this.activation.forward(inputs); //Output layer's activation function
        this.output = this.activation.output(); //Set the output
    }

    @Override
    public void backward(Matrix d_values) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation {
        //NOT IN USE
    }

    @Override
    public double forward(Matrix inputs, Matrix y_true) throws InvalidMatrixOperation, InvalidMatrixAxis, MatrixIndexesOutOfBounds, InvalidMatrixDimension {
        this.activation.forward(inputs); //Output layer's activation function
        this.output = this.activation.output(); //Set the output

        Matrix forward_matrix = this.loss.forward(this.output, y_true);
        return this.loss.calculate(forward_matrix); //Calculate and return loss value
    }

    @Override
    public void backward(Matrix d_values, Matrix y_true) throws InvalidMatrixAxis, MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation {
        this.loss.backward(d_values,y_true);
        this.d_inputs = this.loss.d_inputs();
    }

    @Override
    public Matrix output() {
        return this.output;
    }

    @Override
    public Matrix d_inputs() {
        return this.d_inputs;
    }

    protected static double regularization_loss(LayerDense layer) throws MatrixIndexesOutOfBounds { //Regularization loss calculation
        double regularization_loss = 0.0; //0 by default

        //L1 regularization - weights
        //Calculate only when factor greater than 0
        if (layer.get_weight_regularizer_l1() > 0) {
            Matrix abs = new Matrix(layer.getWeights()).abs();
            regularization_loss += layer.get_weight_regularizer_l1() * abs.sum();
        }

        if (layer.get_weight_regularizer_l2() > 0) {
            Matrix square = new Matrix(layer.getWeights()).pow(2);
            regularization_loss += layer.get_weight_regularizer_l2() * square.sum();
        }

        if (layer.get_bias_regularizer_l1() > 0) {
            Matrix abs = new Matrix(layer.getBiases()).abs();
            regularization_loss += layer.get_bias_regularizer_l1() * abs.sum();
        }

        if (layer.get_bias_regularizer_l2() > 0) {
            Matrix square = new Matrix(layer.getBiases()).pow(2);
            regularization_loss += layer.get_bias_regularizer_l2() * square.sum();
        }

        return regularization_loss;
    }
}
