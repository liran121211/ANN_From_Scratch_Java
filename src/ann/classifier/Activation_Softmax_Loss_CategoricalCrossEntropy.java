package ann.classifier;

import java.io.Serial;
import java.io.Serializable;

public class Activation_Softmax_Loss_CategoricalCrossEntropy implements Activation, Serializable {
    //Softmax classifier - combined Softmax activation
    //and cross-entropy loss for faster backward step

    private Activation activation;
    private Loss loss;
    private Matrix output;
    private Matrix d_inputs;

    @Serial
    private static final long serialVersionUID = 6529685098267757603L;

    //Creates activation and loss function objects
    protected Activation_Softmax_Loss_CategoricalCrossEntropy() {
        this.activation = new Activation_SoftMax();
        this.loss = new Loss_CategoricalCrossEntropy();
    }

    //Forward pass
    public double forward(Matrix inputs, Matrix y_true) throws InvalidMatrixOperation, InvalidMatrixAxis, MatrixIndexesOutOfBounds, InvalidMatrixDimension {
        this.activation.forward(inputs); //Output layer's activation function
        this.output = this.activation.output(); //Set the output

        Matrix forward_matrix = this.loss.forward(this.output, y_true);
        return this.loss.calculate(forward_matrix); //Calculate and return loss value
    }

    //Backward pass
    public void backward(Matrix d_values, Matrix y_true) throws InvalidMatrixAxis, MatrixIndexesOutOfBounds, InvalidMatrixDimension {
        if (y_true.shape() == 2)
            y_true = y_true.argmax(1);

        this.d_inputs = new Matrix(d_values);

        for (int i = 0; i < d_inputs.getRows(); i++)
            d_inputs.setValue(i, (int) y_true.getValue(0, i), d_inputs.getValue(i, (int) y_true.getValue(0, i)) - 1);

        this.d_inputs.divide(d_values.getRows());
    }

    @Override
    public Matrix output() {
        return output;
    }

    @Override
    public Matrix d_inputs() {
        return d_inputs;
    }

    @Override
    public void forward(Matrix inputs) {
        //NOT IN USE
    }

    @Override
    public void backward(Matrix d_values) {
        //NOT IN USE
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
