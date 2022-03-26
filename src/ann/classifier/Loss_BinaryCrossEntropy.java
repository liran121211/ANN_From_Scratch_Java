package ann.classifier;

import java.io.Serial;
import java.io.Serializable;

public class Loss_BinaryCrossEntropy implements Loss, Serializable {
    private Matrix d_inputs;

    @Serial
    private static final long serialVersionUID = 6529685098267757615L;

    @Override
    public Matrix forward(Matrix y_pred, Matrix y_true) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds, InvalidMatrixOperation, InvalidMatrixAxis {
        Matrix y_pred_clipped = Matrix.clip(y_pred, 1e-7, 1 - 1e-7);

        Matrix step1 = new Matrix(y_true).transpose();
        Matrix step2 = new Matrix(y_pred_clipped).log();
        Matrix step3 = step1.multiply(step2);

        Matrix step4 = new Matrix(y_pred_clipped).multiply(-1).subtract(-1);
        Matrix step5 = step4.log();
        Matrix step6 = new Matrix(y_true).multiply(-1).subtract(-1).transpose();
        Matrix step7 = step6.multiply(step5);
        Matrix sample_losses = (step3.add(step7)).multiply(-1);

        return sample_losses.mean(1);
    }

    @Override
    public void backward(Matrix d_values, Matrix y_true) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation, InvalidMatrixAxis {
        int samples = d_values.getRows(); //Number of samples

        // Number of outputs in every sample
        // We'll use the first sample to count them
        int outputs = d_values.getColumns();

        // Clip data to prevent division by 0
        // Clip both sides to not drag mean towards any value
        Matrix clipped_d_values = Matrix.clip(d_values, 1e-7, 1 - 1e-7);

        //Calculate gradient
        Matrix step1 = new Matrix(y_true).multiply(-1).subtract(-1).transpose();
        Matrix step2 = new Matrix(clipped_d_values).multiply(-1).subtract(-1);
        Matrix step3 = step1.divide(step2);
        Matrix step4 = new Matrix(y_true).transpose().divide(clipped_d_values);
        Matrix step5 = step4.subtract(step3).multiply(-1);
        this.d_inputs = step5.divide(outputs);

        //Normalize gradient
        this.d_inputs = this.d_inputs.divide(samples);
    }

    @Override
    public double calculate(Matrix output) {
        return output.mean();
    }

    @Override
    public Matrix d_inputs() {
        return this.d_inputs;
    }

    @Override
    public double regularization_loss(LayerDense layer) throws MatrixIndexesOutOfBounds {
        double regularization_loss = 0.0; //0 by default

        //L1 regularization - weights
        //Calculate only when factor greater than 0
        if (layer.get_weight_regularizer_l1() > 0) {
            Matrix abs = new Matrix(layer.getWeights().abs());
            regularization_loss += layer.get_weight_regularizer_l1() * abs.sum();
        }

        if (layer.get_weight_regularizer_l2() > 0) {
            Matrix square = new Matrix(layer.getWeights().pow(2));
            regularization_loss += layer.get_weight_regularizer_l2() * square.sum();
        }

        if (layer.get_bias_regularizer_l1() > 0) {
            Matrix abs = new Matrix(layer.getBiases().abs());
            regularization_loss += layer.get_bias_regularizer_l1() * abs.sum();
        }

        if (layer.get_bias_regularizer_l2() > 0) {
            Matrix square = new Matrix(layer.getBiases().pow(2));
            regularization_loss += layer.get_bias_regularizer_l2() * square.sum();
        }

        return regularization_loss;
    }


}
