package ann.classifier;

import java.io.Serial;
import java.io.Serializable;

public class Optimizer_Adagrad implements Optimization, Serializable {
    //Adagrad - Adaptive Gradient
    private double learning_rate;
    private double current_learning_rate;
    private double decay;
    private double epsilon;
    private int iterations;

    @Serial
    private static final long serialVersionUID = 6529685098267757609L;


    //Initialize optimizer - set settings
    protected Optimizer_Adagrad(double learning_rate, double decay) {
        this.learning_rate = learning_rate;
        this.current_learning_rate = learning_rate;
        this.decay = decay;
        this.iterations = 0;
        this.epsilon = 1e-7;
    }

    //Initialize optimizer - set settings
    protected Optimizer_Adagrad(double decay) {
        this.learning_rate = 1.0;
        this.current_learning_rate = 1.0;
        this.decay = decay;
        this.iterations = 0;
        this.epsilon = 1e-7;
    }

    //Learning Rate of 1.0 is default for this optimizer
    protected Optimizer_Adagrad() {
        this.learning_rate = 1.0;
        this.current_learning_rate = 1.0;
        this.decay = 0.0;
        this.iterations = 0;
        this.epsilon = 1e-7;
    }

    @Override
    //Call once before any parameter updates
    public void pre_update_params() {
        //if decay rate is other than 0, will update current_learning_rate
        if (this.decay != 0.0)
            this.current_learning_rate = this.learning_rate * (1.0 / (1.0 + this.decay * this.iterations));
    }

    @Override
    //Update parameters
    public void update_params(LayerDense layer) throws InvalidMatrixOperation, InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        //If layer does not contain cache arrays,
        //create them filled with zeros
        if (layer.get_weight_cache() == null) {
            layer.set_weight_cache(new Matrix(layer.getWeights().getRows(), layer.getWeights().getColumns()));
            layer.set_bias_cache(new Matrix(layer.getBiases().getRows(), layer.getBiases().getColumns()));
        }

        // Update cache with squared current gradients
        Matrix get_d_weights_copy = new Matrix(layer.get_d_weights()); //Avoid modifying (d_weights)
        Matrix get_d_biases_copy = new Matrix(layer.get_d_biases()); //Avoid modifying (d_biases)
        layer.set_weight_cache(layer.get_weight_cache().add(get_d_weights_copy.pow(2)));
        layer.set_bias_cache(layer.get_bias_cache().add(get_d_biases_copy.pow(2)));

        //Vanilla SGD parameter update + normalization
        //with square rooted cache
        Matrix weight_cache_copy = new Matrix(layer.get_weight_cache()); //Avoid modifying (weight_cache)
        Matrix step1_w = weight_cache_copy.sqrt().add(this.epsilon);
        Matrix step2_w = layer.get_d_weights().multiply(-this.current_learning_rate);
        Matrix step3_w = step2_w.divide(step1_w);
        layer.setWeights(new Matrix(layer.getWeights()).add(step3_w));

        Matrix bias_cache_copy = new Matrix(layer.get_bias_cache());//Avoid modifying (bias_cache)
        Matrix step1_b = bias_cache_copy.sqrt().add(this.epsilon);
        Matrix step2_b = layer.get_d_biases().multiply(-this.current_learning_rate);
        Matrix step3_b = step2_b.divide(step1_b);
        layer.setBiases(new Matrix(layer.getBiases()).add(step3_b));
    }

    @Override
    //Call once after any parameter updates
    public void post_update_params() {
        this.iterations += 1;
    }

    @Override
    public double get_current_learning_rate() {
        return this.current_learning_rate;
    }

    @Override
    public String get_learning_rate() {
        return String.valueOf(this.learning_rate);
    }
}
