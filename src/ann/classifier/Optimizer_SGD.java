package ann.classifier;

public class Optimizer_SGD implements Optimization {
    private double learning_rate;
    private double current_learning_rate;
    private double decay;
    private double momentum;
    private int iterations;

    //Initialize optimizer - set settings
    protected Optimizer_SGD(double learning_rate, double decay, double momentum) {
        this.learning_rate = learning_rate;
        this.current_learning_rate = learning_rate;
        this.decay = decay;
        this.iterations = 0;
        this.momentum = momentum;
    }

    //Initialize optimizer - set settings
    protected Optimizer_SGD(double decay, double momentum) {
        this.learning_rate = 1.0;
        this.current_learning_rate = 1.0;
        this.decay = decay;
        this.iterations = 0;
        this.momentum = momentum;
    }

    //Initialize optimizer - set settings
    protected Optimizer_SGD(double decay) {
        this.learning_rate = 1.0;
        this.current_learning_rate = 1.0;
        this.decay = decay;
        this.iterations = 0;
        this.momentum = 0.0;
    }

    //Learning Rate of 1.0 is default for this optimizer
    protected Optimizer_SGD() {
        this.learning_rate = 1.0;
        this.current_learning_rate = 1.0;
        this.decay = 0.0;
        this.iterations = 0;
        this.momentum = 0.0;
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
    public void update_params(LayerDense layer) throws InvalidMatrixOperation, InvalidMatrixDimension {
        Matrix weight_updates;
        Matrix bias_updates;

        if (this.momentum != 0.0) { //If we use momentum

            if (layer.get_weight_momentums() == null) {
                //If layer does not contain momentum arrays, create them
                //filled with zeros
                layer.set_weight_momentums(new Matrix(layer.getWeights().getRows(), layer.getWeights().getColumns()));

                //If there is no momentum array for weights
                //The array doesn't exist for biases yet either.
                layer.set_bias_momentums(new Matrix(layer.getBiases().getRows(), layer.getBiases().getColumns()));
            }

            // Build weight updates with momentum - take previous
            // updates multiplied by retain factor and update with
            // current gradients
            weight_updates = layer.get_weight_momentums().multiply(this.momentum).subtract(layer.get_d_weights().multiply(this.current_learning_rate));
            layer.set_weight_momentums(weight_updates);

            //Build bias updates
            bias_updates = layer.get_bias_momentums().multiply(this.momentum).subtract(layer.get_d_biases().multiply(this.current_learning_rate));
            layer.set_bias_momentums(bias_updates);
        } else {
            //Vanilla SGD updates (as before momentum update)
            weight_updates = layer.get_d_weights().multiply(-this.current_learning_rate);
            bias_updates = layer.get_d_biases().multiply(-this.current_learning_rate);
        }
        //Update weights and biases using either
        //vanilla or momentum updates
        layer.setWeights(layer.getWeights().add(weight_updates));
        layer.setBiases(layer.getBiases().add(bias_updates));
    }

    @Override
    //Call once after any parameter updates
    public void post_update_params() {
        this.iterations += 1;
    }

    @Override
    public double get_current_learning_rate() {
        return current_learning_rate;
    }

    @Override
    public String get_learning_rate() {
        return String.valueOf(this.learning_rate);
    }
}
