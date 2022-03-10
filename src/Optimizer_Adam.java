public class Optimizer_Adam implements Optimization {
    //Adam - Adaptive Momentum
    private double learning_rate;
    private double current_learning_rate;
    private double decay;
    private double epsilon;
    private int iterations;
    private double beta_1;
    private double beta_2;

    //Initialize optimizer - set settings
    protected Optimizer_Adam(double learning_rate, double decay, double beta_1, double beta_2) {
        this.learning_rate = learning_rate;
        this.current_learning_rate = learning_rate;
        this.decay = decay;
        this.iterations = 0;
        this.epsilon = 1e-7;
        this.beta_1 = beta_1;
        this.beta_2 = beta_2;
    }

    //Initialize optimizer - set settings
    protected Optimizer_Adam(double decay, double beta_1, double beta_2) {
        this.learning_rate = 0.001;
        this.current_learning_rate = 0.001;
        this.decay = decay;
        this.iterations = 0;
        this.epsilon = 1e-7;
        this.beta_1 = beta_1;
        this.beta_2 = beta_2;
    }

    protected Optimizer_Adam(double learning_rate, double decay) {
        this.learning_rate = learning_rate;
        this.current_learning_rate = learning_rate;
        this.decay = decay;
        this.iterations = 0;
        this.epsilon = 1e-7;
        this.beta_1 = 0.9;
        this.beta_2 = 0.999;
    }

    //Learning Rate of 0.001 is default for this optimizer
    protected Optimizer_Adam() {
        this.learning_rate = 0.001;
        this.current_learning_rate = 0.001;
        this.decay = 0.0;
        this.iterations = 0;
        this.epsilon = 1e-7;
        this.beta_1 = 0.9;
        this.beta_2 = 0.999;
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
            layer.set_weight_momentums(new Matrix(layer.getWeights().getRows(), layer.getWeights().getColumns()));
            layer.set_weight_cache(new Matrix(layer.getWeights().getRows(), layer.getWeights().getColumns()));

            layer.set_bias_momentums(new Matrix(layer.getBiases().getRows(), layer.getBiases().getColumns()));
            layer.set_bias_cache(new Matrix(layer.getBiases().getRows(), layer.getBiases().getColumns()));

        }

        //TODO NOT WORKING

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
}


