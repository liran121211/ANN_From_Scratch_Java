package ann.classifier;

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
            layer.set_weight_cache(new Matrix(layer.getWeights().getRows(), layer.getWeights().getColumns()));
            layer.set_bias_cache(new Matrix(layer.getBiases().getRows(), layer.getBiases().getColumns()));
            layer.set_weight_momentums(new Matrix(layer.getWeights().getRows(), layer.getWeights().getColumns()));
            layer.set_bias_momentums(new Matrix(layer.getBiases().getRows(), layer.getBiases().getColumns()));
        }

        // Update momentum with current gradients
        Matrix get_d_weights_copy = new Matrix(layer.get_d_weights()); //Avoid modifying (d_weights)
        Matrix get_d_biases_copy = new Matrix(layer.get_d_biases()); //Avoid modifying (d_biases)
        layer.set_weight_momentums(layer.get_weight_momentums().multiply(this.beta_1).add(get_d_weights_copy.multiply(1-this.beta_1)));
        layer.set_bias_momentums(layer.get_bias_momentums().multiply(this.beta_1).add(get_d_biases_copy.multiply(1-this.beta_1)));

        // Get corrected momentum
        // this.iteration is 0 at first pass
        // and we need to start with 1 here
        Matrix weight_momentums_corrected = new Matrix(layer.get_weight_momentums()).divide(( 1 - Math.pow(this.beta_1,(this.iterations + 1 ))));
        Matrix bias_momentums_corrected = new Matrix(layer.get_bias_momentums()).divide(( 1 - Math.pow(this.beta_1,(this.iterations + 1 ))));

        //Vanilla SGD parameter update + normalization
        //with square rooted cache
        Matrix weight_cache_copy =new Matrix( layer.get_d_weights()); //Avoid modifying (weight_cache)
        Matrix step1_w = weight_cache_copy.pow(2).multiply( 1 - this.beta_2);
        Matrix step2_w = layer.get_weight_cache().multiply(this.beta_2);
        Matrix step3_w = step2_w.add(step1_w);
        layer.set_weight_cache(step3_w);

        Matrix bias_cache_copy =new Matrix(layer.get_d_biases());//Avoid modifying (bias_cache)
        Matrix step1_b = bias_cache_copy.pow(2).multiply( 1 - this.beta_2);
        Matrix step2_b = layer.get_bias_cache().multiply(this.beta_2);
        Matrix step3_b = step2_b.add(step1_b);
        layer.set_bias_cache(step3_b);

        //Get corrected cache
        Matrix weight_cache_corrected = new Matrix(layer.get_weight_cache()).divide(( 1 - Math.pow(this.beta_2,(this.iterations + 1 ))));
        Matrix bias_cache_corrected = new Matrix(layer.get_bias_cache()).divide(( 1 - Math.pow(this.beta_2,(this.iterations + 1 ))));


        //Vanilla SGD parameter update + normalization
        //with square rooted cache
        Matrix step1_w__ = weight_cache_corrected.sqrt().add(this.epsilon);
        Matrix step2_w__ = weight_momentums_corrected.multiply(-this.current_learning_rate);
        Matrix step3_w__ = step2_w__.divide(step1_w__);
        layer.setWeights(layer.getWeights().add(step3_w__));

        Matrix step1_b__ = bias_cache_corrected.sqrt().add(this.epsilon);
        Matrix step2_b__ = bias_momentums_corrected.multiply(-this.current_learning_rate);
        Matrix step3_b__ = step2_b__.divide(step1_b__);
        layer.setBiases(layer.getBiases().add(step3_b__));

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


