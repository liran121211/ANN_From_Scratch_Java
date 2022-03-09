public class Optimizer_SGD {
    private double learning_rate;

    protected Optimizer_SGD(double learning_rate) {
        //TODO
    }

    protected Optimizer_SGD() {
        this.learning_rate = 1.0;
    }

    protected void update_params(LayerDense layer) throws InvalidMatrixOperation, MatrixIndexesOutOfBounds {
        layer.set_d_weights(layer.get_d_weights().multiply(-this.learning_rate));
        layer.set_d_biases(layer.get_d_biases().multiply(-this.learning_rate));

        layer.setWeights(layer.getWeights().add(layer.get_d_weights()));
        layer.setBiases(layer.getBiases().add(layer.get_d_biases()));
    }
}
