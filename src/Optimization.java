public interface Optimization {
    void pre_update_params();

    void update_params(LayerDense layer) throws InvalidMatrixOperation, InvalidMatrixDimension, MatrixIndexesOutOfBounds;

    void post_update_params();

    double get_current_learning_rate();

}
