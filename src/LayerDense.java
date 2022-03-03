public class LayerDense {
    private Matrix weights;
    private Matrix biases;

    protected LayerDense(int n_inputs, int n_neurons){
        weights = Matrix.random(n_inputs,n_neurons);
    }

    protected void forward(Matrix inputs){

    }
}
