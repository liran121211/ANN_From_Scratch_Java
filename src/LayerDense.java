public class LayerDense {
    private Matrix weights;
    private Matrix biases;

    protected LayerDense(int n_inputs, int n_neurons){
        this.weights = Matrix.random(n_inputs,n_neurons).product(0.01);
        this.biases = new Matrix(1,n_neurons);
    }

    protected void forward(Matrix inputs){

    }
}
