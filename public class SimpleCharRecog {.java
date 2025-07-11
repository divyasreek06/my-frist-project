import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class SimpleCNNMnist {

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int outputNum = 10; // 10 digits
        int rngSeed = 123;  // random seed for reproducibility
        int numEpochs = 3;

        // Load MNIST data
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        // Build CNN model configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(1) // grayscale images have one channel
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(500)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(org.deeplearning4j.nn.conf.inputs.InputType.convolutionalFlat(28,28,1))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Show score every 10 iterations
        model.setListeners(new ScoreIterationListener(10));

        // Train model
        System.out.println("Training model...");
        for(int i = 0; i < numEpochs; i++) {
            model.fit(mnistTrain);
        }

        // Evaluate on test data
        System.out.println("Evaluating model...");
        org.deeplearning4j.eval.Evaluation eval = model.evaluate(mnistTest);
        System.out.println(eval.stats());
    }
}