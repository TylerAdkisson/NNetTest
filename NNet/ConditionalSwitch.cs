using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNet
{
    class ConditionalSwitch
    {
        public static void MainVoid(string[] args)
        {
            Condition1();
        }

        static void Condition1()
        {
            Layer inputLayer = new Layer(1);
            Layer hiddenLayer1 = new Layer(2, Neuron.ActivationType.LeakyReLU);
            Layer outputLayer = new Layer(1, Neuron.ActivationType.ReLU);

            // Connect layers with random weights
            inputLayer.ConnectForwardFully(hiddenLayer1);
            hiddenLayer1.ConnectForwardFully(outputLayer);

            hiddenLayer1.LearningRate = 0.3;
            outputLayer.LearningRate = 0.3;

            hiddenLayer1.LearningMomentum = 0.95;
            outputLayer.LearningMomentum = 0.95;

            hiddenLayer1.WeightDecay = 0.000;
            outputLayer.WeightDecay = 0.000;

            // Change the mode of some of the hidden layer's activation functions
            //hiddenLayer1.Neurons[0].SetNeuronType(Neuron.ActivationType.Tanh);

            int trainingDataCount = 300;
            int totalDataCount = 1000;
            double[][] inputSet = new double[totalDataCount][];
            double[][] trainInputSet = new double[trainingDataCount][];
            double[][] targetSet = new double[totalDataCount][];
            double[][] trainTargetSet = new double[trainingDataCount][];
            Random r = new Random(11);

            // Create training data set (simple if-then-like formula)
            for (int i = 0; i < inputSet.Length; i++)
            {
                double energyPercent = r.NextDouble();
                double targetRodPercent = 0.95;
                if (energyPercent < 0.3)
                {
                    targetRodPercent = 0.0;
                }

                inputSet[i] = new[] { energyPercent };// * 2.0 - 1.0 };
                targetSet[i] = new[] { targetRodPercent };// * 2.0 - 1.0 };
            }

            // Select random items from full data set for training
            Random dataSetRandom = new Random(1 /*seed*/);
            HashSet<int> selectedIndexes = new HashSet<int>();
            for (int i = 0; i < trainingDataCount; i++)
            {
                int index = 0;
                do
                {
                    index = dataSetRandom.Next(0, totalDataCount);
                } while (selectedIndexes.Contains(index));
                selectedIndexes.Add(index);

                trainInputSet[i] = inputSet[index];
                trainTargetSet[i] = targetSet[index];
            }

            bool isTrained = false;
            for (int iter = 0; iter < int.MaxValue && !isTrained; iter++)
            {
                // Jumble input data order as to not get into a loop we can't train
                Program.Shuffle2(trainInputSet, trainTargetSet);

                //if (iter == 200)
                //{
                //    hiddenLayer1.LearningRate = 0.1;
                //    outputLayer.LearningRate = 0.1;
                //}

                // Train over input set
                for (int i = 0; i < trainingDataCount; i++)
                {
                    // Network activation

                    // Supply input data
                    inputLayer.SetValues(trainInputSet[i]);

                    // Activate layers
                    hiddenLayer1.Activate();
                    outputLayer.Activate();


                    // Network training

                    // Network error computation
                    outputLayer.ComputeError(trainTargetSet[i]);
                    hiddenLayer1.ComputeError();

                    // Update weights
                    hiddenLayer1.UpdateWeights();
                    outputLayer.UpdateWeights();
                }

                // Validate against test set
                Console.WriteLine("Iteration {0,9:N0}, Validation test: ", iter + 1);
                bool showData = false;
                double error = 0.0;
                for (int i = 0; i < inputSet.Length; i++)
                {
                    // Supply input data
                    inputLayer.SetValues(inputSet[i]);

                    // Activate layers
                    hiddenLayer1.Activate();
                    outputLayer.Activate();

                    // Input values
                    for (int k = 0; k < inputSet[i].Length; k++)
                    {
                        double inputValue = inputSet[i][k];

                        // For functions that input in rads, we want to display in degrees
                        //inputValue = (inputValue * Math.PI) + Math.PI;
                        //inputValue *= (180 / Math.PI);

                        if (showData)
                            Console.Write("{0,8:F4} ", inputValue);
                    }

                    if (showData)
                        Console.Write("= ");

                    for (int k = 0; k < targetSet[i].Length; k++)
                    {
                        if (showData)
                            Console.Write("[{0,10:F6} Target: {1,8:F4} Err: {2,10:F6}]",
                                outputLayer.Neurons[k].Value,
                                targetSet[i][k],
                                targetSet[i][k] - outputLayer.Neurons[k].Value);

                        double localError = targetSet[i][k] - outputLayer.Neurons[k].Value;
                        error += localError * localError;
                    }
                    if (showData)
                        Console.WriteLine();
                }

                error /= inputSet.Length;
                Console.WriteLine("MSE: {0:F6}", error);
                //output.Write(BitConverter.GetBytes((float)error), 0, 4);

                if (error <= 0.0012)
                    isTrained = true;
            }
        }
    }
}
