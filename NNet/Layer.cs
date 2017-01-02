using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNet
{
    class Layer
    {
        public Neuron[] Neurons;
        public Neuron Bias;
        public double LearningRate = 0.1;
        public double LearningMomentum = 0.0;
        public double WeightDecay = 0.0;
        public Neuron.ActivationType NeuronType;

        public Layer(int neuronCount)
            : this(neuronCount, Neuron.ActivationType.Tanh)
        {
        }

        public Layer(int neuronCount, Neuron.ActivationType neuronType)
        {
            Neurons = new Neuron[neuronCount];
            NeuronType = neuronType;

            InitializeNeurons();
        }


        // Connects this layer as the input to another layer
        public void ConnectForwardFully(Layer nextLayer)
        {
            ConnectForwardFully(nextLayer, false);
        }

        public void ConnectForwardFully(Layer nextLayer, bool clearConnections)
        {
            if (nextLayer == null)
                throw new ArgumentNullException("nextLayer");

            // Adjust each neuron in the next layer's input connection lists to have room for our new connections
            //   Expand by the number of neurons in our layer plus one for the bias node
            for (int i = 0; i < nextLayer.Neurons.Length; i++)
            {
                if (clearConnections)
                    nextLayer.Neurons[i].Inputs = new Synapse[Neurons.Length + 1];
                else
                    Array.Resize(ref nextLayer.Neurons[i].Inputs, nextLayer.Neurons[i].Inputs.Length + Neurons.Length + 1);
            }

            for (int i = 0; i < Neurons.Length; i++)
            {
                // Resize source neuron's output connection list to have room for our new connections
                int beginIndex = Neurons[i].Outputs.Length;
                if (clearConnections)
                {
                    Neurons[i].Outputs = new Synapse[nextLayer.Neurons.Length];
                    beginIndex = 0;
                }
                else
                {
                    Array.Resize(ref Neurons[i].Outputs, Neurons[i].Outputs.Length + nextLayer.Neurons.Length);
                }

                //Synapse[] outputList = new Synapse[Neurons[i].Outputs.Length + nextLayer.Neurons.Length];
                //Array.Copy(Neurons[i].Outputs, 0, outputList, 0, Neurons[i].Outputs.Length);
                //Neurons[i].Outputs = outputList;


                for (int k = 0; k < nextLayer.Neurons.Length; k++)
                {
                    Synapse link = new Synapse(Neurons[i], nextLayer.Neurons[k], Neuron.GetRandomWeight());

                    Neurons[i].Outputs[beginIndex + k] = link;
                    nextLayer.Neurons[k].Inputs[nextLayer.Neurons[k].Inputs.Length - 1 - (Neurons.Length - i)] = link;
                }
            }

            // Connect bias node
            Array.Resize(ref Bias.Outputs, nextLayer.Neurons.Length);
            for (int i = 0; i < nextLayer.Neurons.Length; i++)
            {
                nextLayer.Neurons[i].Inputs[nextLayer.Neurons[i].Inputs.Length - 1] = Bias.Outputs[i]
                    = new Synapse(Bias, nextLayer.Neurons[i], Neuron.GetRandomWeight());
            }
        }

        public void Activate()
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].Activate();
            }
        }

        public void SetValues(double[] values)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].Value = values[i];
            }
        }

        public void ComputeError(Neuron[] targetValues)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                double error = targetValues[i].Value - Neurons[i].Value;
                Neurons[i].ErrorDelta = Neurons[i].ErrorDerivFunction(Neurons[i].Value) * error;
            }
        }

        public void ComputeError(double[] targetValues)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                double error = targetValues[i] - Neurons[i].Value;
                Neurons[i].ErrorDelta = Neurons[i].ErrorDerivFunction(Neurons[i].Value) * error;
            }
        }

        public void ComputeError2(double[] targetValues)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                double error = targetValues[i] - Neurons[i].Value;
                Neurons[i].ErrorDelta = error;
            }
        }

        public void ComputeError()
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                double errorSum = 0.0;
                for (int k = 0; k < Neurons[i].Outputs.Length; k++)
                {
                    Synapse link = Neurons[i].Outputs[k];
                    errorSum += link.Output.ErrorDelta * link.Weight;
                }

                Neurons[i].ErrorDelta = Neurons[i].ErrorDerivFunction(Neurons[i].Value) * errorSum;
            }
        }

        public void ComputeError2(int[] outputIndexes)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                double errorSum = 0.0;
                for (int k = 0; k < outputIndexes.Length; k++)
                {
                    int outputSyn = outputIndexes[k];
                    Synapse link = Neurons[i].Outputs[outputSyn];
                    errorSum += link.Output.ErrorDelta * link.Weight;
                }

                Neurons[i].ErrorDelta = errorSum;
            }
        }

        public void UpdateWeights()
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                // Update each input link's weight
                for (int k = 0; k < Neurons[i].Inputs.Length; k++)
                {
                    Synapse link = Neurons[i].Inputs[k];

                    // Add to the input link weight a percentage (the learning rate) of the
                    //   derived error value (computed previously) multiplied by the input value
                    // We also add a percentage (momentum value) of the last change, to help overcome
                    //   settling at local minima
                    double weightChange = Neurons[i].ErrorDelta * link.Input.Value;

                    double decayVal = 2 * WeightDecay * link.Weight;

                    link.Weight += (LearningRate * weightChange) + (LearningMomentum * link.LastChange) - decayVal;
                    //link.Weight -= link.Weight * 0.0002;
                    //link.Weight *= 0.9998;
                    link.LastChange = weightChange;

                    //int intWeight = (int)(link.Weight * 32767);
                    //link.Weight = intWeight / 32767.0;
                }
            }
        }

        public void UpdateWeight(int synapseIndex)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                // Update each input link's weight
                Synapse link = Neurons[i].Inputs[synapseIndex];

                // Add to the input link weight a percentage (the learning rate) of the
                //   derived error value (computed previously) multiplied by the input value
                // We also add a percentage (momentum value) of the last change, to help overcome
                //   settling at local minima
                double weightChange = Neurons[i].ErrorDelta * link.Input.Value;

                link.Weight += (LearningRate * weightChange) + (LearningMomentum * link.LastChange);
                //link.Weight -= link.Weight * 0.0002;
                //link.Weight *= 0.9998;
                link.LastChange = weightChange;

                //int intWeight = (int)(link.Weight * 32767);
                //link.Weight = intWeight / 32767.0;
            }
        }

        public void UpdateNeuronWeight(int neuronIndex)
        {
            // Update each input link's weight
            for (int k = 0; k < Neurons[neuronIndex].Inputs.Length; k++)
            {
                Synapse link = Neurons[neuronIndex].Inputs[k];

                // Add to the input link weight a percentage (the learning rate) of the
                //   derived error value (computed previously) multiplied by the input value
                // We also add a percentage (momentum value) of the last change, to help overcome
                //   settling at local minima
                double weightChange = Neurons[neuronIndex].ErrorDelta * link.Input.Value;

                link.Weight += (LearningRate * weightChange) + (LearningMomentum * link.LastChange);
                //link.Weight -= link.Weight * 0.0002;
                //link.Weight *= 0.9998;
                link.LastChange = weightChange;

                //int intWeight = (int)(link.Weight * 32767);
                //link.Weight = intWeight / 32767.0;
            }
        }


        private void InitializeNeurons()
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(NeuronType);
            }

            Bias = new Neuron(NeuronType);
            Bias.Value = 1.0;
        }
    };
}
