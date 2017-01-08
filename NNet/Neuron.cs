using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNet
{
    class Neuron
    {
        private static Random _weightRandom;
        private static readonly double _weightRange = 1.0;//0.000125944584;//0.00125944584382871536523929471033;//2.0;

        public enum ActivationType
        {
            Tanh,
            ReLU,
            LeakyReLU
        }

        public Func<double, double> ActivationFunction;
        public Func<double, double> ErrorDerivFunction;

        public Synapse[] Inputs;
        public Synapse[] Outputs;
        public double Value;
        public double ErrorDelta;


        public Neuron()
            : this(ActivationType.Tanh)
        {
        }

        public Neuron(ActivationType type)
        {
            Inputs = new Synapse[0];
            Outputs = new Synapse[0];
            Value = 0;

            SetNeuronType(type);
        }


        public void SetNeuronType(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.Tanh:
                    ActivationFunction = ActivationTanH;
                    ErrorDerivFunction = DerivTanH;
                    break;
                case ActivationType.ReLU:
                    ActivationFunction = ActivationReLU;
                    ErrorDerivFunction = DerivReLU;
                    break;
                case ActivationType.LeakyReLU:
                    ActivationFunction = ActivationLeakyReLU;
                    ErrorDerivFunction = DerivLeakyReLU;
                    break;
                default:
                    break;
            }
        }

        public static double GetRandomWeight()
        {
            if (_weightRandom == null)
                _weightRandom = new Random(1);

            return (_weightRandom.NextDouble() * (_weightRange * 2)) - _weightRange;
        }

        /// <summary>
        /// Take input values and updates our value
        /// </summary>
        public void Activate()
        {
            // Consider values of all of our inputs
            double sum = 0.0;
            for (int i = 0; i < Inputs.Length; i++)
            {
                // Sum all input values, multiplied by the link weights
                //double adj = Sigmoid(Inputs[i].Input.Value * Inputs[i].Weight);
                //sum += adj;
                sum += Inputs[i].Input.Value * Inputs[i].Weight;
            }

            //while (Math.Abs(sum) >= 1.0)
            //    sum /= 10.0;

            //sum /= 10.0;
            //sum /= (Inputs.Length / 100.0);

            // Differentiate sum with sigmoid function (could be tanh)
            Value = ActivationFunction(sum);

            //int intValue = (int)(Value * 127);
            //Value = intValue / 127.0;
        }

        public override string ToString()
        {
            return string.Format("Value: {0:F5}, Inputs: {1:N0}, Outputs: {2:N0}",
                Value,
                Inputs.Length,
                Outputs.Length);
        }

        /// <summary>
        /// Hyperbolic tangent activation function
        /// </summary>
        /// <remarks>
        /// Tanh begins to saturate towards -1.0..+1.0 as input reaches -2.0..+2.0.
        /// One potential issue with sigmoid-shaped activation functions is a vanishing
        /// gradient (derivative) on the reverse pass using backpropagation learning
        /// </remarks>
        /// <param name="value">The input value</param>
        /// <returns>The result of the activation function. In the range of -1.0..+1.0</returns>
        private static double ActivationTanH(double value)
        {
            return Math.Tanh(value);
        }

        /// <summary>
        /// Derivative of the tanh activation function
        /// </summary>
        /// <remarks>
        /// This uses the derivative in the form of 1.0 - x^2, instead of 1.0 - tanh(x)^2
        /// </remarks>
        /// <param name="value">The input value</param>
        /// <returns>The result of the derivative function</returns>
        private static double DerivTanH(double value)
        {
            return 1.0 - (value * value);
        }

        /// <summary>
        /// The linear rectifier (ReLU) activation function
        /// </summary>
        /// <remarks>
        /// This returns a linear result from 0.0..+1.0 if input is between 0.0..+Inf.
        /// While returning 0.0 for any negative value
        /// </remarks>
        /// <param name="value">The input value</param>
        /// <returns>The result of the activation function. In the range of 0.0..+1.0</returns>
        private static double ActivationReLU(double value)
        {
            return Math.Min(Math.Max(0, value), 1.0);
        }

        /// <summary>
        /// The linear rectifier (ReLU) derivative function
        /// </summary>
        /// <remarks>
        /// Returns a either 0.0 if the value is negative, otherwise 1.0
        /// The result at a value of 0 is 1.0
        /// </remarks>
        /// <param name="value">The input value</param>
        /// <returns>The result of the derivative function</returns>
        private static double DerivReLU(double value)
        {
            return value >= 0.0 ? 1.0 : 0.0;
        }

        /// <summary>
        /// The leaky variant of the linear rectifier (ReLU) activation function
        /// </summary>
        /// <remarks>
        /// This returns a linear result in the range 0.0..+1.0 if input is between 0.0..+Inf.
        /// While returning a linear result of -0.001x for any negative value
        /// </remarks>
        /// <param name="value">The input value</param>
        /// <returns>The result of the activation function. In the range of ~-0.001..+1.0</returns>
        private static double ActivationLeakyReLU(double value)
        {
            return Math.Max(-1.0, Math.Min(Math.Max(0.01 * value, value), 1.0));
        }

        /// <summary>
        /// The leaky variant of the linear rectifier (ReLU) derivative function
        /// </summary>
        /// <remarks>
        /// Returns a either 0.001 if the value is negative, otherwise 1.0
        /// </remarks>
        /// <param name="value">The input value</param>
        /// <returns>The result of the derivative function</returns>
        private static double DerivLeakyReLU(double value)
        {
            return value >= 0.0 ? 1.0 : 0.01;
        }
    };

}
