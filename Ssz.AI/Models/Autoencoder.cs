using System;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models
{
    public class Autoencoder
    {
        #region construction and destruction

        public Autoencoder(int inputSize, int bottleneckSize, int maxActiveUnits)
        {
            _inputSize = inputSize;
            _bottleneckSize = bottleneckSize;
            _maxActiveUnits = maxActiveUnits;

            // Инициализация весов и смещений
            _weightsEncoder = CreateRandomTensor(inputSize, bottleneckSize);
            _biasesEncoder = new DenseTensor<float>(new[] { bottleneckSize });
            _weightsDecoder = CreateRandomTensor(bottleneckSize, inputSize);
            _biasesDecoder = new DenseTensor<float>(new[] { inputSize });

            // Буферы для временных данных
            _encoderOutput = new DenseTensor<float>(new[] { bottleneckSize });
            _decoderOutput = new DenseTensor<float>(new[] { inputSize });

            _gradientBuffer = new DenseTensor<float>(new[] { inputSize });            
            _gradientBuffer2 = new DenseTensor<float>(new[] { inputSize });
            _gradientBuffer3 = new DenseTensor<float>(new[] { bottleneckSize });
            _gradientBuffer4 = new DenseTensor<float>(new[] { bottleneckSize });
        }

        #endregion

        #region public functions

        public DenseTensor<float> EncoderOutput => _encoderOutput;
        public DenseTensor<float> DecoderOutput => _decoderOutput;

        public float Train(DenseTensor<float> input, float learningRate)
        {
            ForwardPass(input);

            float binaryCrossEntropy = ComputeBinaryCrossEntropyInternal(input);

            #region BackwardPass

            // Вычисление ошибки
            //ComputeBCEGradient(input.Buffer, _decoderOutput.Buffer, _gradientBuffer.Buffer); // Не работает
            TensorPrimitives.Subtract(input.Buffer, _decoderOutput.Buffer, _gradientBuffer.Buffer);

            // Градиенты для декодера
            MatrixMultiplyGradient(_encoderOutput.Buffer, _gradientBuffer.Buffer, learningRate, _weightsDecoder);
            AddScaled(_gradientBuffer.Buffer, _gradientBuffer2.Buffer, learningRate, _biasesDecoder.Buffer);

            // Градиенты для энкодера
            PropagateError(_gradientBuffer.Buffer, _weightsDecoder, _encoderOutput.Buffer, _gradientBuffer3.Buffer);
            MatrixMultiplyGradient(input.Buffer, _gradientBuffer3.Buffer, learningRate, _weightsEncoder);
            AddScaled(_gradientBuffer3.Buffer, _gradientBuffer4.Buffer, learningRate, _biasesEncoder.Buffer);

            #endregion

            return binaryCrossEntropy;
        }

        public float ComputeBinaryCrossEntropy(DenseTensor<float> input)
        {
            ForwardPass(input);

            float binaryCrossEntropy = ComputeBinaryCrossEntropyInternal(input);

            return binaryCrossEntropy;
        }

        #endregion

        private void ForwardPass(DenseTensor<float> input)
        {
            // Прямой проход: Input -> Bottleneck -> Reconstruction
            MatrixMultiply(input.Buffer, _weightsEncoder, _encoderOutput.Buffer);
            TensorPrimitives.Add(_encoderOutput.Buffer, _biasesEncoder.Buffer, _encoderOutput.Buffer);
            ApplyActivation(_encoderOutput.Buffer);

            // Ограничение на количество активных единиц
            ApplyKSparseConstraint(_encoderOutput.Buffer, _maxActiveUnits);

            MatrixMultiply(_encoderOutput.Buffer, _weightsDecoder, _decoderOutput.Buffer);
            TensorPrimitives.Add(_decoderOutput.Buffer, _biasesDecoder.Buffer, _decoderOutput.Buffer);
            ApplyActivation(_decoderOutput.Buffer);
        }

        private void ApplyKSparseConstraint(float[] activations, int maxActiveUnits)
        {
            // Сохраняем только maxActive наибольших значений, остальные обнуляем
            var indices = activations
                .Select((value, index) => (value, index))
                .OrderByDescending(item => item.value)
                .Take(maxActiveUnits)
                .Select(item => item.index)
                .ToHashSet();

            for (int i = 0; i < activations.Length; i++)
            {
                if (!indices.Contains(i))
                    activations[i] = 0;
            }
        }

        private float ComputeBinaryCrossEntropyInternal(DenseTensor<float> input)
        {
            float loss = 0f;
            var inputBuffer = input.Buffer;
            var outputBuffer = _decoderOutput.Buffer;

            for (int i = 0; i < inputBuffer.Length; i++)
            {
                float yTrue = inputBuffer[i];
                float yPred = outputBuffer[i];
                yPred = MathF.Max(MathF.Min(yPred, 1 - 1e-7f), 1e-7f); // Стабильность чисел
                loss += -yTrue * MathF.Log(yPred) - (1 - yTrue) * MathF.Log(1 - yPred);
            }

            return loss / inputBuffer.Length;
        }

        //private void ComputeBCEGradient(ReadOnlySpan<float> input, ReadOnlySpan<float> output, Span<float> gradient)
        //{
        //    for (int i = 0; i < input.Length; i++)
        //    {
        //        float yTrue = input[i];
        //        float yPred = output[i];
        //        yPred = MathF.Max(MathF.Min(yPred, 1 - 1e-7f), 1e-7f); // Стабильность чисел
        //        gradient[i] = (yPred - yTrue) / (yPred * (1 - yPred));
        //    }
        //}

        private static void AddScaled(ReadOnlySpan<float> source, Span<float> source_temp, float scale, Span<float> target)
        {
            TensorPrimitives.Multiply(source, scale, source_temp);
            TensorPrimitives.Add(target, source_temp, target);
            //for (int i = 0; i < target.Length; i++)
            //{
            //    target[i] += source[i] * scale;
            //}
        }

        private static void MatrixMultiply(ReadOnlySpan<float> input, DenseTensor<float> weights, Span<float> output)
        {
            output.Clear();
            for (int i = 0; i < weights.Dimensions[1]; i++)
            {
                for (int j = 0; j < weights.Dimensions[0]; j++)
                {
                    output[i] += input[j] * weights[j, i];
                }
            }
        }

        private static void MatrixMultiplyGradient(ReadOnlySpan<float> input, ReadOnlySpan<float> gradient, float learningRate, DenseTensor<float> weights)
        {
            for (int i = 0; i < weights.Dimensions[0]; i++)
            {
                for (int j = 0; j < weights.Dimensions[1]; j++)
                {
                    weights[i, j] += input[i] * gradient[j] * learningRate;
                }
            }
        }

        private static void PropagateError(ReadOnlySpan<float> gradient, DenseTensor<float> weights, ReadOnlySpan<float> activations, Span<float> outputBuffer)
        {
            outputBuffer.Clear();
            for (int i = 0; i < weights.Dimensions[0]; i++)
            {
                for (int j = 0; j < weights.Dimensions[1]; j++)
                {
                    outputBuffer[i] += gradient[j] * weights[i, j];
                }
            }

            for (int i = 0; i < activations.Length; i++)
            {
                outputBuffer[i] *= activations[i] * (1 - activations[i]); // Производная сигмоиды
            }
        }

        private static void ApplyActivation(Span<float> values)
        {
            TensorPrimitives.Sigmoid(values, values);
            //for (int i = 0; i < values.Length; i++)
            //{
            //    values[i] = Sigmoid(values[i]);
            //}
        }

        //private static float Sigmoid(float x)
        //{
        //    return 1 / (1 + MathF.Exp(-x));
        //}

        private static DenseTensor<float> CreateRandomTensor(int rows, int columns)
        {
            var tensor = new DenseTensor<float>(new[] { rows, columns });
            var random = new Random();
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    tensor[i, j] = (float)(random.NextDouble() - 0.5);
                }
            }
            return tensor;
        }

        #region private fields

        private readonly DenseTensor<float> _weightsEncoder;
        private readonly DenseTensor<float> _biasesEncoder;
        private readonly DenseTensor<float> _weightsDecoder;
        private readonly DenseTensor<float> _biasesDecoder;

        // Буферы для временных данных
        private readonly DenseTensor<float> _encoderOutput;
        private readonly DenseTensor<float> _decoderOutput;
        private readonly DenseTensor<float> _gradientBuffer;
        private readonly DenseTensor<float> _gradientBuffer2;
        private readonly DenseTensor<float> _gradientBuffer3;
        private readonly DenseTensor<float> _gradientBuffer4;

        private readonly int _inputSize;
        private readonly int _bottleneckSize;
        private readonly int _maxActiveUnits;

        #endregion
    }
}