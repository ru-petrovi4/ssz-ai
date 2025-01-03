using Ssz.Utils.Serialization;
using System;
using System.Linq;
using System.Numerics.Tensors;
using Tensorflow;
using static Tensorflow.Binding;

namespace Ssz.AI.Models
{
    public class Autoencoder : IOwnedDataSerializable
    {
        #region construction and destruction

        public Autoencoder(int inputSize, int bottleneckSize, int maxActiveUnits)
        {
            _inputSize = inputSize;
            _bottleneckSize = bottleneckSize;
            _maxActiveUnits = maxActiveUnits;

            // Инициализация весов и смещений
            _weightsEncoder = CreateRandomTensor(inputSize, bottleneckSize);
            _biasesEncoder = new DenseTensor<float>(bottleneckSize);
            _weightsDecoder = CreateRandomTensor(bottleneckSize, inputSize);
            _biasesDecoder = new DenseTensor<float>(inputSize);

            // Буферы для временных данных
            _bottleneck = new DenseTensor<float>(bottleneckSize);
            _output = new DenseTensor<float>(inputSize);
            _outputNormalized = new DenseTensor<float>(inputSize);
            
            _temp_Input = new DenseTensor<float>(inputSize);            
            _temp_Input2 = new DenseTensor<float>(inputSize);
            _temp_Bottleneck3 = new DenseTensor<float>(bottleneckSize);
            _temp_Bottleneck4 = new DenseTensor<float>(bottleneckSize);
        }

        /// <summary>
        ///     Используется только для десериализации.
        /// </summary>
        public Autoencoder()
        {            
        }

        #endregion

        #region public functions

        public DenseTensor<float> Bottleneck => _bottleneck;
        public DenseTensor<float> Output => _output;

        public long TrainingDurationMilliseconds { get; set; }

        public float CosineSimilarity;

        public int IterationsCount;

        public float ControlCosineSimilarity;

        public float Train(DenseTensor<float> input, float learningRate)
        {
            ForwardPass(input);

            float cosineSimilarity = ComputeCosineSimilarityInternal();

            #region BackwardPass

            // Вычисление ошибки
            //ComputeBCEGradient(input.Buffer, _decoderOutput.Buffer, _gradientBuffer.Buffer); // Не работает
            TensorPrimitives.Subtract(input.Data, _output.Data, _temp_Input.Data);

            // Градиенты для декодера
            MatrixMultiplyGradient(_bottleneck.Data, _temp_Input.Data, _temp_Bottleneck4.Data, learningRate, _weightsDecoder);
            TensorPrimitives.Multiply(_temp_Input.Data, learningRate, _temp_Input2.Data);
            TensorPrimitives.Add(_biasesDecoder.Data, _temp_Input2.Data, _biasesDecoder.Data);            

            // Градиенты для энкодера
            PropagateError(_temp_Input.Data, _weightsDecoder, _bottleneck.Data, _temp_Bottleneck3.Data);
            MatrixMultiplyGradient(input.Data, _temp_Bottleneck3.Data, _temp_Input2.Data, learningRate, _weightsEncoder);
            TensorPrimitives.Multiply(_temp_Bottleneck3.Data, learningRate, _temp_Bottleneck4.Data);
            TensorPrimitives.Add(_biasesEncoder.Data, _temp_Bottleneck4.Data, _biasesEncoder.Data);            

            #endregion

            return cosineSimilarity;
        }

        public float ComputeCosineSimilarity(DenseTensor<float> input)
        {
            ForwardPass(input);

            float cosineSimilarity = ComputeCosineSimilarityInternal();

            return cosineSimilarity;
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.Write(_inputSize);
                writer.Write(_bottleneckSize);
                writer.Write(_maxActiveUnits);

                writer.WriteOwnedDataSerializableAndRecreatable(_weightsEncoder, null);
                writer.WriteOwnedDataSerializableAndRecreatable(_biasesEncoder, null);
                writer.WriteOwnedDataSerializableAndRecreatable(_weightsDecoder, null);
                writer.WriteOwnedDataSerializableAndRecreatable(_biasesDecoder, null);

                writer.Write(TrainingDurationMilliseconds);
                writer.Write(CosineSimilarity);
                writer.Write(IterationsCount);
                writer.Write(ControlCosineSimilarity);
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        _inputSize = reader.ReadInt32();
                        _bottleneckSize = reader.ReadInt32();
                        _maxActiveUnits = reader.ReadInt32();

                        _weightsEncoder = reader.ReadOwnedDataSerializableAndRecreatable<DenseTensor<float>>(null)!;
                        _biasesEncoder = reader.ReadOwnedDataSerializableAndRecreatable<DenseTensor<float>>(null)!;
                        _weightsDecoder = reader.ReadOwnedDataSerializableAndRecreatable<DenseTensor<float>>(null)!;
                        _biasesDecoder = reader.ReadOwnedDataSerializableAndRecreatable<DenseTensor<float>>(null)!;

                        TrainingDurationMilliseconds = reader.ReadInt64();
                        CosineSimilarity = reader.ReadSingle();
                        IterationsCount = reader.ReadInt32();
                        ControlCosineSimilarity = reader.ReadSingle();

                        _input = null;
                        _bottleneck = new DenseTensor<float>(_bottleneckSize);
                        _output = new DenseTensor<float>(_inputSize);
                        _outputNormalized = new DenseTensor<float>(_inputSize);

                        _temp_Input = new DenseTensor<float>(_inputSize);
                        _temp_Input2 = new DenseTensor<float>(_inputSize);
                        _temp_Bottleneck3 = new DenseTensor<float>(_bottleneckSize);
                        _temp_Bottleneck4 = new DenseTensor<float>(_bottleneckSize);
                        break;
                }
            }
        }

        #endregion

        private void ForwardPass(DenseTensor<float> input)
        {
            _input = input;

            // Прямой проход: Input -> Bottleneck -> Reconstruction
            MatrixMultiply(input.Data, _weightsEncoder, _bottleneck.Data);
            TensorPrimitives.Add(_bottleneck.Data, _biasesEncoder.Data, _bottleneck.Data);
            ApplyActivation(_bottleneck.Data);

            // Ограничение на количество активных единиц
            ApplyKSparseConstraint(_bottleneck.Data, _maxActiveUnits);

            MatrixMultiply(_bottleneck.Data, _weightsDecoder, _output.Data);
            TensorPrimitives.Add(_output.Data, _biasesDecoder.Data, _output.Data);
            ApplyActivation(_output.Data);
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

        private float ComputeCosineSimilarityInternal()
        {            
            var inputData = _input!.Data;
            var outputData = _output.Data;
            var outputNormalizedData = _outputNormalized.Data;

            for (int i = 0; i < inputData.Length; i++)
            {                
                outputNormalizedData[i] = outputData[i] > 0.5 ? 1.0f : 0.0f;                
            }
            
            return TensorPrimitives.CosineSimilarity(inputData, outputNormalizedData);
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

        private static void MatrixMultiply(float[] input, DenseTensor<float> weights, float[] output)
        {
            for (int j = 0; j < weights.Dimensions[1]; j++)
            {
                output[j] = TensorPrimitives.Dot(input, weights.GetColumn(j));
            }

            //Array.Clear(output);
            //for (int j = 0; j < weights.Dimensions[1]; j++) // 200
            //{
            //    for (int i = 0; i < weights.Dimensions[0]; i++) // 50
            //    {
            //        output[j] += input[i] * weights[i, j];
            //    }
            //}
        }

        private static void MatrixMultiplyGradient(ReadOnlySpan<float> input, ReadOnlySpan<float> gradient, Span<float> temp_input, float learningRate, DenseTensor<float> weights)
        {
            for (int j = 0; j < weights.Dimensions[1]; j++)
            {
                TensorPrimitives.Multiply(input, gradient[j] * learningRate, temp_input);
                var weightsColumn = weights.GetColumn(j);
                TensorPrimitives.Add(weightsColumn, temp_input, weightsColumn);
            }

            //for (int i = 0; i < weights.Dimensions[0]; i++)
            //{
            //    for (int j = 0; j < weights.Dimensions[1]; j++)
            //    {
            //        weights[i, j] += input[i] * gradient[j] * learningRate;
            //    }
            //}
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
            var tensor = new DenseTensor<float>(rows, columns);
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

        private int _inputSize;
        private int _bottleneckSize;
        private int _maxActiveUnits;

        private DenseTensor<float> _weightsEncoder = null!;
        private DenseTensor<float> _biasesEncoder = null!;
        private DenseTensor<float> _weightsDecoder = null!;
        private DenseTensor<float> _biasesDecoder = null!;

        // Буферы для временных данных
        private DenseTensor<float>? _input;
        private DenseTensor<float> _bottleneck = null!;
        private DenseTensor<float> _output = null!;
        private DenseTensor<float> _outputNormalized = null!;

        private DenseTensor<float> _temp_Input = null!;
        private DenseTensor<float> _temp_Input2 = null!;
        private DenseTensor<float> _temp_Bottleneck3 = null!;
        private DenseTensor<float> _temp_Bottleneck4 = null!;        

        #endregion
    }
}

//private static void AddScaled(ReadOnlySpan<float> source, Span<float> temp_source, float scale, Span<float> target)
//{
//    TensorPrimitives.Multiply(source, scale, temp_source);
//    TensorPrimitives.Add(target, temp_source, target);            

//    //for (int i = 0; i < target.Length; i++)
//    //{
//    //    target[i] += source[i] * scale;
//    //}
//}

//private DenseTensor<float> _temp_Input = null!;
//private DenseTensor<float> _temp_Bottleneck = null!;

//var vector = tf.constant(input, shape: new Shape(1, input.Length));
//var matrix = tf.constant(weights.Data, shape: new Shape(weights.Dimensions[0], weights.Dimensions[1]));
//var result = tf.linalg.matmul(vector, matrix);
//result.ToArray<float>(output);