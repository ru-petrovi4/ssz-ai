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
            _weightsEncoder = CreateRandomMatrixFloat(inputSize, bottleneckSize);
            _biasesEncoder = new float[bottleneckSize];
            _weightsDecoder = CreateRandomMatrixFloat(bottleneckSize, inputSize);
            _biasesDecoder = new float[inputSize];

            _bottleneck = new float[bottleneckSize];

            // Буферы для временных данных            
            _temp_Input = new float[_inputSize];
            _temp_Input2 = new float[_inputSize];
            _temp_Bottleneck3 = new float[_bottleneckSize];
            _temp_Bottleneck4 = new float[_bottleneckSize];
            _temp_Bottleneck5 = new float[_bottleneckSize];

            _temp_Output = new float[_inputSize];
            _temp_OutputNormalized = new float[_inputSize];
        }

        /// <summary>
        ///     Используется только для десериализации.
        /// </summary>
        public Autoencoder()
        {            
        }

        #endregion

        #region public functions

        public float[] Bottleneck => _bottleneck;        

        public long TrainingDurationMilliseconds { get; set; }

        public float CosineSimilarity;

        public int IterationsCount;

        public float ControlCosineSimilarity;        

        public float Train(float[] input, float learningRate)
        {
            ForwardPass(input);

            float cosineSimilarity = ComputeCosineSimilarityInternal();

            #region BackwardPass

            // Вычисление ошибки
            //ComputeBCEGradient(input.Buffer, _decoderOutput.Buffer, _gradientBuffer.Buffer); // Не работает
            TensorPrimitives.Subtract(input, _temp_Output, _temp_Input);

            // Градиенты для декодера
            MatrixMultiplyGradient(_bottleneck, _temp_Input, _temp_Bottleneck4, learningRate, _weightsDecoder);
            TensorPrimitives.Multiply(_temp_Input, learningRate, _temp_Input2);
            TensorPrimitives.Add(_biasesDecoder, _temp_Input2, _biasesDecoder);            

            // Градиенты для энкодера
            PropagateError(_temp_Input, _weightsDecoder, _bottleneck, _temp_Bottleneck4, _temp_Bottleneck5, _temp_Bottleneck3);
            MatrixMultiplyGradient(input, _temp_Bottleneck3, _temp_Input2, learningRate, _weightsEncoder);
            TensorPrimitives.Multiply(_temp_Bottleneck3, learningRate, _temp_Bottleneck4);
            TensorPrimitives.Add(_biasesEncoder, _temp_Bottleneck4, _biasesEncoder);            

            #endregion

            return cosineSimilarity;
        }

        public void CalculateShortHash(float[] input, float[] shortHash)
        {
            _input = input;

            // Прямой проход: Input -> Bottleneck -> Reconstruction
            MatrixMultiply(input, _weightsEncoder, _bottleneck);
            TensorPrimitives.Add(_bottleneck, _biasesEncoder, _bottleneck);           

            Array.Clear(shortHash);
            foreach (var i in _bottleneck
                .Select((value, index) => (value, index))
                .OrderByDescending(item => item.value)
                .Take(_maxActiveUnits)
                .Select(item => item.index))
            {
                shortHash[i] = 1.0f;
            }
        }

        public float ComputeCosineSimilarity(float[] input)
        {
            ForwardPass(input);

            float cosineSimilarity = ComputeCosineSimilarityInternal();

            return cosineSimilarity;
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(2))
            {
                writer.Write(_inputSize);
                writer.Write(_bottleneckSize);
                writer.Write(_maxActiveUnits);

                _weightsEncoder.SerializeOwnedData(writer, null);
                writer.WriteArrayOfSingle(_biasesEncoder);                

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

                        _weightsEncoder = new();
                        _weightsEncoder.DeserializeOwnedData(reader, null);
                        _biasesEncoder = reader.ReadArrayOfSingle();
                        
                        if (_temp_WeightsDecoder is null)
                            _temp_WeightsDecoder = new MatrixFloat(0, 0);
                        _temp_WeightsDecoder!.DeserializeOwnedData(reader, null);
                        reader.ReadArrayOfSingle();
                        //_weightsDecoder = new();
                        //_weightsDecoder.DeserializeOwnedData(reader, null);
                        //_biasesDecoder = reader.ReadArrayOfSingle();

                        TrainingDurationMilliseconds = reader.ReadInt64();
                        CosineSimilarity = reader.ReadSingle();
                        IterationsCount = reader.ReadInt32();
                        ControlCosineSimilarity = reader.ReadSingle();

                        _input = null;
                        _bottleneck = new float[_bottleneckSize];                        
                        break;
                    case 2:
                        _inputSize = reader.ReadInt32();
                        _bottleneckSize = reader.ReadInt32();
                        _maxActiveUnits = reader.ReadInt32();

                        _weightsEncoder = new();
                        _weightsEncoder.DeserializeOwnedData(reader, null);
                        _biasesEncoder = reader.ReadArrayOfSingle();                        

                        TrainingDurationMilliseconds = reader.ReadInt64();
                        CosineSimilarity = reader.ReadSingle();
                        IterationsCount = reader.ReadInt32();
                        ControlCosineSimilarity = reader.ReadSingle();

                        _input = null;
                        _bottleneck = new float[_bottleneckSize];
                        break;
                }
            }
        }

        #endregion

        private void ForwardPass(float[] input)
        {
            _input = input;

            // Прямой проход: Input -> Bottleneck -> Reconstruction
            MatrixMultiply(input, _weightsEncoder, _bottleneck);
            TensorPrimitives.Add(_bottleneck, _biasesEncoder, _bottleneck);
            ApplyActivation(_bottleneck);

            // Ограничение на количество активных единиц
            ApplyKSparseConstraint(_bottleneck, _maxActiveUnits);

            MatrixMultiply(_bottleneck, _weightsDecoder, _temp_Output);
            TensorPrimitives.Add(_temp_Output, _biasesDecoder, _temp_Output);
            ApplyActivation(_temp_Output);
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
            for (int i = 0; i < _input!.Length; i++)
            {
                _temp_OutputNormalized[i] = _temp_Output[i] > 0.5 ? 1.0f : 0.0f;                
            }
            
            return TensorPrimitives.CosineSimilarity(_input!, _temp_OutputNormalized);
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

        private static void MatrixMultiply(float[] input, MatrixFloat weights, float[] output)
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

        private static void MatrixMultiplyGradient(ReadOnlySpan<float> input, ReadOnlySpan<float> gradient, Span<float> temp_input, float learningRate, MatrixFloat weights)
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

        private static void PropagateError(ReadOnlySpan<float> gradient, MatrixFloat weights, ReadOnlySpan<float> activations, Span<float> temp_output, Span<float> temp_output2, Span<float> output)
        {
            output.Clear();
            for (int j = 0; j < weights.Dimensions[1]; j++)
            {
                TensorPrimitives.Multiply(weights.GetColumn(j), gradient[j], temp_output);
                TensorPrimitives.Add(output, temp_output, output);
            }

            //output.Clear();
            //for (int i = 0; i < weights.Dimensions[0]; i++)
            //{
            //    for (int j = 0; j < weights.Dimensions[1]; j++)
            //    {
            //        output[i] += gradient[j] * weights[i, j];
            //    }
            //}

            TensorPrimitives.Multiply(output, activations, temp_output);
            TensorPrimitives.Multiply(temp_output, activations, temp_output2);
            TensorPrimitives.Subtract(temp_output, temp_output2, output);

            //for (int i = 0; i < activations.Length; i++)
            //{
            //    output[i] *= activations[i] * (1 - activations[i]); // Производная сигмоиды
            //}
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

        private static MatrixFloat CreateRandomMatrixFloat(int rows, int columns)
        {
            var matrixFloat = new MatrixFloat(rows, columns);
            var random = new Random();
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    matrixFloat[i, j] = (float)(random.NextDouble() - 0.5);
                }
            }
            return matrixFloat;
        }

        #region private fields

        private int _inputSize;
        private int _bottleneckSize;
        private int _maxActiveUnits;

        private MatrixFloat _weightsEncoder = null!;
        private float[] _biasesEncoder = null!;
        private MatrixFloat _weightsDecoder = null!;
        private float[] _biasesDecoder = null!;

        private float[]? _input;
        private float[] _bottleneck = null!;        

        // Буферы для временных данных       

        private float[] _temp_Output = null!;
        private float[] _temp_OutputNormalized = null!;

        private float[] _temp_Input = null!;
        private float[] _temp_Input2 = null!;
        private float[] _temp_Bottleneck3 = null!;
        private float[] _temp_Bottleneck4 = null!;
        private float[] _temp_Bottleneck5 = null!;

        public static MatrixFloat? _temp_WeightsDecoder;

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

//private MatrixFloat _temp_Input = null!;
//private MatrixFloat _temp_Bottleneck = null!;

//var vector = tf.constant(input, shape: new Shape(1, input.Length));
//var matrix = tf.constant(weights.Data, shape: new Shape(weights.Dimensions[0], weights.Dimensions[1]));
//var result = tf.linalg.matmul(vector, matrix);
//result.ToArray<float>(output);