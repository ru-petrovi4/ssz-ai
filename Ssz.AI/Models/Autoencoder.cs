using Ssz.Utils.Serialization;
using System;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models
{
    public class Autoencoder : ISerializableModelObject
    {
        #region construction and destruction        

        /// <summary>
        ///     Используется только для десериализации.
        /// </summary>
        public Autoencoder()
        {            
        }

        #endregion

        #region public functions         

        public long TrainingDurationMilliseconds { get; set; }

        public float CosineSimilarity;

        public int IterationsCount;

        public float ControlCosineSimilarity;

        public float[] Temp_ShortHash = null!;

        public float[] Temp_Output_Hash = null!;

        public void GenerateOwnedData(int inputSize, int bottleneckSize, int? bottleneck_MaxBitsCount)
        {
            _inputSize = inputSize;
            _bottleneckSize = bottleneckSize;
            _bottleneck_MaxBitsCount = bottleneck_MaxBitsCount;

            // Инициализация весов и смещений
            _weightsEncoder = CreateRandomMatrixFloat(inputSize, bottleneckSize);
            _biasesEncoder = new float[bottleneckSize];
            _weightsDecoder = CreateRandomMatrixFloat(_bottleneckSize, inputSize);
            _biasesDecoder = new float[inputSize];
        }       

        public void Prepare()
        {
            Temp_ShortHash = new float[_bottleneckSize];
            Temp_Output_Hash = new float[_inputSize];

            _temp_BottleneckFloat = new float[_bottleneckSize];

            // Буферы для временных данных            
            _temp_Gradient = new float[_inputSize];
            _temp_Gradient_Rated = new float[_inputSize];            
            _temp_BottleneckGradient = new float[_bottleneckSize];
            _temp_BottleneckGradient_Rated = new float[_bottleneckSize];
            _temp_Input3 = new float[_inputSize];
            _temp_Bottleneck4 = new float[_bottleneckSize];
            _temp_Bottleneck5 = new float[_bottleneckSize];

            _temp_OutputFloat = new float[_inputSize];            
        }

        public float Calculate(float[] input_Hash, float learningRate)
        {
            Calculate_ForwardPass(input_Hash);

            float cosineSimilarity = TensorPrimitives.CosineSimilarity(input_Hash, Temp_Output_Hash);

            #region BackwardPass

            // Вычисление ошибки
            //ComputeBCEGradient(input.Buffer, _decoderOutput.Buffer, _gradientBuffer.Buffer); // Не работает
            TensorPrimitives.Subtract(input_Hash, _temp_OutputFloat, _temp_Gradient);

            // Градиенты для декодера
            TensorPrimitives.Multiply(_temp_Gradient, learningRate, _temp_Gradient_Rated);
            MatrixMultiplyGradient(_temp_BottleneckFloat, _temp_Gradient_Rated, _temp_Bottleneck4, _weightsDecoder);            
            TensorPrimitives.Add(_biasesDecoder, _temp_Gradient_Rated, _biasesDecoder);            

            // Градиенты для энкодера
            PropagateError(_temp_Gradient, _weightsDecoder, _temp_BottleneckFloat, _temp_Bottleneck4, _temp_Bottleneck5, _temp_BottleneckGradient);
            TensorPrimitives.Multiply(_temp_BottleneckGradient, learningRate, _temp_BottleneckGradient_Rated);
            MatrixMultiplyGradient(input_Hash, _temp_BottleneckGradient_Rated, _temp_Input3, _weightsEncoder);            
            TensorPrimitives.Add(_biasesEncoder, _temp_BottleneckGradient_Rated, _biasesEncoder);            

            #endregion

            return cosineSimilarity;
        }

        public void Calculate_ForwardPass(float[] input_Hash)
        {
            float input_Hash_Bits_Count = TensorPrimitives.Sum(input_Hash);

            // Прямой проход: Input -> Bottleneck -> Reconstruction
            MatrixMultiply(input_Hash, _weightsEncoder, _temp_BottleneckFloat);
            TensorPrimitives.Add(_temp_BottleneckFloat, _biasesEncoder, _temp_BottleneckFloat);            
            TensorPrimitives.Sigmoid(_temp_BottleneckFloat, _temp_BottleneckFloat);

            // Ограничение на количество активных единиц
            if (_bottleneck_MaxBitsCount.HasValue)
            {
                ApplyKSparseConstraint(_temp_BottleneckFloat, _bottleneck_MaxBitsCount.Value);

                for (int i = 0; i < Temp_ShortHash.Length; i++)
                {
                    Temp_ShortHash[i] = _temp_BottleneckFloat[i] > 0.0 ? 1.0f : 0.0f;
                }
            }

            MatrixMultiply(_temp_BottleneckFloat, _weightsDecoder, _temp_OutputFloat);
            TensorPrimitives.Add(_temp_OutputFloat, _biasesDecoder, _temp_OutputFloat);            
            TensorPrimitives.Sigmoid(_temp_OutputFloat, _temp_OutputFloat);

            for (int i = 0; i < Temp_Output_Hash.Length; i++)
            {
                Temp_Output_Hash[i] = _temp_OutputFloat[i];
            }
            ApplyKSparseConstraint(Temp_Output_Hash, (int)input_Hash_Bits_Count);
            for (int i = 0; i < Temp_Output_Hash.Length; i++)
            {
                Temp_Output_Hash[i] = Temp_Output_Hash[i] > 0.0 ? 1.0f : 0.0f;
            }
        }

        //public void GetShortHash(float[] input, float[] shortHash)
        //{
        //    Temp_Input_Hash = input;

        //    // Прямой проход: Input -> Bottleneck -> Reconstruction
        //    MatrixMultiply(input, _state_WeightsEncoder, Temp_ShortHash);
        //    TensorPrimitives.Add(Temp_ShortHash, _state_BiasesEncoder, Temp_ShortHash);           

        //    Array.Clear(shortHash);
        //    foreach (var i in Temp_ShortHash
        //        .Select((value, index) => (value, index))
        //        .OrderByDescending(item => item.value)
        //        .Take(_maxActiveUnits)
        //        .Select(item => item.index))
        //    {
        //        shortHash[i] = 1.0f;
        //    }
        //}        

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(3))
            {
                writer.Write(_inputSize);
                writer.Write(_bottleneckSize);
                writer.WriteNullable(_bottleneck_MaxBitsCount);

                _weightsEncoder.SerializeOwnedData(writer, null);
                writer.WriteArrayOfSingle(_biasesEncoder);
                _weightsDecoder.SerializeOwnedData(writer, null);
                writer.WriteArrayOfSingle(_biasesDecoder);

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
                    case 3:
                        _inputSize = reader.ReadInt32();
                        _bottleneckSize = reader.ReadInt32();
                        _bottleneck_MaxBitsCount = reader.ReadNullable<Int32>();

                        _weightsEncoder = new();
                        _weightsEncoder.DeserializeOwnedData(reader, null);
                        _biasesEncoder = reader.ReadArrayOfSingle();
                        _weightsDecoder = new();
                        _weightsDecoder.DeserializeOwnedData(reader, null);
                        _biasesDecoder = reader.ReadArrayOfSingle();

                        TrainingDurationMilliseconds = reader.ReadInt64();
                        CosineSimilarity = reader.ReadSingle();
                        IterationsCount = reader.ReadInt32();
                        ControlCosineSimilarity = reader.ReadSingle();
                        break;
                }
            }
        }

        #endregion

        private void ApplyKSparseConstraint(float[] shortHash, int maxActiveUnits)
        {
            // Сохраняем только maxActive наибольших значений, остальные обнуляем
            var indices = shortHash
                .Select((value, index) => (value, index))
                .OrderByDescending(item => item.value)
                .Take(maxActiveUnits)
                .Select(item => item.index)
                .ToHashSet();

            for (int i = 0; i < shortHash.Length; i++)
            {
                if (!indices.Contains(i))
                    shortHash[i] = 0.0f;                
            }
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

        private static void MatrixMultiply(float[] input, MatrixFloat_ColumnMajor weights, float[] output)
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

        private static void MatrixMultiplyGradient(ReadOnlySpan<float> input, ReadOnlySpan<float> gradient_Rated, Span<float> temp_Input, MatrixFloat_ColumnMajor weights)
        {
            for (int j = 0; j < weights.Dimensions[1]; j++)
            {
                TensorPrimitives.Multiply(input, gradient_Rated[j], temp_Input);
                var weightsColumn = weights.GetColumn(j);
                TensorPrimitives.Add(weightsColumn, temp_Input, weightsColumn);
            }

            //for (int i = 0; i < weights.Dimensions[0]; i++)
            //{
            //    for (int j = 0; j < weights.Dimensions[1]; j++)
            //    {
            //        weights[i, j] += input[i] * gradient[j] * learningRate;
            //    }
            //}
        }

        private static void PropagateError(ReadOnlySpan<float> gradient, MatrixFloat_ColumnMajor weightsDecoder, ReadOnlySpan<float> bottleneckFloat, Span<float> temp_Bottleneck4, Span<float> temp_Bottleneck5, Span<float> bottleneckGradient)
        {
            bottleneckGradient.Clear();
            for (int j = 0; j < weightsDecoder.Dimensions[1]; j++)
            {
                TensorPrimitives.Multiply(weightsDecoder.GetColumn(j), gradient[j], temp_Bottleneck4);
                TensorPrimitives.Add(bottleneckGradient, temp_Bottleneck4, bottleneckGradient);
            }

            //output.Clear();
            //for (int i = 0; i < weights.Dimensions[0]; i++)
            //{
            //    for (int j = 0; j < weights.Dimensions[1]; j++)
            //    {
            //        output[i] += gradient[j] * weights[i, j];
            //    }
            //}

            TensorPrimitives.Multiply(bottleneckGradient, bottleneckFloat, temp_Bottleneck4);
            TensorPrimitives.Multiply(temp_Bottleneck4, bottleneckFloat, temp_Bottleneck5);
            TensorPrimitives.Subtract(temp_Bottleneck4, temp_Bottleneck5, bottleneckGradient);

            //for (int i = 0; i < activations.Length; i++)
            //{
            //    output[i] *= activations[i] * (1 - activations[i]); // Производная сигмоиды
            //}
        }

        //private static void ApplyActivation(Span<float> values)
        //{
        //    TensorPrimitives.Sigmoid(values, values);

        //    //for (int i = 0; i < values.Length; i++)
        //    //{
        //    //    values[i] = Sigmoid(values[i]);
        //    //}
        //}

        //private static float Sigmoid(float x)
        //{
        //    return 1 / (1 + MathF.Exp(-x));
        //}

        private static MatrixFloat_ColumnMajor CreateRandomMatrixFloat(int rows, int columns)
        {
            var matrixFloat = new MatrixFloat_ColumnMajor(rows, columns);
            var random = new Random();
            for (int i = 0; i < matrixFloat.Data.Length; i++)
            {
                matrixFloat.Data[i] = (float)(random.NextDouble() - 0.5);
            }
            return matrixFloat;
        }

        #region private fields

        private int _inputSize;
        private int _bottleneckSize;
        private int? _bottleneck_MaxBitsCount;

        private MatrixFloat_ColumnMajor _weightsEncoder = null!;
        private float[] _biasesEncoder = null!;
        private MatrixFloat_ColumnMajor _weightsDecoder = null!;
        private float[] _biasesDecoder = null!;

        // Буферы для временных данных       
        private float[] _temp_BottleneckFloat = null!;
        private float[] _temp_OutputFloat = null!;        

        private float[] _temp_Gradient = null!;
        private float[] _temp_Gradient_Rated = null!;
        private float[] _temp_BottleneckGradient = null!;
        private float[] _temp_BottleneckGradient_Rated = null!;
        private float[] _temp_Input3 = null!;        
        private float[] _temp_Bottleneck4 = null!;
        private float[] _temp_Bottleneck5 = null!;

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