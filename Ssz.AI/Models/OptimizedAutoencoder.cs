using Ssz.Utils.Serialization;
using System;
using System.Linq;
using System.Numerics.Tensors;
using Tensorflow;
using static Tensorflow.Binding;

namespace Ssz.AI.Models
{
    public class OptimizedAutoencoder : ISerializableModelObject
    {
        #region construction and destruction        

        /// <summary>
        ///     Используется только для десериализации.
        /// </summary>
        public OptimizedAutoencoder()
        {            
        }

        #endregion

        #region public functions         

        public long State_TrainingDurationMilliseconds { get; set; }

        public float State_CosineSimilarity;

        public int State_IterationsCount;

        public float State_ControlCosineSimilarity;

        public float[] Temp_ShortHash = null!;

        public float[] Temp_Output_Hash = null!;

        public MatrixFloat State_WeightsEncoder = null!;
        public float[] State_BiasesEncoder = null!;
        public MatrixFloat State_WeightsDecoder = null!;
        public float[] State_BiasesDecoder = null!;

        public void GenereateOwnedData(int inputSize, int bottleneckSize, int? bottleneck_MaxBitsCount)
        {
            _inputSize = inputSize;
            _bottleneckSize = bottleneckSize;
            _bottleneck_MaxBitsCount = bottleneck_MaxBitsCount;

            // Инициализация весов и смещений
            State_WeightsEncoder = CreateRandomMatrixFloat(inputSize, bottleneckSize);
            State_BiasesEncoder = new float[bottleneckSize];
            State_WeightsDecoder = CreateRandomMatrixFloat(_bottleneckSize, inputSize);
            State_BiasesDecoder = new float[inputSize];
        }       

        public void Prepare()
        {
            Temp_ShortHash = new float[_bottleneckSize];
            Temp_Output_Hash = new float[_inputSize];

            _temp_BottleneckFloat = new float[_bottleneckSize];

            // Буферы для временных данных            
            _temp_Input2 = new float[_inputSize];
            _temp_Input3 = new float[_inputSize];
            _temp_Bottleneck3 = new float[_bottleneckSize];
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
            TensorPrimitives.Subtract(input_Hash, _temp_OutputFloat, _temp_Input2);

            // Градиенты для декодера
            MatrixMultiplyGradient(_temp_BottleneckFloat, _temp_Input2, _temp_Bottleneck4, learningRate, State_WeightsDecoder);
            TensorPrimitives.Multiply(_temp_Input2, learningRate, _temp_Input3);
            TensorPrimitives.Add(State_BiasesDecoder, _temp_Input3, State_BiasesDecoder);            

            // Градиенты для энкодера
            PropagateError(_temp_Input2, State_WeightsDecoder, _temp_BottleneckFloat, _temp_Bottleneck4, _temp_Bottleneck5, _temp_Bottleneck3);
            MatrixMultiplyGradient(input_Hash, _temp_Bottleneck3, _temp_Input3, learningRate, State_WeightsEncoder);
            TensorPrimitives.Multiply(_temp_Bottleneck3, learningRate, _temp_Bottleneck4);
            TensorPrimitives.Add(State_BiasesEncoder, _temp_Bottleneck4, State_BiasesEncoder);            

            #endregion

            return cosineSimilarity;
        }

        public void Calculate_ForwardPass(float[] input_Hash)
        {
            float input_Hash_Bits_Count = TensorPrimitives.Sum(input_Hash);

            // Прямой проход: Input -> Bottleneck -> Reconstruction
            MatrixMultiply(input_Hash, State_WeightsEncoder, _temp_BottleneckFloat);
            TensorPrimitives.Add(_temp_BottleneckFloat, State_BiasesEncoder, _temp_BottleneckFloat);
            ApplyActivation(_temp_BottleneckFloat);

            // Ограничение на количество активных единиц
            if (_bottleneck_MaxBitsCount.HasValue)
            {
                ApplyKSparseConstraint(_temp_BottleneckFloat, _bottleneck_MaxBitsCount.Value);

                for (int i = 0; i < Temp_ShortHash.Length; i++)
                {
                    Temp_ShortHash[i] = _temp_BottleneckFloat[i] > 0.0 ? 1.0f : 0.0f;
                }
            }

            MatrixMultiply(_temp_BottleneckFloat, State_WeightsDecoder, _temp_OutputFloat);
            TensorPrimitives.Add(_temp_OutputFloat, State_BiasesDecoder, _temp_OutputFloat);
            ApplyActivation(_temp_OutputFloat);            

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

                State_WeightsEncoder.SerializeOwnedData(writer, null);
                writer.WriteArrayOfSingle(State_BiasesEncoder);
                State_WeightsDecoder.SerializeOwnedData(writer, null);
                writer.WriteArrayOfSingle(State_BiasesDecoder);

                writer.Write(State_TrainingDurationMilliseconds);
                writer.Write(State_CosineSimilarity);
                writer.Write(State_IterationsCount);
                writer.Write(State_ControlCosineSimilarity);
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    //case 1:                        
                    //    _inputSize = reader.ReadOptimizedInt32();
                    //    _bottleneckSize = reader.ReadOptimizedInt32();
                    //    _bottleneck_MaxBitsCount = reader.ReadOptimizedInt32();

                    //    State_WeightsEncoder = new();
                    //    State_WeightsEncoder.DeserializeOwnedData(reader, null);
                    //    State_BiasesEncoder = reader.ReadArrayOfSingle();
                    //    State_WeightsDecoder = new();
                    //    State_WeightsDecoder.DeserializeOwnedData(reader, null);
                    //    State_BiasesDecoder = reader.ReadArrayOfSingle();

                    //    State_TrainingDurationMilliseconds = reader.ReadInt64();
                    //    State_CosineSimilarity = reader.ReadSingle();
                    //    State_IterationsCount = reader.ReadOptimizedInt32();
                    //    State_ControlCosineSimilarity = reader.ReadSingle();                        
                    //    break;
                    //case 2:
                    //    _inputSize = reader.ReadOptimizedInt32();
                    //    _bottleneckSize = reader.ReadOptimizedInt32();
                    //    _bottleneck_MaxBitsCount = reader.ReadNullable<Int32>();

                    //    State_WeightsEncoder = new();
                    //    State_WeightsEncoder.DeserializeOwnedData(reader, null);
                    //    State_BiasesEncoder = reader.ReadArrayOfSingle();
                    //    State_WeightsDecoder = new();
                    //    State_WeightsDecoder.DeserializeOwnedData(reader, null);
                    //    State_BiasesDecoder = reader.ReadArrayOfSingle();

                    //    State_TrainingDurationMilliseconds = reader.ReadInt64();
                    //    State_CosineSimilarity = reader.ReadSingle();
                    //    State_IterationsCount = reader.ReadOptimizedInt32();
                    //    State_ControlCosineSimilarity = reader.ReadSingle();
                    //    break;
                    case 3:
                        _inputSize = reader.ReadInt32();
                        _bottleneckSize = reader.ReadInt32();
                        _bottleneck_MaxBitsCount = reader.ReadNullable<Int32>();

                        State_WeightsEncoder = new();
                        State_WeightsEncoder.DeserializeOwnedData(reader, null);
                        State_BiasesEncoder = reader.ReadArrayOfSingle();
                        State_WeightsDecoder = new();
                        State_WeightsDecoder.DeserializeOwnedData(reader, null);
                        State_BiasesDecoder = reader.ReadArrayOfSingle();

                        State_TrainingDurationMilliseconds = reader.ReadInt64();
                        State_CosineSimilarity = reader.ReadSingle();
                        State_IterationsCount = reader.ReadInt32();
                        State_ControlCosineSimilarity = reader.ReadSingle();
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

        // Буферы для временных данных       
        private float[] _temp_BottleneckFloat = null!;
        private float[] _temp_OutputFloat = null!;        

        private float[] _temp_Input2 = null!;
        private float[] _temp_Input3 = null!;
        private float[] _temp_Bottleneck3 = null!;
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