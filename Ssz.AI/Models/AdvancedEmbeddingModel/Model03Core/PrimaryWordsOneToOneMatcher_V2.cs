using MathNet.Numerics;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;

/// <summary>
/// Сопоставление на основе перестановок
/// </summary>
public class PrimaryWordsOneToOneMatcher_V2 : ISerializableModelObject
{
    private ILoggersSet _loggersSet;
    private Device _device;
    public LanguageDiscreteEmbeddings LanguageDiscreteEmbeddings_RU;
    public LanguageDiscreteEmbeddings LanguageDiscreteEmbeddings_EN;

    public const int BatchSize = 128;

    public const int WordsCount = 10000;

    /// <summary>
    ///     Index RU -> Index EN
    /// </summary>
    public int[] Mapping_RU_EN = null!;
    /// <summary>
    ///     Index EN -> Index RU
    /// </summary>
    public int[] Mapping_EN_RU = null!;

    //public torch.Tensor Temp_EnergyBits_Tensor_RU = null!;
    public MatrixFloat Temp_ClustersEnergy_Matrix_RU = null!;
    /// <summary>
    ///     [Word.Index, [Bit Index]]
    /// </summary>
    public int[][] Temp_WordBitIndices_Collection_RU = null!;
    /// <summary>
    ///     [Word.Index, [Mapped Bit Index]]
    /// </summary>
    public int[][] Temp_WordMappedBitIndices_Collection_RU = null!;

    //public torch.Tensor Temp_EnergyBits_Tensor_EN = null!;
    public MatrixFloat Temp_ClustersEnergy_Matrix_EN = null!;
    /// <summary>
    ///     [Word.Index, [Bit Index]]
    /// </summary>
    public int[][] Temp_WordBitIndices_Collection_EN = null!;
    /// <summary>
    ///     [Word.Index, [Mapped Bit Index]]
    /// </summary>
    public int[][] Temp_WordMappedBitIndices_Collection_EN = null!;

    public float Temp_MinMeanEnergy;

    public float Temp_ForBatchChangesCount;

    //public float[] Temp_EnergyOfBitCollection = null!;

    public PrimaryWordsOneToOneMatcher_V2(ILoggersSet loggersSet, LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU, LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN)
    {
        _loggersSet = loggersSet;
        _device = cuda.is_available() ? CUDA : CPU;
        LanguageDiscreteEmbeddings_RU = languageDiscreteEmbeddings_RU;
        LanguageDiscreteEmbeddings_EN = languageDiscreteEmbeddings_EN;        
    }

    public void GenerateOwnedData(int vectorLength)
    {
        Mapping_RU_EN = new int[vectorLength];
        Mapping_EN_RU = new int[vectorLength];
        bool[] isMapped_RU_EN = new bool[vectorLength];

        var r = new Random(1);
        
        for (int bitIndex_RU = 0; bitIndex_RU < vectorLength; bitIndex_RU += 1)
        {
            for (; ; )
            {
                int bitIndex_EN = r.Next(vectorLength);                
                if (isMapped_RU_EN[bitIndex_EN])
                    continue;

                Mapping_RU_EN[bitIndex_RU] = bitIndex_EN;
                Mapping_EN_RU[bitIndex_EN] = bitIndex_RU;
                isMapped_RU_EN[bitIndex_EN] = true;
                break;
            }
        }
    }

    public void Prepare()
    {        
        Temp_ClustersEnergy_Matrix_RU = GetClustersEnergy_Matrix_V1(LanguageDiscreteEmbeddings_RU);
        Temp_ClustersEnergy_Matrix_EN = GetClustersEnergy_Matrix_V1(LanguageDiscreteEmbeddings_EN);        

        Temp_WordBitIndices_Collection_RU = new int[LanguageDiscreteEmbeddings_RU.Words.Count][];
        Temp_WordMappedBitIndices_Collection_RU = new int[LanguageDiscreteEmbeddings_RU.Words.Count][];
        List<int> wordBitIndices = new List<int>(16);
        List<int> wordMappedBitIndices = new List<int>(16);
        for (int i = 0; i < WordsCount; i += 1)
        {            
            var discreteVector_PrimaryBitsOnly = LanguageDiscreteEmbeddings_RU.Words[i].DiscreteVector_PrimaryBitsOnly;
            wordBitIndices.Clear();
            wordMappedBitIndices.Clear();
            for (int j = 0; j < discreteVector_PrimaryBitsOnly.Length; j += 1)
            {
                if (discreteVector_PrimaryBitsOnly[j] > 0.0f)
                {
                    wordBitIndices.Add(j);
                    wordMappedBitIndices.Add(Mapping_RU_EN[j]);
                }
            }
            Temp_WordBitIndices_Collection_RU[i] = wordBitIndices.ToArray();
            Temp_WordMappedBitIndices_Collection_RU[i] = wordMappedBitIndices.ToArray();
        }        
              
        Temp_WordBitIndices_Collection_EN = new int[LanguageDiscreteEmbeddings_EN.Words.Count][];
        Temp_WordMappedBitIndices_Collection_EN = new int[LanguageDiscreteEmbeddings_EN.Words.Count][];        
        for (int i = 0; i < WordsCount; i += 1)
        {
            var discreteVector = LanguageDiscreteEmbeddings_EN.Words[i].DiscreteVector_PrimaryBitsOnly;
            wordBitIndices.Clear();
            wordMappedBitIndices.Clear();
            for (int j = 0; j < discreteVector.Length; j += 1)
            {
                if (discreteVector[j] > 0.0f)
                {
                    wordBitIndices.Add(j);
                    wordMappedBitIndices.Add(Mapping_EN_RU[j]);
                }
            }
            Temp_WordBitIndices_Collection_EN[i] = wordBitIndices.ToArray();
            Temp_WordMappedBitIndices_Collection_EN[i] = wordMappedBitIndices.ToArray();
        }
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteArray(Mapping_RU_EN);
            writer.WriteArray(Mapping_EN_RU);
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    Mapping_RU_EN = reader.ReadArray<int>()!;
                    Mapping_EN_RU = reader.ReadArray<int>()!;
                    break;
            }
        }
    }

    public void CalculateMapping()
    {
        var stopwatch = Stopwatch.StartNew();
        var r = new Random(5);
        
        int iterationsCount = WordsCount / BatchSize + 1;
        for (int epoch = 0; epoch < 1; epoch += 1)
        {
            var random_Words_RU = WordsHelper.GetRandomOrderWords(LanguageDiscreteEmbeddings_RU.Words, WordsCount, r).ToArray();
            var random_Words_EN = WordsHelper.GetRandomOrderWords(LanguageDiscreteEmbeddings_EN.Words, WordsCount, r).ToArray();

            for (int iterationN = 0; iterationN < iterationsCount; iterationN += 1)
            {
                int wordIndexStart = iterationN * BatchSize;
                if (wordIndexStart >= WordsCount)
                    break;
                var batchWords_RU = random_Words_RU.AsMemory(wordIndexStart, Math.Min(BatchSize, random_Words_RU.Length - wordIndexStart));
                var batchWords_EN = random_Words_EN.AsMemory(wordIndexStart, Math.Min(BatchSize, random_Words_EN.Length - wordIndexStart));

                float minMeanEnergy = float.MaxValue;
                for (; ; )
                {
                    Temp_ForBatchChangesCount = 0;

                    for (int i = 0; i < Mapping_RU_EN.Length; i += 1)
                    {
                        OptimizeWordsBatch(
                            r.Next(Mapping_RU_EN.Length),
                            batchWords_RU,
                            Mapping_RU_EN,
                            Temp_WordBitIndices_Collection_RU,
                            Temp_WordMappedBitIndices_Collection_RU,
                            Mapping_EN_RU
                            );
                    }

                    if (Temp_MinMeanEnergy == minMeanEnergy) 
                        break;

                    minMeanEnergy = Temp_MinMeanEnergy;
                }                

                //if (iterationN % 10 == 0)
                {
                    _loggersSet.UserFriendlyLogger.LogInformation($"CalculateMapping iteration: {iterationN} done. MinEnergy: {Temp_MinMeanEnergy}; ForBatchChangesCount: {Temp_ForBatchChangesCount}; Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);
                }
            }
        }

        stopwatch.Stop();
        _loggersSet.UserFriendlyLogger.LogInformation("CalculateMapping totally done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    /// <summary>
    ///     1-cos, экспонента, квадрат
    ///     Индексы соотвествуют индексам главных бит в слове.
    /// </summary>
    /// <param name="languageDiscreteEmbeddings"></param>
    /// <returns></returns>
    public static MatrixFloat GetClustersEnergy_Matrix_V1(LanguageDiscreteEmbeddings embeddings)
    {
        int dimension = embeddings.ClusterInfos.Count;
        var matrixFloat = new MatrixFloat(dimension, dimension);
        foreach (var i in Enumerable.Range(0, dimension))
        {
            foreach (var j in Enumerable.Range(0, dimension))
            {
                var clusterI = embeddings.ClusterInfos[i];
                var clusterJ = embeddings.ClusterInfos[j];
                float v = System.Numerics.Tensors.TensorPrimitives.CosineSimilarity(
                    clusterI.CentroidOldVectorNormalized,
                    clusterJ.CentroidOldVectorNormalized);
                v = MathF.Exp((1 - v) * 3.0f) - 1;
                v = v * v;
                matrixFloat[clusterI.HashProjectionIndex, clusterJ.HashProjectionIndex] = v;
            }
        }
        return matrixFloat;
    }

    public static float GetWordEnergy(WordWithDiscreteEmbedding word, MatrixFloat energyMatrix)
    {
        var vector = word.DiscreteVector_PrimaryBitsOnly;

        // Список индексов, где vector[i] == 1 для быстрого перебора (около 8 элементов)
        Span<int> activeIndices = stackalloc int[8];
        int count = 0;
        for (int i = 0; i < vector.Length; i += 1)
        {
            if (vector[i] == 1.0f)
            {
                // Обычно у вас мало единичных элементов, stackalloc более производителен для малых массивов
                activeIndices[count] = i;
                count += 1;
            }
        }

        return GetEnergy(activeIndices, energyMatrix);
    }

    private void OptimizeWordsBatch(
        int toOptimize_BitIndex_A,
        Memory<WordWithDiscreteEmbedding> batchWords_A,
        int[] mapping_A_B,
        int[][] wordBitIndices_Collection_A,
        int[][] wordMappedBitIndices_Collection_A,
        int[] mapping_B_A)
    {   
        Temp_MinMeanEnergy = float.MaxValue;

        int original_BitIndex_B = mapping_A_B[toOptimize_BitIndex_A];
        int minEnergy_BitIndex_B = 0;
        for (int test_BitIndex_B = 0; test_BitIndex_B < mapping_A_B.Length; test_BitIndex_B += 1)
        {
            int old_Test_BitIndex_A = mapping_B_A[test_BitIndex_B];
            int old_ToOptimize_BitIndex_B = mapping_A_B[toOptimize_BitIndex_A];
            mapping_A_B[toOptimize_BitIndex_A] = test_BitIndex_B;
            mapping_A_B[old_Test_BitIndex_A] = old_ToOptimize_BitIndex_B;
            mapping_B_A[test_BitIndex_B] = toOptimize_BitIndex_A;
            mapping_B_A[old_ToOptimize_BitIndex_B] = old_Test_BitIndex_A;

            int energy = 0;

            Parallel.For(
                0, // начальный индекс (включительно)
                batchWords_A.Length, // конечный индекс (не включительно)
                () => 0, // Функция инициализации локальной переменной для каждого потока: localSum = 0
                (batchWordsIndex, loopState, localEnergy) => // Основной делегат, исполняемый в параллельном потоке
                {   
                    int wordIndex_A = batchWords_A.Span[batchWordsIndex].Index;
                    int[] wordBitIndices = wordBitIndices_Collection_A[wordIndex_A];
                    int[] wordMappedBitIndices = wordMappedBitIndices_Collection_A[wordIndex_A];

                    for (int i = 0; i < wordBitIndices.Length; i += 1)
                    {
                        int bitIndex = wordBitIndices[i];
                        wordMappedBitIndices[i] = mapping_A_B[bitIndex];
                    }

                    // Добавляем к локальной сумме результат текущей итерации
                    localEnergy += (int)GetEnergy(wordMappedBitIndices, Temp_ClustersEnergy_Matrix_EN);                    
                    
                    return localEnergy;
                },
                // Итоговая агрегация локальных сумм каждого потока
                localEnergy =>
                {
                    Interlocked.Add(ref energy, localEnergy);                    
                });
            // UNOPTIMIZED
            //for (int batchWordsIndex = 0; batchWordsIndex < batchWords_Span_A.Length; batchWordsIndex += 1)
            //{
            //    int wordIndex_A = batchWords_Span_A[batchWordsIndex].Index;
            //    int[] wordBitIndices = wordBitIndices_Collection_A[wordIndex_A];
            //    int[] wordMappedBitIndices = wordMappedBitIndices_Collection_A[wordIndex_A];

            //    for (int i = 0; i < wordBitIndices.Length; i += 1)
            //    {
            //        int bitIndex = wordBitIndices[i];
            //        wordMappedBitIndices[i] = mapping_A_B[bitIndex];
            //    }

            //    energy += GetEnergy(wordMappedBitIndices, Temp_ClustersEnergy_Matrix_EN);
            //}
            float meanEnergy = (float)energy / (float)batchWords_A.Length;

            if (meanEnergy < Temp_MinMeanEnergy)
            {
                Temp_MinMeanEnergy = meanEnergy;
                minEnergy_BitIndex_B = test_BitIndex_B;
            }

            mapping_A_B[toOptimize_BitIndex_A] = old_ToOptimize_BitIndex_B;
            mapping_A_B[old_Test_BitIndex_A] = test_BitIndex_B;
            mapping_B_A[test_BitIndex_B] = old_Test_BitIndex_A;
            mapping_B_A[old_ToOptimize_BitIndex_B] = toOptimize_BitIndex_A;
        }        

        if (original_BitIndex_B != minEnergy_BitIndex_B)
        {
            int old_Test_BitIndex_A = mapping_B_A[minEnergy_BitIndex_B];
            int old_ToOptimize_BitIndex_B = mapping_A_B[toOptimize_BitIndex_A];
            mapping_A_B[toOptimize_BitIndex_A] = minEnergy_BitIndex_B;
            mapping_A_B[old_Test_BitIndex_A] = old_ToOptimize_BitIndex_B;
            mapping_B_A[minEnergy_BitIndex_B] = toOptimize_BitIndex_A;
            mapping_B_A[old_ToOptimize_BitIndex_B] = old_Test_BitIndex_A;

            Temp_ForBatchChangesCount += 1;
        }        
    }            

    private static float GetEnergy(ReadOnlySpan<int> bitIndices, MatrixFloat energyMatrix)
    {
        float energy = 0.0f;
        // Перебираем только пары (i < j), чтобы не считать симметричную/диагональную энергию дважды
        for (int k = 0; k < bitIndices.Length - 1; k += 1)
        {
            int i = bitIndices[k];
            for (int l = k + 1; l < bitIndices.Length; l += 1)
            {
                int j = bitIndices[l];
                energy += energyMatrix[i, j];
            }
        }
        return energy;
    }
}
