using MathNet.Numerics;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;

/// <summary>
/// Сопоставление на основе перестановок
/// </summary>
public class ClustersOneToOneMatcher_Swapping : ISerializableModelObject
{
    private ILoggersSet _loggersSet;
    private Device _device;
    public LanguageDiscreteEmbeddings LanguageDiscreteEmbeddings_A;
    public LanguageDiscreteEmbeddings LanguageDiscreteEmbeddings_B;

    public const int BatchSize = 5000;

    public int WordsCount = 20000;

    /// <summary>
    ///     Primary Bit Index A -> Primary Bit Index B
    /// </summary>
    public int[] PrimaryBitsMapping_A_B = null!;

    public float[] PrimaryBitsMapping_Strength_A_B = null!;

    /// <summary>
    ///     Primary Bit Index B -> Primary Bit Index A
    /// </summary>
    public int[] PrimaryBitsMapping_B_A = null!;

    public float[] PrimaryBitsMapping_Strength_B_A = null!;

    //public torch.Tensor Temp_EnergyBits_Tensor_A = null!;
    public MatrixFloat Temp_PrimaryBitsEnergy_Matrix_A = null!;
    /// <summary>
    ///     [Word.Index, [Bit Index]]
    /// </summary>
    public int[][] Temp_WordPrimaryBitIndices_Collection_A = null!;
    /// <summary>
    ///     [Word.Index, [Mapped Bit Index]]
    /// </summary>
    public int[][] Temp_WordMappedPrimaryBitIndices_Collection_A = null!;

    //public torch.Tensor Temp_EnergyBits_Tensor_B = null!;
    public MatrixFloat Temp_PrimaryBitsEnergy_Matrix_B = null!;
    /// <summary>
    ///     [Word.Index, [Bit Index]]
    /// </summary>
    public int[][] Temp_WordPrimaryBitIndices_Collection_B = null!;
    /// <summary>
    ///     [Word.Index, [Mapped Bit Index]]
    /// </summary>
    public int[][] Temp_WordMappedPrimaryBitIndices_Collection_B = null!;

    public ClustersOneToOneMatcher_Swapping(ILoggersSet loggersSet, LanguageDiscreteEmbeddings languageDiscreteEmbeddings_A, LanguageDiscreteEmbeddings languageDiscreteEmbeddings_B)
    {
        _loggersSet = loggersSet;
        _device = cuda.is_available() ? CUDA : CPU;
        LanguageDiscreteEmbeddings_A = languageDiscreteEmbeddings_A;
        LanguageDiscreteEmbeddings_B = languageDiscreteEmbeddings_B;        
    }

    public void GenerateOwnedData(int clustersCount, (int clusterIndexA, int clusterIndexB) initialLink)
    {
        PrimaryBitsMapping_A_B = new int[clustersCount];
        PrimaryBitsMapping_Strength_A_B = new float[clustersCount];
        PrimaryBitsMapping_B_A = new int[clustersCount];
        PrimaryBitsMapping_Strength_B_A = new float[clustersCount];
        bool[] isMapped_A_B = new bool[clustersCount];

        int initialBitIndex_A = LanguageDiscreteEmbeddings_A.ClusterInfos[initialLink.clusterIndexA].HashProjectionIndex;
        int initialBitIndex_B = LanguageDiscreteEmbeddings_B.ClusterInfos[initialLink.clusterIndexB].HashProjectionIndex;
        isMapped_A_B[initialBitIndex_B] = true;
        PrimaryBitsMapping_A_B[initialBitIndex_A] = initialBitIndex_B;
        PrimaryBitsMapping_Strength_A_B[initialBitIndex_A] = 1.0f;
        PrimaryBitsMapping_B_A[initialBitIndex_B] = initialBitIndex_A;
        PrimaryBitsMapping_Strength_B_A[initialBitIndex_B] = 1.0f;

        var r = new Random(1);
        
        for (int bitIndex_A = 0; bitIndex_A < clustersCount; bitIndex_A += 1)
        {
            if (bitIndex_A == initialBitIndex_A)
                continue;
            for (; ; )
            {
                int bitIndex_B = r.Next(clustersCount);                
                if (isMapped_A_B[bitIndex_B])
                    continue;

                PrimaryBitsMapping_A_B[bitIndex_A] = bitIndex_B;
                PrimaryBitsMapping_B_A[bitIndex_B] = bitIndex_A;
                isMapped_A_B[bitIndex_B] = true;
                break;
            }
        }
    }

    //public void GenerateOwnedData2(int[] initialPrimaryBitsMapping_A_B)
    //{
    //    PrimaryBitsMapping_A_B = initialPrimaryBitsMapping_A_B;
    //    PrimaryBitsMapping_B_A = new int[PrimaryBitsMapping_A_B.Length];
    //    for (int bitIndex_A = 0; bitIndex_A < PrimaryBitsMapping_A_B.Length; bitIndex_A += 1)
    //    {
    //        int bitIndex_B = PrimaryBitsMapping_A_B[bitIndex_A];
    //        PrimaryBitsMapping_B_A[bitIndex_B] = bitIndex_A;
    //    }
    //}

    public void Prepare()
    {
        WordsCount = Math.Min(Math.Min(WordsCount, LanguageDiscreteEmbeddings_A.Words.Count), LanguageDiscreteEmbeddings_A.Words.Count);

        Temp_PrimaryBitsEnergy_Matrix_A = ModelHelper.GetPrimaryBitsEnergy_Matrix(LanguageDiscreteEmbeddings_A.ClusterInfos);
        Temp_PrimaryBitsEnergy_Matrix_B = ModelHelper.GetPrimaryBitsEnergy_Matrix(LanguageDiscreteEmbeddings_B.ClusterInfos);        

        Temp_WordPrimaryBitIndices_Collection_A = new int[WordsCount][];
        Temp_WordMappedPrimaryBitIndices_Collection_A = new int[WordsCount][];
        List<int> wordPrimaryBitIndices = new List<int>(16);
        List<int> wordMappedPrimaryBitIndices = new List<int>(16);
        for (int i = 0; i < WordsCount; i += 1)
        {            
            var discreteVector_PrimaryBitsOnly = LanguageDiscreteEmbeddings_A.Words[i].DiscreteVector_PrimaryBitsOnly;
            wordPrimaryBitIndices.Clear();
            wordMappedPrimaryBitIndices.Clear();
            for (int j = 0; j < discreteVector_PrimaryBitsOnly.Length; j += 1)
            {
                if (discreteVector_PrimaryBitsOnly[j] > 0.5f)
                {
                    wordPrimaryBitIndices.Add(j);
                    wordMappedPrimaryBitIndices.Add(PrimaryBitsMapping_A_B[j]);
                }
            }
            Temp_WordPrimaryBitIndices_Collection_A[i] = wordPrimaryBitIndices.ToArray();
            Temp_WordMappedPrimaryBitIndices_Collection_A[i] = wordMappedPrimaryBitIndices.ToArray();
        }        
              
        Temp_WordPrimaryBitIndices_Collection_B = new int[WordsCount][];
        Temp_WordMappedPrimaryBitIndices_Collection_B = new int[WordsCount][];        
        for (int i = 0; i < WordsCount; i += 1)
        {
            var discreteVector_PrimaryBitsOnly = LanguageDiscreteEmbeddings_B.Words[i].DiscreteVector_PrimaryBitsOnly;
            wordPrimaryBitIndices.Clear();
            wordMappedPrimaryBitIndices.Clear();
            for (int j = 0; j < discreteVector_PrimaryBitsOnly.Length; j += 1)
            {
                if (discreteVector_PrimaryBitsOnly[j] > 0.5f)
                {
                    wordPrimaryBitIndices.Add(j);
                    wordMappedPrimaryBitIndices.Add(PrimaryBitsMapping_B_A[j]);
                }
            }
            Temp_WordPrimaryBitIndices_Collection_B[i] = wordPrimaryBitIndices.ToArray();
            Temp_WordMappedPrimaryBitIndices_Collection_B[i] = wordMappedPrimaryBitIndices.ToArray();
        }
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(2))
        {
            writer.WriteArray(PrimaryBitsMapping_A_B);
            writer.WriteArray(PrimaryBitsMapping_Strength_A_B);
            writer.WriteArray(PrimaryBitsMapping_B_A);
            writer.WriteArray(PrimaryBitsMapping_Strength_B_A);
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    PrimaryBitsMapping_A_B = reader.ReadArray<int>()!;
                    PrimaryBitsMapping_B_A = reader.ReadArray<int>()!;
                    break;
                case 2:
                    PrimaryBitsMapping_A_B = reader.ReadArray<int>()!;
                    PrimaryBitsMapping_Strength_A_B = reader.ReadArray<float>()!;
                    PrimaryBitsMapping_B_A = reader.ReadArray<int>()!;
                    PrimaryBitsMapping_Strength_B_A = reader.ReadArray<float>()!;
                    break;
            }
        }
    }

    public void CalculateMapping(int[]? idealPrimaryBitsMapping_A_B)
    {
        var stopwatch = Stopwatch.StartNew();
        var r = new Random(5);        

        float globalMinEnergy_A = float.MaxValue;
        float globalMinEnergy_B = float.MaxValue;

        var random_Words_A = LanguageDiscreteEmbeddings_A.Words.Take(WordsCount).ToArray();
        var random_Words_B = LanguageDiscreteEmbeddings_B.Words.Take(WordsCount).ToArray();

        int batchesCount = WordsCount / BatchSize + 1;
        for (int epoch = 0; epoch < 30; epoch += 1)
        {   
            r.Shuffle(random_Words_A);            
            r.Shuffle(random_Words_B);

            bool epochChanged = false;

            for (int batchN = 0; batchN < batchesCount; batchN += 1)
            {
                _loggersSet.UserFriendlyLogger.LogInformation($"Batch started: {batchN}. Epoch: {epoch}; Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);

                int wordIndexStart = batchN * BatchSize;
                if (wordIndexStart >= WordsCount)
                    break;
                var batchWords_A = random_Words_A.AsMemory(wordIndexStart, Math.Min(BatchSize, random_Words_A.Length - wordIndexStart));
                var batchWords_B = random_Words_B.AsMemory(wordIndexStart, Math.Min(BatchSize, random_Words_B.Length - wordIndexStart));

                var batch_AverageWordEnergy_A = GetAverageWordEnergy(
                            batchWords_A,                            
                            Temp_WordPrimaryBitIndices_Collection_A,
                            Temp_PrimaryBitsEnergy_Matrix_A);
                var batch_AverageWordEnergy_B = GetAverageWordEnergy(
                            batchWords_B,
                            Temp_WordPrimaryBitIndices_Collection_B,
                            Temp_PrimaryBitsEnergy_Matrix_B);

                //float? idealMappedAverageWordEnergy = null;
                //if (idealPrimaryBitsMapping_A_B is not null)
                //    idealMappedAverageWordEnergy = GetMappedAverageWordEnergy(
                //            batchWords_A,
                //            idealPrimaryBitsMapping_A_B,
                //            Temp_WordPrimaryBitIndices_Collection_A,
                //            Temp_WordMappedPrimaryBitIndices_Collection_A,
                //            Temp_PrimaryBitsEnergy_Matrix_A,
                //            Temp_PrimaryBitsEnergy_Matrix_B
                //            );
                
                for (int subBatchN = 0; subBatchN < 50; subBatchN += 1)
                {
                    int allBitsTest_TotalSwapsCount = 0;

                    int[] toOptimize_PrimaryBitIndices_A = new int[PrimaryBitsMapping_B_A.Length];        
                    for (int i = 0; i < PrimaryBitsMapping_B_A.Length; i += 1)
                    {
                        toOptimize_PrimaryBitIndices_A[i] = i;
                    }
                    r.Shuffle(toOptimize_PrimaryBitIndices_A);

                    int[] toOptimize_PrimaryBitIndices_B = new int[PrimaryBitsMapping_A_B.Length];
                    for (int i = 0; i < PrimaryBitsMapping_A_B.Length; i += 1)
                    {
                        toOptimize_PrimaryBitIndices_B[i] = i;
                    }
                    r.Shuffle(toOptimize_PrimaryBitIndices_B);

                    if (batchN % 2 == 0)
                    {
                        for (int i = 0; i < toOptimize_PrimaryBitIndices_A.Length; i += 1)
                        {
                            int swapsCount = Optimize_WordsBatch_OnePrimaryBitIndex(
                                toOptimize_PrimaryBitIndices_A[i],
                                batchWords_A,
                                PrimaryBitsMapping_A_B,
                                PrimaryBitsMapping_Strength_A_B,
                                PrimaryBitsMapping_B_A,
                                PrimaryBitsMapping_Strength_B_A,
                                Temp_WordPrimaryBitIndices_Collection_A,
                                Temp_WordMappedPrimaryBitIndices_Collection_A,
                                batch_AverageWordEnergy_A,
                                batch_AverageWordEnergy_B,
                                Temp_PrimaryBitsEnergy_Matrix_A,
                                Temp_PrimaryBitsEnergy_Matrix_B,
                                ref globalMinEnergy_B
                                );

                            if (swapsCount > 0)
                                epochChanged = true;

                            allBitsTest_TotalSwapsCount += swapsCount;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < toOptimize_PrimaryBitIndices_B.Length; i += 1)
                        {
                            int swapsCount = Optimize_WordsBatch_OnePrimaryBitIndex(
                                toOptimize_PrimaryBitIndices_B[i],
                                batchWords_B,
                                PrimaryBitsMapping_B_A,
                                PrimaryBitsMapping_Strength_B_A,
                                PrimaryBitsMapping_A_B,
                                PrimaryBitsMapping_Strength_A_B,
                                Temp_WordPrimaryBitIndices_Collection_B,
                                Temp_WordMappedPrimaryBitIndices_Collection_B,
                                batch_AverageWordEnergy_B,
                                batch_AverageWordEnergy_A,
                                Temp_PrimaryBitsEnergy_Matrix_B,
                                Temp_PrimaryBitsEnergy_Matrix_A,
                                ref globalMinEnergy_A
                                );

                            if (swapsCount > 0)
                                epochChanged = true;

                            allBitsTest_TotalSwapsCount += swapsCount;
                        }
                    }                        

                    float mappedAverageWordEnergy = GetMappedAverageWordEnergy(
                            batchWords_A,
                            PrimaryBitsMapping_A_B,
                            Temp_WordPrimaryBitIndices_Collection_A,
                            Temp_WordMappedPrimaryBitIndices_Collection_A,
                            Temp_PrimaryBitsEnergy_Matrix_A,
                            Temp_PrimaryBitsEnergy_Matrix_B
                            );

                    _loggersSet.UserFriendlyLogger.LogInformation(
                        $"CalculateMapping. AverageMappedWordEnergy: {mappedAverageWordEnergy} (ideal: {batch_AverageWordEnergy_B}); Количество перестановок за 1 проход по всем битам: {allBitsTest_TotalSwapsCount}; Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);

                    if (allBitsTest_TotalSwapsCount == 0)
                        break;
                }                

                //if (iterationN % 10 == 0)
                {
                    _loggersSet.UserFriendlyLogger.LogInformation($"Batch done: {batchN}. Epoch: {epoch}; Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);
                }
            }

            _loggersSet.UserFriendlyLogger.LogInformation($"Epoch done: {epoch}. Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);
            if (!epochChanged)
                break;
        }

        stopwatch.Stop();
        _loggersSet.UserFriendlyLogger.LogInformation("CalculateMapping totally done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    public void CalculateMapping_Test(int[]? idealPrimaryBitsMapping_A_B)
    {
        for (int sourceClusterIndex = 0; sourceClusterIndex < LanguageDiscreteEmbeddings_A.ClusterInfos.Count; sourceClusterIndex += 1)
        {
            var sourceClusterInfo = LanguageDiscreteEmbeddings_A.ClusterInfos[sourceClusterIndex];            

            // Ищем позицию B с максимальным весом среди неиспользованных
            float minEnergy = float.MaxValue;
            int selected = -1;
            for (int targetClusterIndex = 0; targetClusterIndex < LanguageDiscreteEmbeddings_B.ClusterInfos.Count; targetClusterIndex += 1)
            {
                var targetClusterInfo = LanguageDiscreteEmbeddings_B.ClusterInfos[targetClusterIndex];
                if (targetClusterInfo is null)
                    continue;

                float energy = ModelHelper.GetEnergy(sourceClusterInfo.CentroidOldVectorNormalized_Mapped!, targetClusterInfo.CentroidOldVectorNormalized);
                if (energy < minEnergy)
                {
                    minEnergy = energy;
                    selected = targetClusterIndex;
                }
            }
            if (selected != -1)
            {
                PrimaryBitsMapping_A_B[sourceClusterInfo.HashProjectionIndex] = LanguageDiscreteEmbeddings_B.ClusterInfos[selected].HashProjectionIndex;
            }
        }
    }

    private static int Optimize_WordsBatch_OnePrimaryBitIndex(
        int toOptimize_PrimaryBitIndex_A,                
        Memory<Word> batchWords_A,
        int[] primaryBitsMapping_A_B,
        float[] primaryBitsMapping_Strength_A_B,
        int[] primaryBitsMapping_B_A,
        float[] primaryBitsMapping_Strength_B_A,
        int[][] wordPrimaryBitIndices_Collection_A,
        int[][] wordMappedPrimaryBitIndices_Collection_A,        
        float batch_AverageWordEnergy_A,
        float batch_AverageWordEnergy_B,
        MatrixFloat primaryBitsEnergy_Matrix_A,
        MatrixFloat primaryBitsEnergy_Matrix_B,
        ref float globalMinEnergy)
    {
        if (primaryBitsMapping_Strength_A_B[toOptimize_PrimaryBitIndex_A] > 0.5f)
            return 0;

        int original_ToOptimize_PrimaryBitIndex_B = primaryBitsMapping_A_B[toOptimize_PrimaryBitIndex_A];

        float minEnergy = float.MaxValue;
        int minEnergy_Test_PrimaryBitIndex_B = 0;

        for (int test_PrimaryBitIndex_B = 0; test_PrimaryBitIndex_B < primaryBitsMapping_A_B.Length; test_PrimaryBitIndex_B += 1)
        {
            if (primaryBitsMapping_Strength_B_A[test_PrimaryBitIndex_B] > 0.5f)
                continue;

            int old_Test_PrimaryBitIndex_A = primaryBitsMapping_B_A[test_PrimaryBitIndex_B];
            int old_ToOptimize_PrimaryBitIndex_B = primaryBitsMapping_A_B[toOptimize_PrimaryBitIndex_A];
            primaryBitsMapping_A_B[toOptimize_PrimaryBitIndex_A] = test_PrimaryBitIndex_B;
            primaryBitsMapping_A_B[old_Test_PrimaryBitIndex_A] = old_ToOptimize_PrimaryBitIndex_B;
            primaryBitsMapping_B_A[test_PrimaryBitIndex_B] = toOptimize_PrimaryBitIndex_A;
            primaryBitsMapping_B_A[old_ToOptimize_PrimaryBitIndex_B] = old_Test_PrimaryBitIndex_A;

            long energySumLong = 0;

            Parallel.For(
                0, // начальный индекс (включительно)
                batchWords_A.Length, // конечный индекс (не включительно)
                () => 0L, // Функция инициализации локальной переменной для каждого потока: localSum = 0
                (batchWordsIndex, loopState, localEnergyLong) => // Основной делегат, исполняемый в параллельном потоке
                {   
                    int wordIndex_A = batchWords_A.Span[batchWordsIndex].Index;
                    int[] wordPrimaryBitIndices = wordPrimaryBitIndices_Collection_A[wordIndex_A];
                    int[] wordMappedPrimaryBitIndices = wordMappedPrimaryBitIndices_Collection_A[wordIndex_A];

                    for (int i = 0; i < wordPrimaryBitIndices.Length; i += 1)
                    {
                        int bitIndex = wordPrimaryBitIndices[i];
                        wordMappedPrimaryBitIndices[i] = primaryBitsMapping_A_B[bitIndex];
                    }

                    //float wordEnergy = ModelHelper.GetEnergy(wordPrimaryBitIndices, primaryBitsEnergy_Matrix_A);
                    //float mappedWordEnergy = ModelHelper.GetEnergy(wordMappedPrimaryBitIndices, primaryBitsEnergy_Matrix_B);
                    //float e = (wordEnergy - mappedWordEnergy);
                    float e = ModelHelper.GetEnergy(wordMappedPrimaryBitIndices, primaryBitsEnergy_Matrix_B);
                    e = e * e;

                    // Добавляем к локальной сумме результат текущей итерации
                    localEnergyLong += (long)(e * 10000);                    
                    
                    return localEnergyLong;
                },
                // Итоговая агрегация локальных сумм каждого потока
                localEnergyLong =>
                {
                    Interlocked.Add(ref energySumLong, localEnergyLong);                    
                });
            float energySum = (float)energySumLong / (float)10000;
            float energy = energySum / batchWords_A.Length;

            if (energy < minEnergy)
            {
                minEnergy = energy;
                minEnergy_Test_PrimaryBitIndex_B = test_PrimaryBitIndex_B;                
            }

            primaryBitsMapping_A_B[toOptimize_PrimaryBitIndex_A] = old_ToOptimize_PrimaryBitIndex_B;
            primaryBitsMapping_A_B[old_Test_PrimaryBitIndex_A] = test_PrimaryBitIndex_B;
            primaryBitsMapping_B_A[test_PrimaryBitIndex_B] = old_Test_PrimaryBitIndex_A;
            primaryBitsMapping_B_A[old_ToOptimize_PrimaryBitIndex_B] = toOptimize_PrimaryBitIndex_A;
        }
        
        if (minEnergy_Test_PrimaryBitIndex_B != original_ToOptimize_PrimaryBitIndex_B) // minEnergy < globalMinEnergy  && 
        {
            globalMinEnergy = minEnergy;

            int old_Test_PrimaryBitIndex_A = primaryBitsMapping_B_A[minEnergy_Test_PrimaryBitIndex_B];
            int old_ToOptimize_PrimaryBitIndex_B = primaryBitsMapping_A_B[toOptimize_PrimaryBitIndex_A];
            primaryBitsMapping_A_B[toOptimize_PrimaryBitIndex_A] = minEnergy_Test_PrimaryBitIndex_B;
            primaryBitsMapping_A_B[old_Test_PrimaryBitIndex_A] = old_ToOptimize_PrimaryBitIndex_B;
            primaryBitsMapping_B_A[minEnergy_Test_PrimaryBitIndex_B] = toOptimize_PrimaryBitIndex_A;
            primaryBitsMapping_B_A[old_ToOptimize_PrimaryBitIndex_B] = old_Test_PrimaryBitIndex_A;

            return 1;
        }
        else
        {
            return 0;
        }
    }

    private static float GetMappedAverageWordEnergy(
        Memory<Word> batchWords_A,
        int[] primaryBitsMapping_A_B,
        int[][] wordPrimaryBitIndices_Collection_A,
        int[][] wordMappedPrimaryBitIndices_Collection_A,
        MatrixFloat primaryBitsEnergy_Matrix_A,
        MatrixFloat primaryBitsEnergy_Matrix_B)
    {
        long energySumLong = 0;

        Parallel.For(
            0, // начальный индекс (включительно)
            batchWords_A.Length, // конечный индекс (не включительно)
            () => 0L, // Функция инициализации локальной переменной для каждого потока: localSum = 0
            (batchWordsIndex, loopState, localEnergyLong) => // Основной делегат, исполняемый в параллельном потоке
            {
                int wordIndex_A = batchWords_A.Span[batchWordsIndex].Index;
                int[] wordPrimaryBitIndices = wordPrimaryBitIndices_Collection_A[wordIndex_A];
                int[] wordMappedPrimaryBitIndices = wordMappedPrimaryBitIndices_Collection_A[wordIndex_A];

                for (int i = 0; i < wordPrimaryBitIndices.Length; i += 1)
                {
                    int bitIndex = wordPrimaryBitIndices[i];
                    wordMappedPrimaryBitIndices[i] = primaryBitsMapping_A_B[bitIndex];
                }
                
                var wordMappedEnergy = ModelHelper.GetEnergy(wordMappedPrimaryBitIndices, primaryBitsEnergy_Matrix_B);

                // Добавляем к локальной сумме результат текущей итерации
                localEnergyLong += (long)(wordMappedEnergy * 10000);

                return localEnergyLong;
            },
            // Итоговая агрегация локальных сумм каждого потока
            localEnergyLong =>
            {
                Interlocked.Add(ref energySumLong, localEnergyLong);
            });

        float energySum = (float)energySumLong / (float)10000;

        return energySum / batchWords_A.Length;
    }

    private float GetAverageWordEnergy(Memory<Word> batchWords, int[][] wordPrimaryBitIndices_Collection, MatrixFloat primaryBitsEnergy_Matrix)
    {
        long energySumLong = 0;

        Parallel.For(
            0, // начальный индекс (включительно)
            batchWords.Length, // конечный индекс (не включительно)
            () => 0L, // Функция инициализации локальной переменной для каждого потока: localSum = 0
            (batchWordsIndex, loopState, localEnergyLong) => // Основной делегат, исполняемый в параллельном потоке
            {
                int wordIndex = batchWords.Span[batchWordsIndex].Index;
                int[] wordPrimaryBitIndices = wordPrimaryBitIndices_Collection[wordIndex];

                var wordEnergy = ModelHelper.GetEnergy(wordPrimaryBitIndices, primaryBitsEnergy_Matrix);                                

                // Добавляем к локальной сумме результат текущей итерации
                localEnergyLong += (long)(wordEnergy * 10000);

                return localEnergyLong;
            },
            // Итоговая агрегация локальных сумм каждого потока
            localEnergyLong =>
            {
                Interlocked.Add(ref energySumLong, localEnergyLong);
            });

        float energySum = (float)energySumLong / (float)10000;

        return energySum / batchWords.Length;
    }
}
