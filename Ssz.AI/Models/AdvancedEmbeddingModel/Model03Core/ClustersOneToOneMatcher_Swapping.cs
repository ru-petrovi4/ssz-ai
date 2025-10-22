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
    public LanguageDiscreteEmbeddings LanguageDiscreteEmbeddings_RU;
    public LanguageDiscreteEmbeddings LanguageDiscreteEmbeddings_EN;

    public const int BatchSize = 10000;

    public int WordsCount = 10000;

    /// <summary>
    ///     Primary Bit Index RU -> Primary Bit Index EN
    /// </summary>
    public int[] PrimaryBitsMapping_RU_EN = null!;
    /// <summary>
    ///     Primary Bit Index EN -> Primary Bit Index RU
    /// </summary>
    public int[] PrimaryBitsMapping_EN_RU = null!;

    //public torch.Tensor Temp_EnergyBits_Tensor_RU = null!;
    public MatrixFloat Temp_PrimaryBitsEnergy_Matrix_RU = null!;
    /// <summary>
    ///     [Word.Index, [Bit Index]]
    /// </summary>
    public int[][] Temp_WordPrimaryBitIndices_Collection_RU = null!;
    /// <summary>
    ///     [Word.Index, [Mapped Bit Index]]
    /// </summary>
    public int[][] Temp_WordMappedPrimaryBitIndices_Collection_RU = null!;

    //public torch.Tensor Temp_EnergyBits_Tensor_EN = null!;
    public MatrixFloat Temp_PrimaryBitsEnergy_Matrix_EN = null!;
    /// <summary>
    ///     [Word.Index, [Bit Index]]
    /// </summary>
    public int[][] Temp_WordPrimaryBitIndices_Collection_EN = null!;
    /// <summary>
    ///     [Word.Index, [Mapped Bit Index]]
    /// </summary>
    public int[][] Temp_WordMappedPrimaryBitIndices_Collection_EN = null!;

    public ClustersOneToOneMatcher_Swapping(ILoggersSet loggersSet, LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU, LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN)
    {
        _loggersSet = loggersSet;
        _device = cuda.is_available() ? CUDA : CPU;
        LanguageDiscreteEmbeddings_RU = languageDiscreteEmbeddings_RU;
        LanguageDiscreteEmbeddings_EN = languageDiscreteEmbeddings_EN;        
    }

    public void GenerateOwnedData(int clustersCount)
    {
        PrimaryBitsMapping_RU_EN = new int[clustersCount];
        PrimaryBitsMapping_EN_RU = new int[clustersCount];
        bool[] isMapped_RU_EN = new bool[clustersCount];

        var r = new Random(1);
        
        for (int bitIndex_RU = 0; bitIndex_RU < clustersCount; bitIndex_RU += 1)
        {
            for (; ; )
            {
                int bitIndex_EN = r.Next(clustersCount);                
                if (isMapped_RU_EN[bitIndex_EN])
                    continue;

                PrimaryBitsMapping_RU_EN[bitIndex_RU] = bitIndex_EN;
                PrimaryBitsMapping_EN_RU[bitIndex_EN] = bitIndex_RU;
                isMapped_RU_EN[bitIndex_EN] = true;
                break;
            }
        }
    }

    public void Prepare()
    {
        WordsCount = Math.Min(Math.Min(WordsCount, LanguageDiscreteEmbeddings_RU.Words.Count), LanguageDiscreteEmbeddings_RU.Words.Count);

        Temp_PrimaryBitsEnergy_Matrix_RU = ModelHelper.GetPrimaryBitsEnergy_Matrix(LanguageDiscreteEmbeddings_RU);
        Temp_PrimaryBitsEnergy_Matrix_EN = ModelHelper.GetPrimaryBitsEnergy_Matrix(LanguageDiscreteEmbeddings_EN);        

        Temp_WordPrimaryBitIndices_Collection_RU = new int[WordsCount][];
        Temp_WordMappedPrimaryBitIndices_Collection_RU = new int[WordsCount][];
        List<int> wordPrimaryBitIndices = new List<int>(16);
        List<int> wordMappedPrimaryBitIndices = new List<int>(16);
        for (int i = 0; i < WordsCount; i += 1)
        {            
            var discreteVector_PrimaryBitsOnly = LanguageDiscreteEmbeddings_RU.Words[i].DiscreteVector_PrimaryBitsOnly;
            wordPrimaryBitIndices.Clear();
            wordMappedPrimaryBitIndices.Clear();
            for (int j = 0; j < discreteVector_PrimaryBitsOnly.Length; j += 1)
            {
                if (discreteVector_PrimaryBitsOnly[j] > 0.5f)
                {
                    wordPrimaryBitIndices.Add(j);
                    wordMappedPrimaryBitIndices.Add(PrimaryBitsMapping_RU_EN[j]);
                }
            }
            Temp_WordPrimaryBitIndices_Collection_RU[i] = wordPrimaryBitIndices.ToArray();
            Temp_WordMappedPrimaryBitIndices_Collection_RU[i] = wordMappedPrimaryBitIndices.ToArray();
        }        
              
        Temp_WordPrimaryBitIndices_Collection_EN = new int[WordsCount][];
        Temp_WordMappedPrimaryBitIndices_Collection_EN = new int[WordsCount][];        
        for (int i = 0; i < WordsCount; i += 1)
        {
            var discreteVector_PrimaryBitsOnly = LanguageDiscreteEmbeddings_EN.Words[i].DiscreteVector_PrimaryBitsOnly;
            wordPrimaryBitIndices.Clear();
            wordMappedPrimaryBitIndices.Clear();
            for (int j = 0; j < discreteVector_PrimaryBitsOnly.Length; j += 1)
            {
                if (discreteVector_PrimaryBitsOnly[j] > 0.5f)
                {
                    wordPrimaryBitIndices.Add(j);
                    wordMappedPrimaryBitIndices.Add(PrimaryBitsMapping_EN_RU[j]);
                }
            }
            Temp_WordPrimaryBitIndices_Collection_EN[i] = wordPrimaryBitIndices.ToArray();
            Temp_WordMappedPrimaryBitIndices_Collection_EN[i] = wordMappedPrimaryBitIndices.ToArray();
        }
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteArray(PrimaryBitsMapping_RU_EN);
            writer.WriteArray(PrimaryBitsMapping_EN_RU);
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    PrimaryBitsMapping_RU_EN = reader.ReadArray<int>()!;
                    PrimaryBitsMapping_EN_RU = reader.ReadArray<int>()!;
                    break;
            }
        }
    }

    public void CalculateMapping(int[]? idealPrimaryBitsMapping_A_B)
    {
        var stopwatch = Stopwatch.StartNew();
        var r = new Random(5);        

        float error = float.MaxValue;

        int batchesCount = WordsCount / BatchSize + 1;
        for (int epoch = 0; epoch < 15; epoch += 1)
        {
            var random_Words_RU = LanguageDiscreteEmbeddings_RU.Words.Take(WordsCount).ToArray();
            r.Shuffle(random_Words_RU);
            var random_Words_EN = LanguageDiscreteEmbeddings_EN.Words.Take(WordsCount).ToArray();
            r.Shuffle(random_Words_EN);

            bool changed = false;

            for (int batchN = 0; batchN < batchesCount; batchN += 1)
            {
                _loggersSet.UserFriendlyLogger.LogInformation($"Batch started: {batchN}. Epoch: {epoch}; Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);

                int wordIndexStart = batchN * BatchSize;
                if (wordIndexStart >= WordsCount)
                    break;
                var batchWords_RU = random_Words_RU.AsMemory(wordIndexStart, Math.Min(BatchSize, random_Words_RU.Length - wordIndexStart));
                var batchWords_EN = random_Words_EN.AsMemory(wordIndexStart, Math.Min(BatchSize, random_Words_EN.Length - wordIndexStart));

                float? idealMappedAverageWordEnergy = null;
                if (idealPrimaryBitsMapping_A_B is not null)
                    idealMappedAverageWordEnergy = GetMappedAverageWordEnergy(
                            batchWords_RU,
                            idealPrimaryBitsMapping_A_B,
                            Temp_WordPrimaryBitIndices_Collection_RU,
                            Temp_WordMappedPrimaryBitIndices_Collection_RU
                            );
                
                int batchTotalSwapsCount = 0;
                for (; ; )
                {
                    batchTotalSwapsCount = 0;

                    int[] numbers = new int[PrimaryBitsMapping_EN_RU.Length]; // numbers[] — отсюда получим уникальные значения.                    
                    for (int i = 0; i < PrimaryBitsMapping_EN_RU.Length; i += 1)
                    {
                        numbers[i] = i;
                    }
                    r.Shuffle(numbers);

                    for (int i = 0; i < PrimaryBitsMapping_EN_RU.Length; i += 1)
                    {
                        (error, int swapsCount) = Optimize_WordsBatch_OnePrimaryBitIndex_A(
                            numbers[i],
                            error,                                                    
                            batchWords_RU,
                            PrimaryBitsMapping_RU_EN,
                            Temp_WordPrimaryBitIndices_Collection_RU,
                            Temp_WordMappedPrimaryBitIndices_Collection_RU,
                            PrimaryBitsMapping_EN_RU
                            );

                        if (swapsCount > 0)
                            changed = true;

                        batchTotalSwapsCount += swapsCount;

                        //OptimizeWordsBatch(
                        //    r.Next(Mapping_EN_RU.Length),
                        //    batchWords_EN,
                        //    Mapping_EN_RU,
                        //    Temp_WordBitIndices_Collection_EN,
                        //    Temp_WordMappedBitIndices_Collection_EN,
                        //    Mapping_RU_EN
                        //    );
                    }

                    float mappedAverageWordEnergy = GetMappedAverageWordEnergy(
                            batchWords_RU,
                            PrimaryBitsMapping_RU_EN,
                            Temp_WordPrimaryBitIndices_Collection_RU,
                            Temp_WordMappedPrimaryBitIndices_Collection_RU
                            );

                    _loggersSet.UserFriendlyLogger.LogInformation(
                        $"CalculateMapping. AverageMappedWordEnergy: {mappedAverageWordEnergy} (ideal: {idealMappedAverageWordEnergy ?? float.NaN}); Количество перестановок: {batchTotalSwapsCount}; Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);

                    if (batchTotalSwapsCount == 0)
                        break;
                }                

                //if (iterationN % 10 == 0)
                {
                    _loggersSet.UserFriendlyLogger.LogInformation($"Batch done: {batchN}. Epoch: {epoch}; Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);
                }
            }

            _loggersSet.UserFriendlyLogger.LogInformation($"Epoch done: {epoch}. Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);
            if (!changed)
                break;
        }

        stopwatch.Stop();
        _loggersSet.UserFriendlyLogger.LogInformation("CalculateMapping totally done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }    

    private (float error, int swapsCount) Optimize_WordsBatch_OnePrimaryBitIndex_A(
        int toOptimize_PrimaryBitIndex_A,
        float prevError,            
        Memory<Word> batchWords_A,
        int[] primaryBitsMapping_A_B,
        int[][] wordPrimaryBitIndices_Collection_A,
        int[][] wordMappedPrimaryBitIndices_Collection_A,
        int[] primaryBitsMapping_B_A)
    { 
        float minError = float.MaxValue;

        int minEnergyTest_PrimaryBitIndex_B = 0;
        for (int test_PrimaryBitIndex_B = 0; test_PrimaryBitIndex_B < primaryBitsMapping_A_B.Length; test_PrimaryBitIndex_B += 1)
        {
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

                    // Добавляем к локальной сумме результат текущей итерации
                    localEnergyLong += (long)(ModelHelper.GetEnergy(wordMappedPrimaryBitIndices, Temp_PrimaryBitsEnergy_Matrix_EN) * 10000);                    
                    
                    return localEnergyLong;
                },
                // Итоговая агрегация локальных сумм каждого потока
                localEnergyLong =>
                {
                    Interlocked.Add(ref energySumLong, localEnergyLong);                    
                });
            float energySum = (float)energySumLong / (float)10000;
            float mappedAverageWordEnergy = energySum / batchWords_A.Length;

            float error;
            // TEMPCODE
            //if (idealMappedAverageWordEnergy is not null)
            //{
            //    error = idealMappedAverageWordEnergy.Value - mappedAverageWordEnergy;
            //    error = error * error;
            //}
            //else
            {
                error = mappedAverageWordEnergy;
            }

            if (error < minError)
            {
                minError = error;
                minEnergyTest_PrimaryBitIndex_B = test_PrimaryBitIndex_B;                
            }

            primaryBitsMapping_A_B[toOptimize_PrimaryBitIndex_A] = old_ToOptimize_PrimaryBitIndex_B;
            primaryBitsMapping_A_B[old_Test_PrimaryBitIndex_A] = test_PrimaryBitIndex_B;
            primaryBitsMapping_B_A[test_PrimaryBitIndex_B] = old_Test_PrimaryBitIndex_A;
            primaryBitsMapping_B_A[old_ToOptimize_PrimaryBitIndex_B] = toOptimize_PrimaryBitIndex_A;
        }

        if (minError < prevError)
        {
            int old_Test_PrimaryBitIndex_A = primaryBitsMapping_B_A[minEnergyTest_PrimaryBitIndex_B];
            int old_ToOptimize_PrimaryBitIndex_B = primaryBitsMapping_A_B[toOptimize_PrimaryBitIndex_A];
            primaryBitsMapping_A_B[toOptimize_PrimaryBitIndex_A] = minEnergyTest_PrimaryBitIndex_B;
            primaryBitsMapping_A_B[old_Test_PrimaryBitIndex_A] = old_ToOptimize_PrimaryBitIndex_B;
            primaryBitsMapping_B_A[minEnergyTest_PrimaryBitIndex_B] = toOptimize_PrimaryBitIndex_A;
            primaryBitsMapping_B_A[old_ToOptimize_PrimaryBitIndex_B] = old_Test_PrimaryBitIndex_A;

            return (minError, 1);
        }
        else
        {
            return (prevError, 0);
        }
    }

    private float GetMappedAverageWordEnergy(
        Memory<Word> batchWords_A,
        int[] primaryBitsMapping_A_B,
        int[][] wordPrimaryBitIndices_Collection_A,
        int[][] wordMappedPrimaryBitIndices_Collection_A)
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

                // Добавляем к локальной сумме результат текущей итерации
                localEnergyLong += (long)(ModelHelper.GetEnergy(wordMappedPrimaryBitIndices, Temp_PrimaryBitsEnergy_Matrix_EN) * 10000);

                return localEnergyLong;
            },
            // Итоговая агрегация локальных сумм каждого потока
            localEnergyLong =>
            {
                Interlocked.Add(ref energySumLong, localEnergyLong);
            });

        float energySum = (float)energySumLong / (float)10000;

        float mappedAverageWordEnergy = energySum / batchWords_A.Length;

        return mappedAverageWordEnergy;
    }
}
