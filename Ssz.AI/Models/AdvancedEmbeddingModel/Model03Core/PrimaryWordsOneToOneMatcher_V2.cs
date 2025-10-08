using MathNet.Numerics;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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

    public int WordsCount;

    /// <summary>
    ///     Index RU -> Index EN
    /// </summary>
    public int[] Mapping_RU_EN = null!;
    /// <summary>
    ///     Index EN -> Index RU
    /// </summary>
    public int[] Mapping_EN_RU = null!;

    public torch.Tensor Temp_ProxBits_Tensor_RU = null!;
    public MatrixFloat Temp_ProxBits_RU = null!;
    /// <summary>
    ///     [Word, [Bit Index]]
    /// </summary>
    public int[][] Temp_WordBitIndices_Collection_RU = null!;
    /// <summary>
    ///     [Word, [Mapped Bit Index]]
    /// </summary>
    public int[][] Temp_WordMappedBitIndices_Collection_RU = null!;

    public torch.Tensor Temp_ProxBits_Tensor_EN = null!;
    public MatrixFloat Temp_ProxBits_EN = null!;
    /// <summary>
    ///     [Word, [Bit Index]]
    /// </summary>
    public int[][] Temp_WordBitIndices_Collection_EN = null!;
    /// <summary>
    ///     [Word, [Mapped Bit Index]]
    /// </summary>
    public int[][] Temp_WordMappedBitIndices_Collection_EN = null!;

    public float Temp_MinEnergy;

    public float Temp_ForWordChangesCount;

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
        WordsCount = 10000;
        using (var disposeScope = torch.NewDisposeScope())
        {
            var primaryBitsOnlyEmbeddingsTensor_RU = LanguageDiscreteEmbeddings_RU.GetDiscrete_PrimaryBitsOnlyEmbeddingsTensor(WordsCount, _device);
            Temp_ProxBits_Tensor_RU = torch.mm(primaryBitsOnlyEmbeddingsTensor_RU.t(), primaryBitsOnlyEmbeddingsTensor_RU).to(CPU).DetachFromDisposeScope();
            Temp_ProxBits_RU = MatrixFloat.FromTensor(Temp_ProxBits_Tensor_RU);

            var primaryBitsOnlyEmbeddingsTensor_EN = LanguageDiscreteEmbeddings_EN.GetDiscrete_PrimaryBitsOnlyEmbeddingsTensor(WordsCount, _device);
            Temp_ProxBits_Tensor_EN = torch.mm(primaryBitsOnlyEmbeddingsTensor_EN.t(), primaryBitsOnlyEmbeddingsTensor_EN).to(CPU).DetachFromDisposeScope();
            Temp_ProxBits_EN = MatrixFloat.FromTensor(Temp_ProxBits_Tensor_EN);
        }        

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

        for (int i = 0; i < 3 * 3600 * 10; i += 1)
        {
            var random_Words_RU = WordsHelper.GetRandomOrderWords(LanguageDiscreteEmbeddings_RU.Words, WordsCount, r);
            var random_Words_EN = WordsHelper.GetRandomOrderWords(LanguageDiscreteEmbeddings_EN.Words, WordsCount, r);
            for (int randomWordsIndex = 0; randomWordsIndex < 10000; randomWordsIndex += 1)
            {   
                OptimizeWord_RU(random_Words_RU[randomWordsIndex].Index, r);
                OptimizeWord_EN(random_Words_EN[randomWordsIndex].Index, r);

                if (randomWordsIndex % 1000 == 0)
                {
                    _loggersSet.UserFriendlyLogger.LogInformation($"CalculateMapping iteration; randomWordsIndex: {i}:{randomWordsIndex} done. MinEnergy: {Temp_MinEnergy}; ForWordChangesCount: {Temp_ForWordChangesCount}; Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);                    
                }
            }
        }

        stopwatch.Stop();
        _loggersSet.UserFriendlyLogger.LogInformation("CalculateMapping totally done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    private void OptimizeWord_RU(int wordIndex_RU, Random r)
    {
        int[] wordBitIndices = Temp_WordBitIndices_Collection_RU[wordIndex_RU];
        int[] wordMappedBitIndices = Temp_WordMappedBitIndices_Collection_RU[wordIndex_RU];

        Temp_ForWordChangesCount = 0;
        for (int i = 0; i < wordBitIndices.Length; i += 1)
        {
            int bitIndex = wordBitIndices[i];
            wordMappedBitIndices[i] = Mapping_RU_EN[bitIndex];
        }
        int toOptimize_i = r.Next(wordBitIndices.Length);
        //for (int toOptimize_i = 0; toOptimize_i < wordBitIndices.Length; toOptimize_i += 1)
        {             
            Temp_MinEnergy = float.MaxValue;
            int originalBitIndex = wordMappedBitIndices[toOptimize_i];
            int minEnergy_BitIndex_EN = 0;
            for (int test_BitIndex_EN = 0; test_BitIndex_EN < Mapping_RU_EN.Length; test_BitIndex_EN += 1)
            {
                wordMappedBitIndices[toOptimize_i] = test_BitIndex_EN;
                float energy = GetEnergy(toOptimize_i, wordMappedBitIndices, Temp_ProxBits_EN);
                if (energy < Temp_MinEnergy)
                {
                    Temp_MinEnergy = energy;
                    minEnergy_BitIndex_EN = test_BitIndex_EN;
                }
            }
            wordMappedBitIndices[toOptimize_i] = minEnergy_BitIndex_EN;

            if (originalBitIndex != minEnergy_BitIndex_EN)
            {
                Temp_ForWordChangesCount += 1;
                int bitIndexToOptimize_RU = wordBitIndices[toOptimize_i];
                int old_MinEnergy_BitIndex_RU = Mapping_EN_RU[minEnergy_BitIndex_EN];
                int old_MinEnergy_BitIndex_EN = Mapping_RU_EN[bitIndexToOptimize_RU];
                Mapping_RU_EN[bitIndexToOptimize_RU] = minEnergy_BitIndex_EN;
                Mapping_RU_EN[old_MinEnergy_BitIndex_RU] = old_MinEnergy_BitIndex_EN;
                Mapping_EN_RU[minEnergy_BitIndex_EN] = bitIndexToOptimize_RU;
                Mapping_EN_RU[old_MinEnergy_BitIndex_EN] = old_MinEnergy_BitIndex_RU;
            }
        }
    }

    private void OptimizeWord_EN(int wordIndex_EN, Random r)
    {
        int[] wordBitIndices = Temp_WordBitIndices_Collection_EN[wordIndex_EN];
        int[] wordMappedBitIndices = Temp_WordMappedBitIndices_Collection_EN[wordIndex_EN];

        for (int i = 0; i < wordBitIndices.Length; i += 1)
        {
            int bitIndex = wordBitIndices[i];
            wordMappedBitIndices[i] = Mapping_EN_RU[bitIndex];
        }
        int toOptimize_i = r.Next(wordBitIndices.Length);
        //for (int toOptimize_i = 0; toOptimize_i < wordBitIndices.Length; toOptimize_i += 1)
        {   
            float minEnergy = float.MaxValue;
            int originalBitIndex = wordMappedBitIndices[toOptimize_i];
            int minEnergy_BitIndex_RU = 0;
            for (int test_BitIndex_RU = 0; test_BitIndex_RU < Mapping_EN_RU.Length; test_BitIndex_RU += 1)
            {
                wordMappedBitIndices[toOptimize_i] = test_BitIndex_RU;
                float energy = GetEnergy(toOptimize_i, wordMappedBitIndices, Temp_ProxBits_RU);
                if (energy < minEnergy)
                {
                    minEnergy = energy;
                    minEnergy_BitIndex_RU = test_BitIndex_RU;
                }
            }
            wordMappedBitIndices[toOptimize_i] = minEnergy_BitIndex_RU;

            if (originalBitIndex != minEnergy_BitIndex_RU)
            {
                int bitIndexToOptimize_EN = wordBitIndices[toOptimize_i];
                int old_MinEnergy_BitIndex_EN = Mapping_RU_EN[minEnergy_BitIndex_RU];
                int old_MinEnergy_BitIndex_RU = Mapping_EN_RU[bitIndexToOptimize_EN];
                Mapping_EN_RU[bitIndexToOptimize_EN] = minEnergy_BitIndex_RU;
                Mapping_EN_RU[old_MinEnergy_BitIndex_EN] = old_MinEnergy_BitIndex_RU;
                Mapping_RU_EN[minEnergy_BitIndex_RU] = bitIndexToOptimize_EN;
                Mapping_RU_EN[old_MinEnergy_BitIndex_RU] = old_MinEnergy_BitIndex_EN;
            }                
        }
    }

    private float GetEnergy(int toOptimize_i, int[] wordMappedBitIndices, MatrixFloat proxBits)
    {
        int toOptimizeBitIndex = wordMappedBitIndices[toOptimize_i];
        float energy = 0;
        for (int i = 0; i < wordMappedBitIndices.Length; i += 1)
        {
            if (i == toOptimize_i)
                continue;
            int mappedBitIndex = wordMappedBitIndices[i];
            if (mappedBitIndex == toOptimizeBitIndex)
            {
                return 1000.0f; // Collision
            }
            else
            {
                energy -= proxBits[toOptimizeBitIndex, mappedBitIndex];
            }   
        }
        return energy;
    }
}
