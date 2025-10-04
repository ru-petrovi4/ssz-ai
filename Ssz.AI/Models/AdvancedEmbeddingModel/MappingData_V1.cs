using MathNet.Numerics;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public class MappingData_V1 : ISerializableModelObject
{
    private ILoggersSet _loggersSet;
    public LanguageInfo LanguageInfo_RU;
    public LanguageInfo LanguageInfo_EN;

    /// <summary>
    ///     Index RU -> Index EN
    /// </summary>
    public int[] Mapping_RU_EN = null!;
    /// <summary>
    ///     Index EN -> Index RU
    /// </summary>
    public int[] Mapping_EN_RU = null!;

    public MatrixFloat Temp_ProxBits_RU = new();
    /// <summary>
    ///     Word for each bit.
    /// </summary>
    public Word[] Temp_BitWords_RU = null!;
    /// <summary>
    ///     [Word, [Bit Index]]
    /// </summary>
    public int[][] Temp_WordBitIndices_Collection_RU = null!;
    /// <summary>
    ///     [Word, [Mapped Bit Index]]
    /// </summary>
    public int[][] Temp_WordMappedBitIndices_Collection_RU = null!;

    public MatrixFloat Temp_ProxBits_EN = new();
    /// <summary>
    ///     Word for each bit.
    /// </summary>
    public Word[] Temp_BitWords_EN = null!;
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

    public MappingData_V1(ILoggersSet loggersSet, LanguageInfo languageInfo_RU, LanguageInfo languageInfo_EN)
    {
        _loggersSet = loggersSet;
        LanguageInfo_RU = languageInfo_RU;
        LanguageInfo_EN = languageInfo_EN;        
    }

    public void GenerateOwnedData(int vectorLength)
    {
        Mapping_RU_EN = new int[vectorLength];
        Mapping_EN_RU = new int[vectorLength];
        bool[] isMapped_EN = new bool[vectorLength];

        var r = new Random(1);
        
        for (int bitIndex_RU = 0; bitIndex_RU < vectorLength; bitIndex_RU += 1)
        {
            for (; ; )
            {
                int bitIndex_EN = r.Next(vectorLength);                
                if (isMapped_EN[bitIndex_EN])
                    continue;

                Mapping_RU_EN[bitIndex_RU] = bitIndex_EN;
                Mapping_EN_RU[bitIndex_EN] = bitIndex_RU;
                isMapped_EN[bitIndex_EN] = true;
                break;
            }
        }
    }

    public void Prepare()
    {
        Temp_ProxBits_RU = new MatrixFloat(Mapping_RU_EN.Length, Mapping_RU_EN.Length);
        Temp_BitWords_RU = new Word[Mapping_RU_EN.Length];
        var h = LanguageInfo_RU.Clusterization_AlgorithmData.PrimaryWords.Select(w => w.Index).ToHashSet();
        foreach (var primaryWord in LanguageInfo_RU.Clusterization_AlgorithmData.PrimaryWords)
        {
            int prjectionIndex = LanguageInfo_RU.ProjectionOptimization_AlgorithmData.WordsHashProjectionIndices[primaryWord.Index];
            Temp_BitWords_RU[prjectionIndex] = primaryWord;
        }
        for (int i = 0; i < Mapping_RU_EN.Length; i += 1)
        {
            for (int j = 0; j < Mapping_RU_EN.Length; j += 1)
            {
                Temp_ProxBits_RU[i, j] = TensorPrimitives.CosineSimilarity(
                    LanguageInfo_RU.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_RU[i].Index],
                    LanguageInfo_RU.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_RU[j].Index]);
            }
        }
        Temp_WordBitIndices_Collection_RU = new int[LanguageInfo_RU.Words.Count][];
        Temp_WordMappedBitIndices_Collection_RU = new int[LanguageInfo_RU.Words.Count][];
        List<int> wordBitIndices = new List<int>(16);
        List<int> wordMappedBitIndices = new List<int>(16);
        for (int i = 0; i < LanguageInfo_RU.Words.Count; i += 1)
        {            
            var discreteVector_PrimaryBitsOnly = LanguageInfo_RU.DiscreteVectorsAndMatrices.DiscreteVectors_PrimaryBitsOnly[LanguageInfo_RU.Words[i].Index];
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

        Temp_ProxBits_EN = new MatrixFloat(Mapping_RU_EN.Length, Mapping_RU_EN.Length);
        Temp_BitWords_EN = new Word[Mapping_RU_EN.Length];
        foreach (var primaryWord in LanguageInfo_EN.Clusterization_AlgorithmData.PrimaryWords)
        {
            int prjectionIndex = LanguageInfo_EN.ProjectionOptimization_AlgorithmData.WordsHashProjectionIndices[primaryWord.Index];
            Temp_BitWords_EN[prjectionIndex] = primaryWord;
        }
        for (int i = 0; i < Mapping_RU_EN.Length; i += 1)
        {
            for (int j = 0; j < Mapping_RU_EN.Length; j += 1)
            {
                Temp_ProxBits_EN[i, j] = TensorPrimitives.CosineSimilarity(
                    LanguageInfo_EN.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_EN[i].Index],
                    LanguageInfo_EN.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_EN[j].Index]);
            }
        }
        Temp_WordBitIndices_Collection_EN = new int[LanguageInfo_EN.Words.Count][];
        Temp_WordMappedBitIndices_Collection_EN = new int[LanguageInfo_EN.Words.Count][];        
        for (int i = 0; i < LanguageInfo_EN.Words.Count; i += 1)
        {
            var discreteVector = LanguageInfo_EN.DiscreteVectorsAndMatrices.DiscreteVectors[LanguageInfo_EN.Words[i].Index];
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

        //Temp_EnergyOfBitCollection = new float[Mapping_RU_EN.Length];
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
            var random_Words_RU = WordsHelper.GetRandomOrderWords(LanguageInfo_RU.Words, r);
            var random_Words_EN = WordsHelper.GetRandomOrderWords(LanguageInfo_EN.Words, r);
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
            Temp_MinEnergy = Single.MaxValue;
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
            float minEnergy = Single.MaxValue;
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
