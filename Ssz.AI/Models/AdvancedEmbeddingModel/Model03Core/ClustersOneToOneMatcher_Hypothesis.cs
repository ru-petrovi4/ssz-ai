using MathNet.Numerics;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.Utils;
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
/// Сопоставление на основе гипотез
/// </summary>
public class ClustersOneToOneMatcher_Hypothesis : ISerializableModelObject
{
    private IUserFriendlyLogger _userFriendlyLogger;   
    private Device _device;
    public LanguageDiscreteEmbeddings LanguageDiscreteEmbeddings_RU;
    public LanguageDiscreteEmbeddings LanguageDiscreteEmbeddings_EN;
    public int VectorLength;

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

    //public torch.Tensor Temp_EnergyBits_Tensor_EN = null!;
    public MatrixFloat Temp_PrimaryBitsEnergy_Matrix_EN = null!;

    public PrimaryBitsNearest Temp_PrimaryBitsNearestA = null!;

    public PrimaryBitsNearest Temp_PrimaryBitsNearestB = null!;

    /// <summary>
    ///     [Word.Index, [Bit Index]]
    /// </summary>
    public int[][] Temp_WordPrimaryBitIndices_Collection_RU = null!;

    /// <summary>
    ///     [Word.Index, [Bit Index]]
    /// </summary>
    public int[][] Temp_WordPrimaryBitIndices_Collection_EN = null!;

    /// <summary>
    ///     Таблица накопления весов для гипотез: [позицияA, позицияB] -> float
    /// </summary>
    public MatrixFloat Temp_HypothesisSupport = null!;

    public const int WordsCount = 10000;

    /// <summary>
    /// Количество ближайших для подкрепления
    /// </summary>
    public const int NearestCount = 32;

    public ClustersOneToOneMatcher_Hypothesis(
        IUserFriendlyLogger userFriendlyLogger, 
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU, 
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN)
    {
        _userFriendlyLogger = userFriendlyLogger;
        _device = cuda.is_available() ? CUDA : CPU;
        LanguageDiscreteEmbeddings_RU = languageDiscreteEmbeddings_RU;
        LanguageDiscreteEmbeddings_EN = languageDiscreteEmbeddings_EN;
        VectorLength = Model01.Constants.DiscreteVectorLength;        
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
        Temp_PrimaryBitsEnergy_Matrix_RU = ModelHelper.GetPrimaryBitsEnergy_Matrix(LanguageDiscreteEmbeddings_RU);
        Temp_PrimaryBitsEnergy_Matrix_EN = ModelHelper.GetPrimaryBitsEnergy_Matrix(LanguageDiscreteEmbeddings_EN);

        Temp_HypothesisSupport = new MatrixFloat(VectorLength, VectorLength);

        Temp_PrimaryBitsNearestA = BuildPrimaryBitsNearest(Temp_PrimaryBitsEnergy_Matrix_RU);
        Temp_PrimaryBitsNearestB = BuildPrimaryBitsNearest(Temp_PrimaryBitsEnergy_Matrix_EN);

        Temp_WordPrimaryBitIndices_Collection_RU = BuildWordPrimaryBitIndices_Collection(LanguageDiscreteEmbeddings_RU);
        Temp_WordPrimaryBitIndices_Collection_EN = BuildWordPrimaryBitIndices_Collection(LanguageDiscreteEmbeddings_EN);        
    }    

    /// <summary>
    ///     Найти ближайшие позиции для всех (по строкам)
    /// </summary>
    /// <param name="primaryBitsEnergy_Matrix"></param>
    /// <returns></returns>
    public PrimaryBitsNearest BuildPrimaryBitsNearest(MatrixFloat primaryBitsEnergy_Matrix)
    {
        var nearestArray = new FastList<int>[VectorLength];
        for (int i = 0; i < primaryBitsEnergy_Matrix.Dimensions[0]; i += 1)
        {
            var list = new List<(int idx, float val)>(VectorLength);
            for (int j = 0; j < primaryBitsEnergy_Matrix.Dimensions[1]; j += 1)
            {
                if (i != j)
                    list.Add((j, primaryBitsEnergy_Matrix[i, j]));
            }
            // Берём NearestCount ближайших
            nearestArray[i] = new FastList<int>(list.OrderBy(it => it.val).Take(NearestCount).Select(x => x.idx).ToArray());
        }
        return new PrimaryBitsNearest()
        {
            Array = nearestArray
        };
    }

    public int[][] BuildWordPrimaryBitIndices_Collection(LanguageDiscreteEmbeddings languageDiscreteEmbeddings)
    {
        var wordPrimaryBitIndices_Collection = new int[WordsCount][];
        List<int> wordPrimaryBitIndices = new List<int>(16);
        for (int i = 0; i < WordsCount; i += 1)
        {
            var discreteVector_PrimaryBitsOnly = languageDiscreteEmbeddings.Words[i].DiscreteVector_PrimaryBitsOnly;
            wordPrimaryBitIndices.Clear();
            for (int j = 0; j < discreteVector_PrimaryBitsOnly.Length; j += 1)
            {
                if (discreteVector_PrimaryBitsOnly[j] > 0.5f)
                {
                    wordPrimaryBitIndices.Add(j);
                }
            }
            wordPrimaryBitIndices_Collection[i] = wordPrimaryBitIndices.ToArray();
        }
        return wordPrimaryBitIndices_Collection;
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

    /// <summary>
    /// Подкрепление гипотез на примерах
    /// </summary>
    /// <param name="setA"></param>
    /// <param name="setB"></param>
    public void SupportHypotheses_V1()
    {
        var primaryBitsNearestB = Temp_PrimaryBitsNearestB.Array;
        
        for (int i = 0; i < WordsCount; i += 1)
        {
            if (i % 100 == 0)
                _userFriendlyLogger.LogInformation($"A i = {i}");

            var vecA = Temp_WordPrimaryBitIndices_Collection_RU[i];
            // Индексы позиций с единицей
            for (int idxA = 0; idxA < vecA.Length; idxA += 1)
            {
                for (int idxB = 0; idxB < VectorLength; idxB += 1)
                {
                    // Гипотеза: idxA → idxB
                    //Temp_HypothesisSupport[vecA[idxA], idxB] += 1.0f;

                    // Подкрепляем также все пары ближайших
                    foreach (var nearB in primaryBitsNearestB[idxB].Items)
                    {
                        for (int idxA2 = 0; idxA2 < vecA.Length; idxA2 += 1)
                        {
                            if (idxA2 != idxA)
                            {
                                Temp_HypothesisSupport[vecA[idxA2], nearB] += 1.0f;
                            }
                        }
                    }
                }
            }
        }

        //finalCount = count ?? setA.Dimensions[1];
        //for (int i = 0; i < finalCount; i += 1)
        //{
        //    if (i % 100 == 0)
        //        _userFriendlyLogger.LogInformation($"B i = {i}");

        //    var vecB = setB.GetColumn(i);
        //    // Индексы позиций с единицей
        //    for (int idxB = 0; idxB < VectorLength; idxB += 1)
        //    {
        //        if (vecB[idxB] > 0.5f)
        //        {
        //            for (int idxA = 0; idxA < VectorLength; idxA += 1)
        //            {
        //                // Гипотеза: idxA → idxB
        //                //HypothesisSupport[idxA, idxB] += 1.0f;

        //                // Подкрепляем также все пары в 16 ближайших
        //                foreach (var nearA in nearestA[idxA].Items)
        //                {
        //                    for (int idxB2 = 0; idxB2 < VectorLength; idxB2 += 1)
        //                    {
        //                        if (idxB2 != idxB && vecB[idxB2] > 0.5f)
        //                        {
        //                            HypothesisSupport[nearA, idxB2] += 1.0f;
        //                        }
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}
    }

    /// <summary>
    /// Подкрепление гипотез на примерах
    /// </summary>
    /// <param name="setA"></param>
    /// <param name="setB"></param>
    public void SupportHypotheses_V2()
    {
        var primaryBitsNearestA = Temp_PrimaryBitsNearestA.Array;
        var primaryBitsNearestB = Temp_PrimaryBitsNearestB.Array;
        
        for (int i = 0; i < WordsCount; i += 1)
        {
            if (i % 100 == 0)
                _userFriendlyLogger.LogInformation($"A i = {i}");

            var vecA = Temp_WordPrimaryBitIndices_Collection_RU[i];

            for (int idxB = 0; idxB < VectorLength; idxB += 1)
            {
                for (int idxA2 = 0; idxA2 < vecA.Length; idxA2 += 1)
                {
                    Temp_HypothesisSupport[vecA[idxA2], idxB] += 1.0f;
                }

                // Подкрепляем также все пары в 16 ближайших
                foreach (var nearB in primaryBitsNearestB[idxB].Items)
                {
                    for (int idxA2 = 0; idxA2 < vecA.Length; idxA2 += 1)
                    {
                        Temp_HypothesisSupport[vecA[idxA2], nearB] += 1.0f;
                    }
                }
            }
        }

        //finalCount = count ?? setA.Dimensions[1];
        //for (int i = 0; i < finalCount; i += 1)
        //{
        //    if (i % 100 == 0)
        //        _userFriendlyLogger.LogInformation($"B i = {i}");

        //    var vecB = setB.GetColumn(i);
        //    // Индексы позиций с единицей
        //    for (int idxB = 0; idxB < VectorLength; idxB += 1)
        //    {
        //        if (vecB[idxB] > 0.5f)
        //        {
        //            for (int idxA = 0; idxA < VectorLength; idxA += 1)
        //            {
        //                // Гипотеза: idxA → idxB
        //                //HypothesisSupport[idxA, idxB] += 1.0f;

        //                // Подкрепляем также все пары в 16 ближайших
        //                foreach (var nearA in nearestA[idxA].Items)
        //                {
        //                    for (int idxB2 = 0; idxB2 < VectorLength; idxB2 += 1)
        //                    {
        //                        if (idxB2 != idxB && vecB[idxB2] > 0.5f)
        //                        {
        //                            HypothesisSupport[nearA, idxB2] += 1.0f;
        //                        }
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}
    }

    // Получить итоговое соответствие
    public int[] GetFinalMappingForcedExclusive()
    {
        var result = new int[VectorLength];
        var usedB = new HashSet<int>();

        for (int i = 0; i < VectorLength; i++)
        {
            // Ищем позицию B с максимальным весом среди неиспользованных
            float max = float.MinValue;
            int selected = -1;
            for (int j = 0; j < VectorLength; j++)
            {
                if (!usedB.Contains(j) && Temp_HypothesisSupport[i, j] > max)
                {
                    max = Temp_HypothesisSupport[i, j];
                    selected = j;
                }
            }
            if (selected != -1)
            {
                result[i] = selected;
                usedB.Add(selected);
            }
        }
        return result;
    }

    public int[] GetFinalClustersMapping()
    {
        var result = new int[VectorLength];

        //for (int i = 0; i < VectorLength; i++)
        //{
        //    // Ищем позицию B с максимальным весом среди неиспользованных
        //    float max = float.MinValue;
        //    int selected = -1;
        //    for (int j = 0; j < VectorLength; j++)
        //    {
        //        if (Temp_HypothesisSupport[i, j] > max)
        //        {
        //            max = Temp_HypothesisSupport[i, j];
        //            selected = j;
        //        }
        //    }
        //    if (selected != -1)
        //    {
        //        result[i] = selected;
        //    }
        //}
        return result;
    }

    public class PrimaryBitsNearest : IOwnedDataSerializable
    {
        public FastList<int>[] Array = null!;

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.WriteArrayOfOwnedDataSerializable(Array, context);
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        Array = reader.ReadArrayOfOwnedDataSerializable(() => new FastList<int>(0), context);
                        break;
                }
            }
        }
    }
}
