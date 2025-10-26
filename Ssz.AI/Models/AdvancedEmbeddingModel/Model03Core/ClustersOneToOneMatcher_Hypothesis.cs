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
    public LanguageDiscreteEmbeddings LanguageDiscreteEmbeddings_A;
    public LanguageDiscreteEmbeddings LanguageDiscreteEmbeddings_B;
    public int VectorLength;
    
    public MatrixFloat Temp_PrimaryBitsEnergy_Matrix_A = null!;    
    
    public MatrixFloat Temp_PrimaryBitsEnergy_Matrix_B = null!;

    public PrimaryBitsNearest Temp_PrimaryBitsNearest_A = null!;

    public PrimaryBitsNearest Temp_PrimaryBitsNearest_B = null!;

    ///// <summary>
    /////     [Word.Index, [Bit Index]]
    ///// </summary>
    //public int[][] Temp_WordPrimaryBitIndices_Collection_A = null!;

    ///// <summary>
    /////     [Word.Index, [Bit Index]]
    ///// </summary>
    //public int[][] Temp_WordPrimaryBitIndices_Collection_B = null!;

    /// <summary>
    ///     Таблица гипотез: [позицияA, позицияB]
    /// </summary>
    public DenseMatrix<Link> Temp_Links = null!;

    public int WordsCount = 10000;

    /// <summary>
    /// Количество ближайших для подкрепления
    /// </summary>
    public int NearestCount = 7;

    public ClustersOneToOneMatcher_Hypothesis(
        IUserFriendlyLogger userFriendlyLogger, 
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_A, 
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_B)
    {
        _userFriendlyLogger = userFriendlyLogger;
        _device = cuda.is_available() ? CUDA : CPU;
        LanguageDiscreteEmbeddings_A = languageDiscreteEmbeddings_A;
        LanguageDiscreteEmbeddings_B = languageDiscreteEmbeddings_B;
        VectorLength = languageDiscreteEmbeddings_A.ClusterInfos.Count;        
    }

    public void GenerateOwnedData()
    {        
    }

    public void Prepare()
    {
        WordsCount = Math.Min(Math.Min(WordsCount, LanguageDiscreteEmbeddings_A.Words.Count), LanguageDiscreteEmbeddings_A.Words.Count);

        Temp_PrimaryBitsEnergy_Matrix_A = ModelHelper.GetPrimaryBitsEnergy_Matrix(LanguageDiscreteEmbeddings_A.ClusterInfos);
        Temp_PrimaryBitsEnergy_Matrix_B = ModelHelper.GetPrimaryBitsEnergy_Matrix(LanguageDiscreteEmbeddings_B.ClusterInfos);

        Temp_Links = new DenseMatrix<Link>(VectorLength, VectorLength);
        Temp_Links.CreateElementInstances(() => new Link());

        Temp_PrimaryBitsNearest_A = ModelHelper.BuildPrimaryBitsNearest(Temp_PrimaryBitsEnergy_Matrix_A, NearestCount);
        Temp_PrimaryBitsNearest_B = ModelHelper.BuildPrimaryBitsNearest(Temp_PrimaryBitsEnergy_Matrix_B, NearestCount);

        //Temp_WordPrimaryBitIndices_Collection_A = BuildWordPrimaryBitIndices_Collection(LanguageDiscreteEmbeddings_A);
        //Temp_WordPrimaryBitIndices_Collection_B = BuildWordPrimaryBitIndices_Collection(LanguageDiscreteEmbeddings_B);        
    }        

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {            
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:                    
                    break;
            }
        }
    }

    /// <summary>
    /// Формирование дерева гипотез, исходя из начальных.
    /// </summary>
    /// <param name="initialLinks"></param>
    public void SupportHypotheses((int i, int j)[] initialLinks)
    {
        var primaryBitsNearest_B = Temp_PrimaryBitsNearest_B.Array;

        int initialHypothesisNum = -1;
        foreach (var it in initialLinks)
        {
            var initialLink = Temp_Links[it.i, it.j];
            initialHypothesisNum += 1;
            initialLink.Chain += ModelHelper.NumToSymbol(initialHypothesisNum);
            initialLink.Strength = 1.0f;
        }

        //// Индексы позиций
        //Parallel.For(
        //    0, // начальный индекс (включительно)
        //    VectorLength, // конечный индекс (не включительно)                    
        //    bitIndex_A => // Основной делегат, исполняемый в параллельном потоке
        //    {
        //        var primaryBitsNearest_A = Temp_PrimaryBitsNearest_A.Array[bitIndex_A];

        //        int hypothesisNum = -1;
        //        for (int bitIndex_B = 0; bitIndex_B < VectorLength; bitIndex_B += 1)
        //        {
        //            var primaryBitsNearest_B = Temp_PrimaryBitsNearest_B.Array[bitIndex_B];

        //            float strength = 0.0f;
        //            for (int n = 0; n < primaryBitsNearest_A.Count; n += 1)
        //            {
        //                var linkN = Temp_Links[primaryBitsNearest_A[n], primaryBitsNearest_B[n]];
        //                if (linkN == @"")
        //                    continue;
        //                ComputeNeighborWeight(n);
        //            }
        //        } 
        //    });        
    }    

    //// Получить итоговое соответствие
    //public int[] GetFinalPrimaryBitsMapping_ForcedExclusive()
    //{
    //    var result = new int[VectorLength];
    //    var usedB = new HashSet<int>();

    //    for (int i = 0; i < VectorLength; i++)
    //    {
    //        // Ищем позицию B с максимальным весом среди неиспользованных
    //        float max = float.MinValue;
    //        int selectedJ = -1;
    //        for (int j = 0; j < VectorLength; j++)
    //        {
    //            if (!usedB.Contains(j) && Temp_HypothesisSupport[i, j] > max)
    //            {
    //                max = Temp_HypothesisSupport[i, j];
    //                selectedJ = j;
    //            }
    //        }
    //        if (selectedJ != -1)
    //        {
    //            result[i] = selectedJ;
    //            usedB.Add(selectedJ);
    //        }
    //    }
    //    return result;
    //}

    public int[] GetFinalPrimaryBitsMapping()
    {
        var result = new int[VectorLength];
        Array.Fill(result, -1);

        for (int i = 0; i < VectorLength; i++)
        {            
            //float max = float.MinValue;
            int selectedJ = -1;
            for (int j = 0; j < VectorLength; j++)
            {
                if (!String.IsNullOrEmpty(Temp_Links[i, j].Chain))
                {
                    //max = Temp_HypothesisSupport[i, j];
                    selectedJ = j;
                    break;
                }
            }
            if (selectedJ != -1)
            {
                result[i] = selectedJ;
            }
        }

        return result;
    }
    
    //private int[][] BuildWordPrimaryBitIndices_Collection(LanguageDiscreteEmbeddings languageDiscreteEmbeddings)
    //{
    //    var wordPrimaryBitIndices_Collection = new int[WordsCount][];
    //    List<int> wordPrimaryBitIndices = new List<int>(16);
    //    for (int i = 0; i < WordsCount; i += 1)
    //    {
    //        var discreteVector_PrimaryBitsOnly = languageDiscreteEmbeddings.Words[i].DiscreteVector_PrimaryBitsOnly;
    //        wordPrimaryBitIndices.Clear();
    //        for (int j = 0; j < discreteVector_PrimaryBitsOnly.Length; j += 1)
    //        {
    //            if (discreteVector_PrimaryBitsOnly[j] > 0.5f)
    //            {
    //                wordPrimaryBitIndices.Add(j);
    //            }
    //        }
    //        wordPrimaryBitIndices_Collection[i] = wordPrimaryBitIndices.ToArray();
    //    }
    //    return wordPrimaryBitIndices_Collection;
    //}    

    /// <summary>
    ///     Primry bit to bit link.
    /// </summary>
    public class Link : IOwnedDataSerializable
    {
        public string Chain = "";

        public float Strength;

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            writer.Write(Chain);
            writer.Write(Strength);
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            Chain = reader.ReadString();
            Strength = reader.ReadSingle();
        }
    }
}
