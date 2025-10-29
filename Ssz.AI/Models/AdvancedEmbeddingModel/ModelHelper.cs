using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Evaluation;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public static partial class ModelHelper
{
    /// <summary>
    ///     1-cos, экспонента
    ///     Индексы соотвествуют индексам главных бит в слове.
    /// </summary>
    /// <param name="languageDiscreteEmbeddings"></param>
    /// <returns></returns>
    public static MatrixFloat GetPrimaryBitsEnergy_Matrix(List<ClusterInfo> clusterInfos)
    {
        int dimension = clusterInfos.Count;
        var matrixFloat = new MatrixFloat(dimension, dimension);
        foreach (var i in Enumerable.Range(0, dimension))
        {
            foreach (var j in Enumerable.Range(0, dimension))
            {
                var clusterI = clusterInfos[i];
                var clusterJ = clusterInfos[j];
                float v = GetEnergy(
                    clusterI.CentroidOldVectorNormalized,
                    clusterJ.CentroidOldVectorNormalized);                    
                matrixFloat[clusterI.HashProjectionIndex, clusterJ.HashProjectionIndex] = v;                    
            }
        }
        return matrixFloat;
    }

    /// <summary>
    ///     1-cos
    ///     Индексы соотвествуют индексам главных бит в слове.
    /// </summary>
    /// <param name="languageDiscreteEmbeddings"></param>
    /// <returns></returns>
    public static MatrixFloat GetPrimaryBitsEnergy_Matrix_Cos(List<ClusterInfo> clusterInfos)
    {
        int dimension = clusterInfos.Count;
        var matrixFloat = new MatrixFloat(dimension, dimension);
        foreach (var i in Enumerable.Range(0, dimension))
        {
            foreach (var j in Enumerable.Range(0, dimension))
            {
                var clusterI = clusterInfos[i];
                var clusterJ = clusterInfos[j];
                float v = GetEnergy_Cos(
                    clusterI.CentroidOldVectorNormalized,
                    clusterJ.CentroidOldVectorNormalized);
                matrixFloat[clusterI.HashProjectionIndex, clusterJ.HashProjectionIndex] = v;
            }
        }
        return matrixFloat;
    }

    /// <summary>
    ///     1-cos, экспонента
    ///     Индексы соотвествуют индексам главных бит в слове.
    /// </summary>
    /// <param name="clusterInfos"></param>
    /// <returns></returns>
    public static MatrixFloat GetMappedPrimaryBitsEnergy_Matrix(List<ClusterInfo> clusterInfos)
    {
        int dimension = clusterInfos.Count;
        var matrixFloat = new MatrixFloat(dimension, dimension);
        foreach (var i in Enumerable.Range(0, dimension))
        {
            foreach (var j in Enumerable.Range(0, dimension))
            {
                var clusterI = clusterInfos[i];
                var clusterJ = clusterInfos[j];
                float v = GetEnergy(
                    clusterI.CentroidOldVectorNormalized_Mapped!,
                    clusterJ.CentroidOldVectorNormalized_Mapped!);
                matrixFloat[clusterI.HashProjectionIndex, clusterJ.HashProjectionIndex] = v;
            }
        }
        return matrixFloat;
    }

    /// <summary>
    ///     Найти ближайшие позиции для всех (по строкам)
    /// </summary>
    /// <param name="primaryBitsEnergy_Matrix"></param>
    /// <returns></returns>
    public static PrimaryBitsNearest BuildPrimaryBitsNearest(MatrixFloat primaryBitsEnergy_Matrix, int nearestCount)
    {
        var nearestArray = new FastList<int>[primaryBitsEnergy_Matrix.Dimensions[0]];
        for (int i = 0; i < primaryBitsEnergy_Matrix.Dimensions[0]; i += 1)
        {
            var list = new List<(int idx, float val)>(primaryBitsEnergy_Matrix.Dimensions[1]);
            for (int j = 0; j < primaryBitsEnergy_Matrix.Dimensions[1]; j += 1)
            {
                if (i != j)
                    list.Add((j, primaryBitsEnergy_Matrix[i, j]));
            }
            // Берём NearestCount ближайших
            nearestArray[i] = new FastList<int>(list.OrderBy(it => it.val).Take(nearestCount).Select(x => x.idx).ToArray());
        }
        return new PrimaryBitsNearest()
        {
            Array = nearestArray
        };
    }

    /// <summary>
    ///     1-cos, экспонента 
    /// </summary>        
    public static float GetEnergy(float[] oldVectorNormalizedA, float[] oldVectorNormalizedB)
    {
        float v = System.Numerics.Tensors.TensorPrimitives.CosineSimilarity(
                    oldVectorNormalizedA,
                    oldVectorNormalizedB);
        v = MathF.Exp((1.0f - v) * 2.0f) - 1;            
        return v;
    }

    /// <summary>
    ///     1-cos, экспонента 
    /// </summary>        
    public static float GetEnergy_Cos(float[] oldVectorNormalizedA, float[] oldVectorNormalizedB)
    {
        float v = System.Numerics.Tensors.TensorPrimitives.CosineSimilarity(
                    oldVectorNormalizedA,
                    oldVectorNormalizedB);
        v = 1.0f - v;
        return v;
    }

    public static float GetWord_PrimaryBitsOnly_Energy(float[] discreteVector_PrimaryBitsOnly, MatrixFloat energyMatrix)
    {
        // Список индексов, где vector[i] == 1 для быстрого перебора (около 8 элементов)
        Span<int> primaryBitsOnly_Indices = stackalloc int[8];
        int count = 0;
        for (int i = 0; i < discreteVector_PrimaryBitsOnly.Length; i += 1)
        {
            if (discreteVector_PrimaryBitsOnly[i] > 0.5f)
            {
                // Обычно у вас мало единичных элементов, stackalloc более производителен для малых массивов
                primaryBitsOnly_Indices[count] = i;
                count += 1;
            }
        }

        Debug.Assert(count == Model01.Constants.DiscreteVector_PrimaryBitsCount);

        return GetEnergy(primaryBitsOnly_Indices, energyMatrix);
    }

    public static float GetEnergy(ReadOnlySpan<int> primaryBitsOnly_Indices, MatrixFloat energyMatrix)
    {
        float energy = 0.0f;
        // Перебираем только пары (i < j), чтобы не считать симметричную/диагональную энергию дважды
        for (int k = 0; k < primaryBitsOnly_Indices.Length - 1; k += 1)
        {
            int i = primaryBitsOnly_Indices[k];
            for (int l = k + 1; l < primaryBitsOnly_Indices.Length; l += 1)
            {
                int j = primaryBitsOnly_Indices[l];
                energy += energyMatrix[i, j];
            }
        }
        return energy;
    }

    public static void SetClusterStatistics(ClusterInfo clusterInfo, Word[] clusterWords)
    {
        clusterInfo.WordsCount = clusterWords.Length;
        float sum = 0.0f;
        for (int i = 0; i < clusterWords.Length; i += 1)
        {
            sum += TensorPrimitives.Norm(clusterWords[i].OldVector);
        }
        clusterInfo.AverageWordsNorm = sum / clusterWords.Length;

        var clustersWordsTop10 = clusterWords.Take(10).ToArray();
        sum = 0.0f;
        for (int i = 0; i < clustersWordsTop10.Length; i += 1)
        {
            sum += TensorPrimitives.Norm(clustersWordsTop10[i].OldVector);
        }
        clusterInfo.AverageWordsNormTop10 = sum / clustersWordsTop10.Length;
    }

    public static void ShowWords(
        LanguageDiscreteEmbeddings source, 
        LanguageDiscreteEmbeddings target, 
        int[] clustersMapping, 
        ILogger logger,
        int[]? ideal_ClustersMapping = null)
    {
        var clustersMappingFiltered = clustersMapping.Where(m => m != -1).ToArray();
        var hs = clustersMappingFiltered.ToHashSet();
        logger.LogInformation($"Количество уникальных сопоставлений: {hs.Count}/{clustersMappingFiltered.Length}");

        if (ideal_ClustersMapping is not null)
        {
            //Debug.Assert(ideal_ClustersMapping.ToHashSet().Count == ideal_ClustersMapping.Length);
            logger.LogInformation($"Совпадений с идеалом: {clustersMappingFiltered.Select((cm, i) => cm == ideal_ClustersMapping[i] ? 1 : 0).Sum()}/{clustersMappingFiltered.Length}");
        }

        var counts = new Dictionary<int, int>(clustersMappingFiltered.Length);            
        foreach (int number in clustersMappingFiltered)
        {                
            if (counts.ContainsKey(number))
                counts[number] += 1;
            else
                counts[number] = 1;
        }            
        foreach (var pair in counts)
        {
            logger.LogInformation($"Target cluster {pair.Key} count: {pair.Value}.");
        }

        for (int sourceClusterIndex = 0; sourceClusterIndex < source.ClusterInfos.Count; sourceClusterIndex += 1)
        {
            source.ClusterInfos[sourceClusterIndex].Temp_ClusterIndex = sourceClusterIndex;
        }

        foreach (var clusterInfo in source.ClusterInfos.OrderBy(ci => ci.WordsCount))
        {
            ShowWords(source, clusterInfo.Temp_ClusterIndex, logger);
            ShowWords(target, clustersMapping[clusterInfo.Temp_ClusterIndex], logger);
            logger.LogInformation($"------------------------");
        }
    }

    private static void ShowWords(LanguageDiscreteEmbeddings embeddings, int clusterIndex, ILogger logger)
    {
        if (clusterIndex == -1)
            return;
        var clusterInfo = embeddings.ClusterInfos[clusterIndex];
        if (clusterInfo is null)
            return;

        logger.LogInformation($"Кластер: {clusterIndex}; Слов в кластере: {clusterInfo.WordsCount}; AverageWordsNorm: {clusterInfo.AverageWordsNorm}; AverageWordsNormTop10: {clusterInfo.AverageWordsNormTop10}; Concentration: {clusterInfo.Concentration}; MixingCoefficient: {clusterInfo.MixingCoefficient}");            

        foreach (var word in embeddings.Words
            .Where(w => w.ClusterIndex == clusterIndex)
            .OrderBy(w => GetEnergy(w.OldVectorNormalized, clusterInfo.CentroidOldVectorNormalized))
            .Take(10))
        {
            logger.LogInformation(word.Name);
        }

        logger.LogInformation($"------------------------");
    }        

    public static async Task Operation1(AddonsManager addonsManager, IServiceProvider serviceProvider, IConfiguration configuration, ILoggersSet loggersSet)
    {
        string programDataDirectoryFullName = Directory.GetCurrentDirectory();

        var _10000 = (await File.ReadAllLinesAsync(Path.Combine(programDataDirectoryFullName, @"10000.csv"))).ToHashSet();

        var shortLines = new List<string>();
        foreach (var line in File.ReadAllLines(Path.Combine(programDataDirectoryFullName, @"model.csv")))
        {
            var spaceIndex = line.IndexOf(" ");
            var underscoreIndex = line.IndexOf("_");
            if (underscoreIndex > 0 && underscoreIndex < spaceIndex)
            {
                var wordWithoutPostfix = line.Substring(0, underscoreIndex);
                if (_10000.Contains(wordWithoutPostfix))
                {
                    shortLines.Add(line.Replace(' ', ','));
                }
            }
        }

        await File.WriteAllLinesAsync(Path.Combine(programDataDirectoryFullName, @"model_short.csv"), shortLines);
        await File.WriteAllLinesAsync(Path.Combine(programDataDirectoryFullName, @"model_short_sorted.csv"), shortLines.OrderBy(l => l));
    }

    public static async Task Operation2(AddonsManager addonsManager, IServiceProvider serviceProvider, IConfiguration configuration, ILoggersSet loggersSet)
    {
        await Task.Delay(10);

        //model.Initialize(loggersSet);
    }

    public static async Task Operation3(AddonsManager addonsManager, IServiceProvider serviceProvider, IConfiguration configuration, ILoggersSet loggersSet)
    {
        string programDataDirectoryFullName = Directory.GetCurrentDirectory();

        List<(string, double)> freqRaw = new(60000);
        foreach (var line in (await File.ReadAllLinesAsync(Path.Combine(programDataDirectoryFullName, @"freqrnc2011.csv"))))
        {
            var parts = CsvHelper.ParseCsvLine("\t", line);
            if (parts.Length < 3)
                continue;
            string? word = parts[0];
            if (String.IsNullOrEmpty(word))
                continue;
            switch (parts[1])
            {
                case "s":
                    word += "_NOUN";
                    break;
                case "a":
                    word += "_ADJ";
                    break;
                case "v":
                    word += "_VERB";
                    break;
                case "adv":
                    word += "_ADV";
                    break;
                //case "spro":
                //    word += "_ADJ";
                //    break;
                default:
                    word = null;
                    break;
            }
            if (!String.IsNullOrEmpty(word))
                freqRaw.Add((word, new Any(parts[2]).ValueAsDouble(false)));
        }
        var freqDictionary = freqRaw.OrderByDescending(i => i.Item2).Take(20000).ToDictionary(i => i.Item1, i => i.Item2);

        var shortLines = new List<(string, double)>();
        foreach (var line in File.ReadAllLines(Path.Combine(programDataDirectoryFullName, @"model.csv")))
        {
            var spaceIndex = line.IndexOf(" ");                
            if (spaceIndex > 0)
            {
                var word = line.Substring(0, spaceIndex);
                if (freqDictionary.TryGetValue(word, out double freq))
                {
                    shortLines.Add((line.Replace(' ', ',') + "," + new Any(freq).ValueAsString(false), freq));
                }
            }
        }

        await File.WriteAllLinesAsync(Path.Combine(programDataDirectoryFullName, @"model_short.csv"), shortLines.OrderByDescending(l => l.Item2).Select(l => l.Item1));
        //await File.WriteAllLinesAsync(Path.Combine(programDataDirectoryFullName, @"model_short_sorted.csv"), shortLines.OrderBy(l => l));
    }

    [return:NotNullIfNotNull(nameof(clustersMapping_A_B))]
    public static int[]? GetPrimaryBitsMapping(int[]? clustersMapping_A_B, List<ClusterInfo> clusterInfos_A, List<ClusterInfo> clusterInfos_B)
    {
        if (clustersMapping_A_B is null)
            return null;

        int[] primaryBitsMapping_A_B = new int[clustersMapping_A_B.Length];

        foreach (int clusterIndexA in Enumerable.Range(0, clustersMapping_A_B.Length))
        {
            primaryBitsMapping_A_B[clusterInfos_A[clusterIndexA].HashProjectionIndex] =
                clusterInfos_B[clustersMapping_A_B[clusterIndexA]].HashProjectionIndex;
        }

        //Debug.Assert(primaryBitsMapping_A_B.ToHashSet().Count == primaryBitsMapping_A_B.Length);

        return primaryBitsMapping_A_B;
    }

    public static int[] GetClustersMapping(int[] primaryBitsMapping_A_B, List<ClusterInfo> clusterInfos_A, List<ClusterInfo> clusterInfos_B)
    {
        int[] clustersMapping_A_B = new int[primaryBitsMapping_A_B.Length];

        foreach (int bitIndexA in Enumerable.Range(0, primaryBitsMapping_A_B.Length))
        {
            clustersMapping_A_B[clusterInfos_A.FindIndex(ci => ci.HashProjectionIndex == bitIndexA)] =
                clusterInfos_B.FindIndex(ci => ci.HashProjectionIndex == primaryBitsMapping_A_B[bitIndexA]);
        }

        Debug.Assert(clustersMapping_A_B.ToHashSet().Count == clustersMapping_A_B.Length);

        return clustersMapping_A_B;
    }

    public static float[] GetDiscreteVector_PrimaryBitsOnly_Mapped(float[] discreteVector_PrimaryBitsOnly_A, int[] primaryBitsMapping_A_B)
    {
        var discreteVector_PrimaryBitsOnly_Mapped = new float[discreteVector_PrimaryBitsOnly_A.Length];

        int count = 0;
        for (int i = 0; i < discreteVector_PrimaryBitsOnly_A.Length; i += 1)
        {
            if (discreteVector_PrimaryBitsOnly_A[i] > 0.5f)
            {
                // Обычно у вас мало единичных элементов, stackalloc более производителен для малых массивов
                discreteVector_PrimaryBitsOnly_Mapped[primaryBitsMapping_A_B[i]] = 1.0f;
                count += 1;
            }
        }

        Debug.Assert(count == Model01.Constants.DiscreteVector_PrimaryBitsCount);

        return discreteVector_PrimaryBitsOnly_Mapped;
    }

    public static char NumToSymbol(int n)
    {
        return (char)('A' + n);
    }

    /// <summary>
    /// Вычисляет вес голоса от соседа ранга p
    /// </summary>
    /// <param name="p">Ранг соседа (0 — ближайший, k-1 — самый дальний из k соседей)</param>
    /// <returns></returns>
    public static float ComputeNeighborWeight(int p)
    {
        // Экспоненциальное убывание веса с увеличением ранга
        // Параметр beta контролирует скорость убывания
        const float beta = 0.5f;
        return MathF.Exp(-beta * p);
    }

    public static float GetStrength(
        int bitIndex_A,
        int[] primaryBitsMapping_A_B,
        PrimaryBitsNearest primaryBitsNearest_A,
        PrimaryBitsNearest primaryBitsNearest_B,
        int nearestCount)
    {
        var bitsNearest_A = primaryBitsNearest_A.Array[bitIndex_A];
        var bitsNearest_B = primaryBitsNearest_B.Array[primaryBitsMapping_A_B[bitIndex_A]];
        float strength = 0.0f;
        for (int n1 = 0; n1 < 1; n1 += 1)
        {
            var bitsNearest_N1_A = bitsNearest_A[n1];
            var bitsNearest_N1_A_Mapped = primaryBitsMapping_A_B[bitsNearest_N1_A];
            for (int n2 = 0; n2 < nearestCount; n2 += 1)
            {                     
                if (bitsNearest_N1_A_Mapped == bitsNearest_B[n2])
                {
                    //float w = n2 + 1;//ComputeNeighborWeight(n2); // TEMPCODE
                    strength += n2 + 1;
                    break;
                }
            }
        }
        return strength;
    }
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