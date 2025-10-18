using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Evaluation;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{
    public static partial class ModelHelper
    {
        /// <summary>
        ///     1-cos, экспонента
        ///     Индексы соотвествуют индексам главных бит в слове.
        /// </summary>
        /// <param name="languageDiscreteEmbeddings"></param>
        /// <returns></returns>
        public static MatrixFloat GetClustersEnergy_Matrix(LanguageDiscreteEmbeddings embeddings)
        {
            int dimension = embeddings.ClusterInfos.Count;
            var matrixFloat = new MatrixFloat(dimension, dimension);
            foreach (var i in Enumerable.Range(0, dimension))
            {
                foreach (var j in Enumerable.Range(0, dimension))
                {
                    var clusterI = embeddings.ClusterInfos[i];
                    var clusterJ = embeddings.ClusterInfos[j];
                    float v = GetEnergy(
                        clusterI.CentroidOldVectorNormalized,
                        clusterJ.CentroidOldVectorNormalized);                    
                    matrixFloat[clusterI.HashProjectionIndex, clusterJ.HashProjectionIndex] = v;                    
                }
            }
            return matrixFloat;
        }

        /// <summary>
        ///     1-cos, экспонента 
        /// </summary>        
        public static float GetEnergy(float[] oldVectorNormalizedA, float[] oldVectorNormalizedB)
        {
            float v = System.Numerics.Tensors.TensorPrimitives.CosineSimilarity(
                        oldVectorNormalizedA,
                        oldVectorNormalizedB);
            v = MathF.Exp((1 - v) * 2.0f) - 1;            
            return v;
        }

        public static float GetWord_PrimaryBitsOnly_Energy(WordWithDiscreteEmbedding word, MatrixFloat energyMatrix)
        {
            var vector = word.DiscreteVector_PrimaryBitsOnly;

            // Список индексов, где vector[i] == 1 для быстрого перебора (около 8 элементов)
            Span<int> activeIndices = stackalloc int[8];
            int count = 0;
            for (int i = 0; i < vector.Length; i += 1)
            {
                if (vector[i] > 0.5f)
                {
                    // Обычно у вас мало единичных элементов, stackalloc более производителен для малых массивов
                    activeIndices[count] = i;
                    count += 1;
                }
            }

            Debug.Assert(count == Model01.Constants.DiscreteVector_PrimaryBitsCount);

            return GetEnergy(activeIndices, energyMatrix);
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

        public static void ShowWords(LanguageDiscreteEmbeddings source, LanguageDiscreteEmbeddings target, int[] clustersMapping, ILogger logger)
        {
            var hs = clustersMapping.ToHashSet();
            logger.LogInformation($"Количество уникальных сопоставлений: {hs.Count}");
            var counts = new Dictionary<int, int>(clustersMapping.Length);            
            foreach (int number in clustersMapping)
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
                ShowWords(source, sourceClusterIndex, logger);
                ShowWords(target, clustersMapping[sourceClusterIndex], logger);
                logger.LogInformation($"------------------------");
            }
        }

        private static void ShowWords(LanguageDiscreteEmbeddings embeddings, int clusterIndex, ILogger logger)
        {
            var clusterInfo = embeddings.ClusterInfos[clusterIndex];

            logger.LogInformation($"Кластер: {clusterIndex}; Слов в кластере: {clusterInfo.WordsCount}");            

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
    }
}
