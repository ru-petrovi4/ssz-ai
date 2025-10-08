using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{
    public static partial class ModelHelper
    {
        public static void ShowWords(LanguageDiscreteEmbeddings source, LanguageDiscreteEmbeddings target, int[] clustersMapping, ILogger logger)
        {
            for (int sourceClusterIndex = 0; sourceClusterIndex < source.ClusterInfos.Count; sourceClusterIndex += 1)
            {
                ShowWords(source, sourceClusterIndex, logger);
                ShowWords(target, clustersMapping[sourceClusterIndex], logger);
                logger.LogInformation($"------------------------");
            }
        }

        private static void ShowWords(LanguageDiscreteEmbeddings source, int clusterIndex, ILogger logger)
        {
            logger.LogInformation($"Кластер: {clusterIndex}");

            var clusterInfo = source.ClusterInfos[clusterIndex];

            foreach (var word in source.Words
                .Where(w => w.ClusterIndex == clusterIndex)
                .OrderByDescending(w => TensorPrimitives.Dot(w.OldVectorNormalized, clusterInfo.CentroidOldVectorNormalized))
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
