using Microsoft.Extensions.Logging;
using Ssz.Utils;
using Ssz.Utils.Logging;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;

public static class WordsHelper
{
    public const int OldVectorLength_RU = 300;

    public const int OldVectorLength_EN = 300;

    public static void InitializeWords_RU(LanguageInfo languageInfo_RU, ILoggersSet loggersSet, bool loadOldVectors = true)
    {
        languageInfo_RU.Words = new(20000); // Initial reserved capacity                        

        foreach (var line in File.ReadAllLines(@"Data\Ssz.AI.AdvancedEmbedding\RU\model_20000.csv"))
        {
            var parts = CsvHelper.ParseCsvLine(",", line);
            if (parts.Length < 300 || string.IsNullOrEmpty(parts[0]))
                continue;
            Word word = new Word
            {
                Index = languageInfo_RU.Words.Count,
                Name = parts[0]!,
            };
            if (parts.Length - 2 != OldVectorLength_RU)
            {
                loggersSet.UserFriendlyLogger.LogError("Incorrect vector length in input = " + (parts.Length - 2));
                return;
            }
            if (loadOldVectors)
            {
                var oldVectror = word.OldVector;
                foreach (int i in Enumerable.Range(0, parts.Length - 2))
                {
                    oldVectror[i] = float.Parse(parts[i + 1] ?? @"", CultureInfo.InvariantCulture);
                }
                var oldVectrorNormalized = word.OldVectorNormalized;
                float norm = TensorPrimitives.Norm(oldVectror);
                TensorPrimitives.Divide(oldVectror, norm, oldVectrorNormalized);
            }
            word.Freq = new Any(parts[^1]).ValueAsDouble(false);

            languageInfo_RU.Words.Add(word);
        }
    }

    public static void InitializeWords_EN(LanguageInfo languageInfo_EN, ILoggersSet loggersSet, bool loadOldVectors = true)
    {
        languageInfo_EN.Words = new(20100); // Initial reserved capacity                        

        foreach (var line in File.ReadAllLines(@"Data\Ssz.AI.AdvancedEmbedding\EN\glove.42B.300d_20000.txt"))
        {
            var parts = CsvHelper.ParseCsvLine(" ", line);
            if (parts.Length < 300 || string.IsNullOrEmpty(parts[0]))
                continue;
            Word word = new Word
            {
                Index = languageInfo_EN.Words.Count,
                Name = parts[0]!,
            };
            if (parts.Length - 1 != OldVectorLength_EN)
            {
                loggersSet.UserFriendlyLogger.LogError("Incorrect vector length in input = " + (parts.Length - 2));
                return;
            }
            if (loadOldVectors)
            {
                var oldVectror = word.OldVector;
                foreach (int i in Enumerable.Range(0, parts.Length - 2))
                {
                    oldVectror[i] = float.Parse(parts[i + 1] ?? @"", CultureInfo.InvariantCulture);
                }
                var oldVectrorNormalized = word.OldVectorNormalized;
                float norm = TensorPrimitives.Norm(oldVectror);
                TensorPrimitives.Divide(oldVectror, norm, oldVectrorNormalized);
            }            
            word.Freq = new Any(parts[^1]).ValueAsDouble(false);

            languageInfo_EN.Words.Add(word);
        }
    }

    public static List<Word> GetRandomOrderWords(List<Word> words, Random r)
    {
        var randomOrderWords = new List<Word>(words.Count);
        foreach (var word in words)
        {
            word.Temp_Flag = false;
        }
        for (int wordIndex = 0; wordIndex < words.Count; wordIndex += 1)
        {   
            for (; ; )
            {
                var word = words[r.Next(words.Count)];
                if (word.Temp_Flag)
                    continue;

                randomOrderWords.Add(word);
                word.Temp_Flag = true;
                break;
            }
        }
        return randomOrderWords;
    }
}
