using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;
using Ssz.Utils;
using Ssz.Utils.Logging;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Text.RegularExpressions;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;

public static class WordsHelper
{
    public const int OldVectorLength_RU = 300;

    public const int OldVectorLength_EN = 300;

    /// <summary>
    ///     FastText
    /// </summary>
    /// <param name="languageInfo_RU"></param>
    /// <param name="wordsMaxCount"></param>
    /// <param name="loggersSet"></param>
    /// <param name="loadOldVectors"></param>
    public static void InitializeWords_RU(LanguageInfo languageInfo_RU, int wordsMaxCount, ILoggersSet loggersSet, bool loadOldVectors = true)
    {
        languageInfo_RU.Words = new(wordsMaxCount); // Initial reserved capacity                        

        //string fileFullName = @"Data\Ssz.AI.AdvancedEmbedding\RU\cc.ru.300.vec";
        string fileFullName = @"Data\Ssz.AI.AdvancedEmbedding\RU\model_RU_20000.csv";
        if (String.Equals(Path.GetExtension(fileFullName), @".vec"))
        {
            Regex regex = new("^[а-яА-ЯёЁ]+$");
            InitializeWords_FastText(
                regex.IsMatch,
                fileFullName,
                OldVectorLength_RU, 
                languageInfo_RU, 
                wordsMaxCount, 
                loggersSet, 
                loadOldVectors);
        }
        else
        {
            foreach (var line in File.ReadAllLines(fileFullName))
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
                //word.Freq = new Any(parts[^1]).ValueAsDouble(false);

                languageInfo_RU.Words.Add(word);

                if (languageInfo_RU.Words.Count >= wordsMaxCount)
                    break;
            }
        }            
    }

    /// <summary>
    ///     FastText
    /// </summary>
    /// <param name="languageInfo_EN"></param>
    /// <param name="wordsMaxCount"></param>
    /// <param name="loggersSet"></param>
    /// <param name="loadOldVectors"></param>
    public static void InitializeWords_EN(LanguageInfo languageInfo_EN, int wordsMaxCount, ILoggersSet loggersSet, bool loadOldVectors = true)
    {
        languageInfo_EN.Words = new(wordsMaxCount); // Initial reserved capacity                        

        //string fileFullName = @"Data\Ssz.AI.AdvancedEmbedding\EN\cc.en.300.vec";
        string fileFullName = @"Data\Ssz.AI.AdvancedEmbedding\EN\glove.42B.300d_20000.txt";
        if (String.Equals(Path.GetExtension(fileFullName), @".vec"))
        {
            Regex regex = new("^[a-zA-Z]+$");
            InitializeWords_FastText(
                regex.IsMatch,
                fileFullName, 
                OldVectorLength_EN, 
                languageInfo_EN, 
                wordsMaxCount, 
                loggersSet, 
                loadOldVectors);
        }
        else
        {
            foreach (var line in File.ReadAllLines(fileFullName))
            {
                var parts = CsvHelper.ParseCsvLine(" ", line);
                if (parts.Length < OldVectorLength_EN + 1 || string.IsNullOrEmpty(parts[0]))
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
                //word.Freq = new Any(parts[^1]).ValueAsDouble(false);

                languageInfo_EN.Words.Add(word);

                if (languageInfo_EN.Words.Count >= wordsMaxCount)
                    break;
            }
        } 
    }

    public static void InitializeWords_FastText(
        Func<string, bool> isMatch,
        string fileFullName, 
        int oldVectorLength,
        LanguageInfo languageInfo, 
        int wordsMaxCount, 
        ILoggersSet loggersSet, 
        bool loadOldVectors = true)
    {
        languageInfo.Words = new(wordsMaxCount); // Initial reserved capacity                        

        HashSet<string> uniqueWords = new HashSet<string>();
        foreach (var line in File.ReadLines(fileFullName))
        {
            var parts = CsvHelper.ParseCsvLine(" ", line);
            if (parts.Length < oldVectorLength + 1 || string.IsNullOrEmpty(parts[0]))
                continue;            
            if (parts.Length - 1 != oldVectorLength)
            {
                loggersSet.UserFriendlyLogger.LogError("Incorrect vector length in input = " + (parts.Length - 2));
                return;
            }
            Word word = new Word
            {
                Index = languageInfo.Words.Count,
                Name = parts[0]!,
            };
            if (String.IsNullOrEmpty(word.Name) || !isMatch(word.Name))
                continue;
            if (!uniqueWords.Add(word.Name.ToLowerInvariant()))
                continue;

            if (loadOldVectors)
            {
                var oldVectror = word.OldVector;
                foreach (int i in Enumerable.Range(0, parts.Length - 2))
                {
                    oldVectror[i] = float.Parse(parts[i + 1] ?? @"", CultureInfo.InvariantCulture);
                }                
                float norm = TensorPrimitives.Norm(oldVectror);
                if (norm < 1e-8f)
                {
                    oldVectror[0] = 0.01f; // Устанавливаем небольшое значение для избежания нулевого вектора
                    norm = TensorPrimitives.Norm(oldVectror);
                }                
                TensorPrimitives.Divide(oldVectror, norm, word.OldVectorNormalized);
                //word.Freq = norm;
            }            

            languageInfo.Words.Add(word);

            if (languageInfo.Words.Count >= wordsMaxCount)
                break;
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
