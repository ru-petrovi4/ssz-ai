using Avalonia.Media;
using MathNet.Numerics;
using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.Utils;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public class MappingData_V3 : ISerializableModelObject
{
    public const string FileName_PrimaryWords_RU_EN = "PrimaryWords_RU_EN.csv";

    private ILoggersSet _loggersSet;
    public LanguageInfo LanguageInfo_RU;
    public LanguageInfo LanguageInfo_EN;    

    public MatrixFloat Temp_ProxBits_RU = new();
    /// <summary>
    ///     Word for each bit.
    /// </summary>
    public Word[] Temp_BitWords_RU = null!;
    public MatrixFloat Temp_Emb_RU = null!;

    public MatrixFloat Temp_ProxBits_EN = new();
    /// <summary>
    ///     Word for each bit.
    /// </summary>
    public Word[] Temp_BitWords_EN = null!;
    public MatrixFloat Temp_Emb_EN = null!;

    /// <summary>
    ///     Index RU -> Index EN
    /// </summary>
    public int[] Mapping_RU_EN = null!;

    public MappingData_V3(ILoggersSet loggersSet, LanguageInfo languageInfo_RU, LanguageInfo languageInfo_EN)
    {
        _loggersSet = loggersSet;
        LanguageInfo_RU = languageInfo_RU;
        LanguageInfo_EN = languageInfo_EN;        
    }

    public void GenerateOwnedData(int vectorLength)
    {
        Mapping_RU_EN = new int[Model01.Constants.DiscreteVectorLength];
    }    

    public void Prepare3_AnalogVectors()
    {
        var fileData = CsvHelper.LoadCsvFile(Path.Combine(@"Data", FileName_PrimaryWords_RU_EN), includeFiles: false);

        List<(Word, Word)> primaryWords_RU_EN = new(300);
        HashSet<Word> primaryWords_EN = new(300);
        foreach (var line in fileData.Values)
        {
            string word_RU_String = line[0] ?? @"";
            string word_EN_String = (line[1] ?? @"").ToLowerInvariant().Trim();
            var word_RU = LanguageInfo_RU.Words.FirstOrDefault(w => w.Name == word_RU_String);
            var word_EN = LanguageInfo_EN.Words.FirstOrDefault(w => w.Name == word_EN_String);
            if (word_RU is not null && word_EN is not null)
            {
                primaryWords_RU_EN.Add((word_RU, word_EN));
                if (primaryWords_EN.TryGetValue(word_EN, out var word_EN2))
                {
                    _loggersSet.UserFriendlyLogger.LogInformation($"Duplicate EN. word_EN = {word_EN.Name}");
                    throw new Exception();
                }
                primaryWords_EN.Add(word_EN);
            }
        }

        var r = new Random(5);

        var n = primaryWords_RU_EN.Count;
        int d = Model01.Constants.OldVectorLength;

        // TEMPCODE
        //Temp_BitWords_RU = WordsHelper.GetRandomOrderWords(primaryWords_RU_EN.Select(it => it.Item1).ToList(), r).ToArray();
        Temp_BitWords_RU = primaryWords_RU_EN.Select(it => it.Item1).ToArray();
        Temp_ProxBits_RU = new MatrixFloat(n, n);
        for (int i = 0; i < n; i += 1)
        {
            for (int j = 0; j < n; j += 1)
            {
                Temp_ProxBits_RU[i, j] = TensorPrimitives.Dot(
                    Temp_BitWords_RU[i].OldVectorNormalized,
                    Temp_BitWords_RU[j].OldVectorNormalized);
            }
        }
        Temp_Emb_RU = new MatrixFloat(d, n);
        for (int j = 0; j < n; j++)
        {
            var word = Temp_BitWords_RU[j];
            for (int i = 0; i < d; i++)
            {
                Temp_Emb_RU[i, j] = word.OldVectorNormalized[i];
            }
        }

        // TEMPCODE
        //Temp_BitWords_EN = WordsHelper.GetRandomOrderWords(primaryWords_RU_EN.Select(it => it.Item2).ToList(), r).ToArray();
        //Temp_BitWords_EN = primaryWords_RU_EN.Select(it => it.Item2).ToArray();
        Temp_BitWords_EN = WordsHelper.GetRandomOrderWords(primaryWords_RU_EN.Select(it => it.Item1).ToList(), r).ToArray();

        Temp_ProxBits_EN = new MatrixFloat(n, n);
        for (int i = 0; i < n; i += 1)
        {
            for (int j = 0; j < n; j += 1)
            {
                Temp_ProxBits_EN[i, j] = TensorPrimitives.Dot(
                    Temp_BitWords_EN[i].OldVectorNormalized,
                    Temp_BitWords_EN[j].OldVectorNormalized);
            }
        }
        Temp_Emb_EN = new MatrixFloat(d, n);
        for (int j = 0; j < n; j++)
        {
            var word = Temp_BitWords_EN[j];
            for (int i = 0; i < d; i++)
            {
                Temp_Emb_EN[i, j] = word.OldVectorNormalized[i];
            }
        }

        _loggersSet.UserFriendlyLogger.LogInformation($"Prepare3() Done. N = {n}");
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteArray(Mapping_RU_EN);
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
                    break;
            }
        }
    }

    public void CalculatePrimaryWordsMapping()
    {
        var stopwatch = Stopwatch.StartNew();

        // 1.Найти оптимальную матрицу поворота
        VectorSetAlignment vectorSetAlignment = new(_loggersSet);
        MatrixFloat R = vectorSetAlignment.FindOptimalRotation(Temp_Emb_RU, Temp_Emb_EN);

        // 2.Повернуть A: MatrixFloat rotatedA = CosineStructureAligner.ApplyRotation(R, A);
        MatrixFloat rotated_Emb_RU = vectorSetAlignment.ApplyRotation(R, Temp_Emb_RU);
        vectorSetAlignment.NormalizeColumns(rotated_Emb_RU); // На всякий случай

        // Уникальное сопоставление
        Mapping_RU_EN = vectorSetAlignment.ComputeUniqueMatching(rotated_Emb_RU, Temp_Emb_EN);

        stopwatch.Stop();
        _loggersSet.UserFriendlyLogger.LogInformation("CalculateMapping totally done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    public void DisplayWords()
    {
        List<string?[]> fileData = new List<string?[]>();

        for (int primaryWordIndex_RU = 0; primaryWordIndex_RU < Temp_BitWords_RU.Length; primaryWordIndex_RU += 1)
        {
            var word_RU = Temp_BitWords_RU[primaryWordIndex_RU].Name;            
            int primaryWordIndex_EN = Mapping_RU_EN[primaryWordIndex_RU];
            string word_EN = Temp_BitWords_EN[primaryWordIndex_EN].Name;
            _loggersSet.UserFriendlyLogger.LogInformation($"{word_RU} -> {word_EN}");
        }

        _loggersSet.UserFriendlyLogger.LogInformation($"DisplayWords() Done. D = {Temp_BitWords_RU.Length}");
    }

    public async void TranslatePrimaryWords()
    {
        List<string?[]> fileData = new List<string?[]>();

        for (int primaryWordIndex_RU = 0; primaryWordIndex_RU < Model01.Constants.DiscreteVectorLength; primaryWordIndex_RU += 1)
        {
            var word_RU = Temp_BitWords_RU[primaryWordIndex_RU].Name;

            var parts = word_RU.Split('_');
            if (parts.Length != 2)
                throw new InvalidOperationException();

            string word = parts[0];
            string pos = parts[1];

            // Получение перевода
            string word_EN = await TranslationHelper.TranslateWordAsync(word, _loggersSet);

            fileData.Add([word_RU, word_EN]);
            //fileData.Add([ word_RU.Substring(0, word_RU.IndexOf("_")) ]);
            //int primaryWordIndex_EN = Mapping_RU_EN[primaryWordIndex_RU];
            _loggersSet.UserFriendlyLogger.LogInformation($"{word_RU} -> {word_EN}");
        }

        CsvHelper.SaveCsvFile(Path.Combine(@"Data", FileName_PrimaryWords_RU_EN), fileData);
    }
}
