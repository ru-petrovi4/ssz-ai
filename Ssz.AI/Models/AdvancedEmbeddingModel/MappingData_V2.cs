using Avalonia.Media;
using MathNet.Numerics;
using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
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

public class MappingData_V2 : ISerializableModelObject
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

    public MatrixFloat Temp_ProxBits_EN = new();
    /// <summary>
    ///     Word for each bit.
    /// </summary>
    public Word[] Temp_BitWords_EN = null!;

    /// <summary>
    ///     Index RU -> Index EN
    /// </summary>
    public int[] Mapping_RU_EN = null!;

    public MappingData_V2(ILoggersSet loggersSet, LanguageInfo languageInfo_RU, LanguageInfo languageInfo_EN)
    {
        _loggersSet = loggersSet;
        LanguageInfo_RU = languageInfo_RU;
        LanguageInfo_EN = languageInfo_EN;        
    }

    public void GenerateOwnedData(int vectorLength)
    {
        Mapping_RU_EN = new int[Model01.Constants.DiscreteVectorLength];
    }

    public void Prepare()
    {
        Temp_ProxBits_RU = new MatrixFloat(Model01.Constants.DiscreteVectorLength, Model01.Constants.DiscreteVectorLength);
        Temp_BitWords_RU = new Word[Model01.Constants.DiscreteVectorLength];        
        foreach (var primaryWord in LanguageInfo_RU.Clusterization_AlgorithmData.PrimaryWords)
        {
            int prjectionIndex = LanguageInfo_RU.ProjectionOptimization_AlgorithmData.WordsProjectionIndices[primaryWord.Index];
            Temp_BitWords_RU[prjectionIndex] = primaryWord;
        }
        for (int i = 0; i < Model01.Constants.DiscreteVectorLength; i += 1)
        {
            for (int j = 0; j < Model01.Constants.DiscreteVectorLength; j += 1)
            {
                Temp_ProxBits_RU[i, j] = TensorPrimitives.CosineSimilarity(
                    LanguageInfo_RU.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_RU[i].Index],
                    LanguageInfo_RU.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_RU[j].Index]);
            }
        }

        Temp_ProxBits_EN = new MatrixFloat(Model01.Constants.DiscreteVectorLength, Model01.Constants.DiscreteVectorLength);
        Temp_BitWords_EN = new Word[Model01.Constants.DiscreteVectorLength];        
        foreach (var primaryWord in LanguageInfo_EN.Clusterization_AlgorithmData.PrimaryWords)
        {
            int prjectionIndex = LanguageInfo_EN.ProjectionOptimization_AlgorithmData.WordsProjectionIndices[primaryWord.Index];
            Temp_BitWords_EN[prjectionIndex] = primaryWord;
        }
        for (int i = 0; i < Model01.Constants.DiscreteVectorLength; i += 1)
        {
            for (int j = 0; j < Model01.Constants.DiscreteVectorLength; j += 1)
            {
                Temp_ProxBits_EN[i, j] = TensorPrimitives.CosineSimilarity(
                    LanguageInfo_EN.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_EN[i].Index],
                    LanguageInfo_EN.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_EN[j].Index]);
            }
        }
    }

    public void Prepare2()
    {
        var fileData = CsvHelper.LoadCsvFile(Path.Combine(@"Data", FileName_PrimaryWords_RU_EN), includeFiles: false);

        List<(Word, Word)> primaryWords_RU_EN = new(300);
        foreach (var line in fileData.Values)
        {
            string word_RU_String = line[0] ?? @"";
            string word_EN_String = (line[1] ?? @"").ToLowerInvariant().Trim();
            var word_RU = LanguageInfo_RU.Words.FirstOrDefault(w => w.Name == word_RU_String);
            var word_EN = LanguageInfo_EN.Words.FirstOrDefault(w => w.Name == word_EN_String);
            if (word_RU is not null && word_EN is not null)
            {
                primaryWords_RU_EN.Add((word_RU, word_EN));
            }
        }

        var d = primaryWords_RU_EN.Count;
        Temp_ProxBits_RU = new MatrixFloat(d, d);
        Temp_BitWords_RU = new Word[d];
        foreach (var i in Enumerable.Range(0, d))
        {      
            var it = primaryWords_RU_EN[i];
            Temp_BitWords_RU[i] = it.Item1;
        }
        for (int i = 0; i < d; i += 1)
        {
            for (int j = 0; j < d; j += 1)
            {
                Temp_ProxBits_RU[i, j] = TensorPrimitives.CosineSimilarity(
                    LanguageInfo_RU.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_RU[i].Index],
                    LanguageInfo_RU.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_RU[j].Index]);
            }
        }

        Temp_ProxBits_EN = new MatrixFloat(d, d);
        Temp_BitWords_EN = new Word[d];
        foreach (var i in Enumerable.Range(0, d))
        {
            var it = primaryWords_RU_EN[i];
            Temp_BitWords_EN[i] = it.Item2;
        }
        for (int i = 0; i < d; i += 1)
        {
            for (int j = 0; j < d; j += 1)
            {
                Temp_ProxBits_EN[i, j] = TensorPrimitives.CosineSimilarity(
                    LanguageInfo_EN.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_EN[i].Index],
                    LanguageInfo_EN.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_EN[j].Index]);
            }
        }
    }

    public void Prepare3()
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

        var r = new Random();

        var d = primaryWords_RU_EN.Count;
        
        Temp_BitWords_RU = WordsHelper.GetRandomOrderWords(primaryWords_RU_EN.Select(it => it.Item1).ToList(), r).ToArray();
        Temp_ProxBits_RU = new MatrixFloat(d, d);
        for (int i = 0; i < d; i += 1)
        {
            for (int j = 0; j < d; j += 1)
            {
                Temp_ProxBits_RU[i, j] = TensorPrimitives.Dot(
                    Temp_BitWords_RU[i].OldVectorNormalized,
                    Temp_BitWords_RU[j].OldVectorNormalized);
            }
        }
        
        Temp_BitWords_EN = WordsHelper.GetRandomOrderWords(primaryWords_RU_EN.Select(it => it.Item2).ToList(), r).ToArray();
        Temp_ProxBits_EN = new MatrixFloat(d, d);
        for (int i = 0; i < d; i += 1)
        {
            for (int j = 0; j < d; j += 1)
            {
                Temp_ProxBits_EN[i, j] = TensorPrimitives.Dot(
                    Temp_BitWords_EN[i].OldVectorNormalized,
                    Temp_BitWords_EN[j].OldVectorNormalized);
            }
        }

        _loggersSet.UserFriendlyLogger.LogInformation($"Prepare3() Done. D = {d}");
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

        // Строим матрицу стоимости сопоставления
        var costMatrix = BuildStructureCostMatrix(Temp_ProxBits_RU, Temp_ProxBits_EN);

        // Ищем взаимно однозначное соответствие с минимальной структурной ошибкой
        Mapping_RU_EN = HungarianAlgorithm.FindAssignments(costMatrix);

        stopwatch.Stop();
        _loggersSet.UserFriendlyLogger.LogInformation("CalculateMapping totally done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    /// <summary>
    /// Получить матрицу стоимости сопоставления по разнице попарных косинусных расстояний.
    /// </summary>
    /// <param name="distA">Матрица расстояний для первого множества.</param>
    /// <param name="distB">Матрица расстояний для второго множества.</param>
    /// <returns>Матрица стоимости размера N x N, где элемент (i,j) — "разница структур" при сопоставлении i и j.</returns>
    private static int[,] BuildStructureCostMatrix(MatrixFloat distA, MatrixFloat distB)
    {
        int N = distA.Dimensions[0];
        var costMatrix = new int[N, N];

        // Для каждого возможного соответствия рассчитываем сумму разниц по всем остальным парам
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float sum = 0f;
                for (int k = 0; k < N; k++)
                {
                    for (int l = 0; l < N; l++)
                    {
                        // Сравниваем структуру: расстояние между i, k в A и между j, l в B
                        sum += MathF.Abs(distA[i, k] - distB[j, l]);
                    }
                }
                costMatrix[i, j] = (int)(sum * 100.0f);
            }
        }
        return costMatrix;
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
