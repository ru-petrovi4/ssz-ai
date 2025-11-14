using Ssz.Utils;
using System;
using System.Collections.Generic;
using System.IO;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2.EmbeddingsEvaluation;

/// <summary>
/// Класс для загрузки датасетов оценки семантической близости
/// </summary>
public static class DatasetLoader
{
    /// <summary>
    /// Загружает датасет из CSV файла
    /// Формат: word1,word2,human_score
    /// Пример: "кот_S,кошка_S,9.0"
    /// </summary>
    /// <param name="filePath">Путь к CSV файлу с датасетом</param>
    /// <returns>Список пар слов с человеческими оценками</returns>
    public static List<WordPair> LoadDataset(string filePath, string tagsMappingFilePath)
    {
        //Console.WriteLine($"Загрузка датасета из файла: {filePath}");

        var tagsMapping = CsvHelper.LoadCsvFile(tagsMappingFilePath, false);

        List<WordPair> wordPairs = new List<WordPair>();

        using (StreamReader reader = new StreamReader(filePath))
        {
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                if (line.StartsWith("#"))
                    continue;

                WordPair? pair = ParseLine(line, tagsMapping);
                if (pair != null)
                {
                    wordPairs.Add(pair);
                }
            }
        }

        //Console.WriteLine($"Загружено пар слов: {wordPairs.Count}");
        return wordPairs;
    }

    /// <summary>
    /// Парсит строку CSV и создает объект WordPair
    /// </summary>
    /// <param name="line">Строка в формате: word1,word2,score</param>
    /// <returns>Объект WordPair или null при ошибке парсинга</returns>
    private static WordPair? ParseLine(string line, CaseInsensitiveOrderedDictionary<List<string?>> tagsMapping)
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            return null;
        }

        // Разделяем строку по запятой или табуляции
        string[] parts = line.Split(new[] { ',', '\t' },
            StringSplitOptions.RemoveEmptyEntries);

        // Проверяем, что строка содержит три части: слово1, слово2, оценка
        if (parts.Length < 3)
        {
            return null;
        }

        // Парсим оценку человека
        if (!double.TryParse(parts[2].Trim(), out double humanScore))
        {
            return null;
        }

        return new WordPair
        {
            Word1 = Normalize(parts[0], tagsMapping),
            Word2 = Normalize(parts[1], tagsMapping),
            HumanScore = humanScore,
            CosineSimilarity = 0.0  // Будет вычислена позже
        };
    }

    private static string Normalize(string v, CaseInsensitiveOrderedDictionary<List<string?>> tagsMapping)
    {
        v = v.Trim();
        var i = v.IndexOf('_');
        if (i > 0)
        {
            string tag = v.Substring(i + 1);
            string wordShortName = v.Substring(0, i);
            if (tagsMapping.TryGetValue(tag, out var l) && l.Count > 1)
            {
                tag = l[1] ?? @"";
            }
            v = wordShortName + "_" + tag;
        }
        return v;
    }
}

/// <summary>
/// Класс для хранения информации о паре слов и оценках сходства
/// </summary>
public class WordPair
{
    // Первое слово в паре (лемматизированное с тегом части речи)
    public string Word1 { get; set; } = null!;

    // Второе слово в паре (лемматизированное с тегом части речи)
    public string Word2 { get; set; } = null!;

    // Оценка сходства, выставленная человеком (gold standard)
    // Обычно в диапазоне от 0.0 до 10.0
    public double HumanScore { get; set; }

    // Косинусная близость, вычисленная по эмбеддингам
    // Значение в диапазоне от -1.0 до 1.0
    public double CosineSimilarity { get; set; }
}
