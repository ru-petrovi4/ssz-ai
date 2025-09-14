using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02.Dictionary;

/// <summary>
/// Класс словаря для хранения соответствия между словами и их индексами.
/// Обеспечивает двустороннее отображение слово ↔ индекс.
/// Оптимизирован для быстрого поиска и минимального потребления памяти.
/// </summary>
public class Dictionary
{
    private readonly List<string> _idToWord;
    private readonly System.Collections.Generic.Dictionary<string, int> _wordToId;
    private readonly string _language;

    /// <summary>
    /// Инициализация словаря из списка слов.
    /// </summary>
    /// <param name="words">Список слов в порядке убывания частотности</param>
    /// <param name="language">Код языка (например, "en", "es")</param>
    public Dictionary(List<string> words, string language)
    {
        _language = language;
        _idToWord = new List<string>(words);
        _wordToId = new System.Collections.Generic.Dictionary<string, int>(words.Count);

        // Построение обратного индекса
        for (int i = 0; i < words.Count; i++)
        {
            _wordToId[words[i]] = i;
        }
    }

    /// <summary>
    /// Инициализация словаря из файла.
    /// Каждая строка файла содержит одно слово.
    /// </summary>
    /// <param name="filePath">Путь к файлу словаря</param>
    /// <param name="language">Код языка</param>
    /// <param name="maxWords">Максимальное количество слов для загрузки</param>
    public Dictionary(string filePath, string language, int maxWords = -1)
    {
        _language = language;

        var words = new List<string>();
        using var reader = new StreamReader(filePath);

        string? line;
        while ((line = reader.ReadLine()) != null && (maxWords == -1 || words.Count < maxWords))
        {
            var trimmedWord = line.Trim();
            if (!string.IsNullOrEmpty(trimmedWord))
            {
                words.Add(trimmedWord);
            }
        }

        _idToWord = words;
        _wordToId = new System.Collections.Generic.Dictionary<string, int>(words.Count);

        for (int i = 0; i < words.Count; i++)
        {
            _wordToId[words[i]] = i;
        }
    }

    /// <summary>
    /// Получение слова по индексу.
    /// </summary>
    /// <param name="id">Индекс слова</param>
    /// <returns>Слово, соответствующее индексу</returns>
    /// <exception cref="ArgumentOutOfRangeException">Если индекс выходит за границы</exception>
    public string GetWord(int id)
    {
        if (id < 0 || id >= _idToWord.Count)
            throw new ArgumentOutOfRangeException(nameof(id), $"Индекс {id} выходит за границы словаря (0-{_idToWord.Count - 1})");

        return _idToWord[id];
    }

    /// <summary>
    /// Получение индекса слова.
    /// </summary>
    /// <param name="word">Слово для поиска</param>
    /// <returns>Индекс слова или -1 если слово не найдено</returns>
    public int GetId(string word)
    {
        return _wordToId.TryGetValue(word, out int id) ? id : -1;
    }

    /// <summary>
    /// Проверка наличия слова в словаре.
    /// </summary>
    /// <param name="word">Слово для проверки</param>
    /// <returns>True если слово присутствует в словаре</returns>
    public bool Contains(string word)
    {
        return _wordToId.ContainsKey(word);
    }

    /// <summary>
    /// Размер словаря.
    /// </summary>
    public int Length => _idToWord.Count;

    /// <summary>
    /// Код языка словаря.
    /// </summary>
    public string Language => _language;

    /// <summary>
    /// Получение всех слов словаря.
    /// </summary>
    public IReadOnlyList<string> Words => _idToWord.AsReadOnly();

    /// <summary>
    /// Создание подсловаря с первыми N наиболее частотными словами.
    /// </summary>
    /// <param name="maxWords">Максимальное количество слов</param>
    /// <returns>Новый словарь с ограниченным количеством слов</returns>
    public Dictionary CreateSubDictionary(int maxWords)
    {
        if (maxWords >= _idToWord.Count)
            return this;

        var subWords = _idToWord.Take(maxWords).ToList();
        return new Dictionary(subWords, _language);
    }

    /// <summary>
    /// Фильтрация словаря по минимальной длине слова.
    /// </summary>
    /// <param name="minLength">Минимальная длина слова</param>
    /// <returns>Новый отфильтрованный словарь</returns>
    public Dictionary FilterByLength(int minLength)
    {
        var filteredWords = _idToWord.Where(word => word.Length >= minLength).ToList();
        return new Dictionary(filteredWords, _language);
    }

    /// <summary>
    /// Экспорт словаря в файл.
    /// </summary>
    /// <param name="filePath">Путь к выходному файлу</param>
    public void Export(string filePath)
    {
        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using var writer = new StreamWriter(filePath);
        foreach (var word in _idToWord)
        {
            writer.WriteLine(word);
        }
    }

    /// <summary>
    /// Получение статистики словаря.
    /// </summary>
    /// <returns>Статистическая информация о словаре</returns>
    public DictionaryStats GetStatistics()
    {
        var lengths = _idToWord.Select(w => w.Length).ToList();

        return new DictionaryStats
        {
            TotalWords = _idToWord.Count,
            AverageWordLength = lengths.Average(),
            MinWordLength = lengths.Min(),
            MaxWordLength = lengths.Max(),
            UniqueWords = _idToWord.Distinct().Count(),
            Language = _language
        };
    }

    /// <summary>
    /// Поиск слов по префиксу.
    /// </summary>
    /// <param name="prefix">Префикс для поиска</param>
    /// <param name="maxResults">Максимальное количество результатов</param>
    /// <returns>Список слов с данным префиксом</returns>
    public List<string> FindByPrefix(string prefix, int maxResults = 10)
    {
        return _idToWord
            .Where(word => word.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            .Take(maxResults)
            .ToList();
    }

    /// <summary>
    /// Сравнение двух словарей на предмет пересечения.
    /// </summary>
    /// <param name="other">Другой словарь для сравнения</param>
    /// <returns>Статистика пересечения словарей</returns>
    public DictionaryIntersection GetIntersection(Dictionary other)
    {
        var intersection = _wordToId.Keys.Intersect(other._wordToId.Keys).ToList();
        var unionSize = _wordToId.Keys.Union(other._wordToId.Keys).Count();

        return new DictionaryIntersection
        {
            IntersectionSize = intersection.Count,
            UnionSize = unionSize,
            JaccardSimilarity = (double)intersection.Count / unionSize,
            CommonWords = intersection
        };
    }
}

/// <summary>
/// Статистическая информация о словаре.
/// </summary>
public class DictionaryStats
{
    public int TotalWords { get; set; }
    public double AverageWordLength { get; set; }
    public int MinWordLength { get; set; }
    public int MaxWordLength { get; set; }
    public int UniqueWords { get; set; }
    public string Language { get; set; } = "";
}

/// <summary>
/// Информация о пересечении двух словарей.
/// </summary>
public class DictionaryIntersection
{
    public int IntersectionSize { get; set; }
    public int UnionSize { get; set; }
    public double JaccardSimilarity { get; set; }
    public List<string> CommonWords { get; set; } = new();
}
