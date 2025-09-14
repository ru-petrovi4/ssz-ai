using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Dictionary;
using Ssz.Utils.Logging;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Utils;

/// <summary>
/// Загрузчик эмбеддингов из текстовых файлов.
/// Поддерживает форматы word2vec, GloVe и fastText.
/// Оптимизирован для загрузки больших файлов с эффективным использованием памяти.
/// </summary>
public static class EmbeddingLoader
{
    /// <summary>
    /// Загрузка эмбеддингов из файла с автоматическим определением формата.
    /// </summary>
    /// <param name="filePath">Путь к файлу эмбеддингов</param>
    /// <param name="maxVocab">Максимальный размер словаря (-1 для неограниченного)</param>
    /// <param name="embeddingDim">Ожидаемая размерность эмбеддингов</param>
    /// <param name="language">Язык для создания словаря</param>
    /// <returns>Кортеж (словарь, матрица эмбеддингов)</returns>
    public static (Dictionary.Dictionary dictionary, MatrixFloat embeddings) LoadEmbeddings(
        string filePath, int maxVocab, int embeddingDim, string language)
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;
        logger.LogInformation($"Загрузка эмбеддингов из {filePath}...");

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Файл эмбеддингов не найден: {filePath}");

        // Определение формата файла по первой строке
        var format = DetectEmbeddingFormat(filePath);
        logger.LogInformation($"Обнаружен формат: {format}");

        return format switch
        {
            EmbeddingFormat.Word2Vec => LoadWord2VecFormat(filePath, maxVocab, embeddingDim, language),
            EmbeddingFormat.GloVe => LoadGloVeFormat(filePath, maxVocab, embeddingDim, language),
            EmbeddingFormat.FastText => LoadFastTextFormat(filePath, maxVocab, embeddingDim, language),
            _ => throw new NotSupportedException($"Неподдерживаемый формат эмбеддингов: {format}")
        };
    }

    /// <summary>
    /// Перечисление поддерживаемых форматов эмбеддингов.
    /// </summary>
    private enum EmbeddingFormat
    {
        Word2Vec,  // Первая строка содержит размер словаря и размерность
        GloVe,     // Каждая строка: слово vector1 vector2 ... vectorN
        FastText   // Похож на word2vec, но с дополнительными метаданными
    }

    /// <summary>
    /// Автоматическое определение формата файла эмбеддингов по первой строке.
    /// </summary>
    /// <param name="filePath">Путь к файлу</param>
    /// <returns>Обнаруженный формат</returns>
    private static EmbeddingFormat DetectEmbeddingFormat(string filePath)
    {
        using var reader = new StreamReader(filePath, Encoding.UTF8);
        var firstLine = reader.ReadLine();

        if (string.IsNullOrEmpty(firstLine))
            throw new InvalidDataException("Файл эмбеддингов пуст");

        var parts = firstLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        // Word2Vec формат: первая строка содержит два числа (vocab_size embedding_dim)
        if (parts.Length == 2 && int.TryParse(parts[0], out _) && int.TryParse(parts[1], out _))
        {
            return EmbeddingFormat.Word2Vec;
        }

        // GloVe формат: первая строка содержит слово и векторы
        if (parts.Length > 2)
        {
            // Проверяем, что элементы после первого являются числами
            bool allFloats = parts.Skip(1).All(p => float.TryParse(p, out _));
            if (allFloats)
            {
                return EmbeddingFormat.GloVe;
            }
        }

        // По умолчанию предполагаем FastText
        return EmbeddingFormat.FastText;
    }

    /// <summary>
    /// Загрузка эмбеддингов в формате Word2Vec.
    /// Формат: первая строка содержит размер словаря и размерность,
    /// затем каждая строка содержит слово и его вектор.
    /// </summary>
    private static (Dictionary.Dictionary dictionary, MatrixFloat embeddings) LoadWord2VecFormat(
        string filePath, int maxVocab, int embeddingDim, string language)
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;

        using var reader = new StreamReader(filePath, Encoding.UTF8);

        // Чтение заголовка
        var headerLine = reader.ReadLine();
        var headerParts = headerLine!.Split(' ');
        int vocabSize = int.Parse(headerParts[0]);
        int fileDim = int.Parse(headerParts[1]);

        if (embeddingDim > 0 && fileDim != embeddingDim)
        {
            logger.LogWarning($"Размерность в файле ({fileDim}) не соответствует ожидаемой ({embeddingDim})");
        }

        embeddingDim = fileDim;

        // Ограничение размера словаря
        if (maxVocab > 0)
        {
            vocabSize = Math.Min(vocabSize, maxVocab);
        }

        logger.LogInformation($"Загрузка {vocabSize} слов с размерностью {embeddingDim}");

        // Инициализация структур данных
        var words = new List<string>(vocabSize);
        var embeddings = new MatrixFloat(new[] { vocabSize, embeddingDim });

        // Чтение эмбеддингов
        int loadedWords = 0;
        string? line;

        while ((line = reader.ReadLine()) != null && loadedWords < vocabSize)
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            if (parts.Length != embeddingDim + 1)
            {
                logger.LogWarning($"Пропуск строки {loadedWords + 2}: неверное количество элементов");
                continue;
            }

            string word = parts[0];

            // Парсинг вектора эмбеддинга
            try
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    embeddings[loadedWords, j] = float.Parse(parts[j + 1]);
                }

                words.Add(word);
                loadedWords++;

                // Прогресс-бар для больших файлов
                if (loadedWords % 10000 == 0)
                {
                    logger.LogDebug($"Загружено {loadedWords}/{vocabSize} слов...");
                }
            }
            catch (FormatException)
            {
                logger.LogWarning($"Пропуск слова '{word}': ошибка парсинга числовых значений");
            }
        }

        // Обрезка матрицы если загружено меньше слов
        if (loadedWords < vocabSize)
        {
            var trimmedEmbeddings = new MatrixFloat(new[] { loadedWords, embeddingDim });
            Array.Copy(embeddings.Data, trimmedEmbeddings.Data, loadedWords * embeddingDim);
            embeddings = trimmedEmbeddings;
        }

        // Создание словаря
        var dictionary = new Dictionary.Dictionary(words, language);

        logger.LogInformation($"Успешно загружено {loadedWords} эмбеддингов");
        return (dictionary, embeddings);
    }

    /// <summary>
    /// Загрузка эмбеддингов в формате GloVe.
    /// Формат: каждая строка содержит слово и его вектор.
    /// </summary>
    private static (Dictionary.Dictionary dictionary, MatrixFloat embeddings) LoadGloVeFormat(
        string filePath, int maxVocab, int embeddingDim, string language)
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;

        // Первый проход: определение размерности и подсчет строк
        int actualEmbeddingDim = 0;
        int totalLines = 0;

        using (var reader = new StreamReader(filePath, Encoding.UTF8))
        {
            var firstLine = reader.ReadLine();
            if (!string.IsNullOrEmpty(firstLine))
            {
                var parts = firstLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                actualEmbeddingDim = parts.Length - 1; // Вычитаем слово
                totalLines = 1;

                // Подсчет остальных строк
                while (reader.ReadLine() != null)
                {
                    totalLines++;
                }
            }
        }

        if (embeddingDim > 0 && actualEmbeddingDim != embeddingDim)
        {
            logger.LogWarning($"Размерность в файле ({actualEmbeddingDim}) не соответствует ожидаемой ({embeddingDim})");
        }

        embeddingDim = actualEmbeddingDim;
        int vocabSize = maxVocab > 0 ? Math.Min(totalLines, maxVocab) : totalLines;

        logger.LogInformation($"Загрузка {vocabSize} слов с размерностью {embeddingDim}");

        // Второй проход: загрузка данных
        var words = new List<string>(vocabSize);
        var embeddings = new MatrixFloat(new[] { vocabSize, embeddingDim });

        using var reader2 = new StreamReader(filePath, Encoding.UTF8);
        int loadedWords = 0;
        string? line;

        while ((line = reader2.ReadLine()) != null && loadedWords < vocabSize)
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            if (parts.Length != embeddingDim + 1)
            {
                logger.LogWarning($"Пропуск строки {loadedWords + 1}: неверное количество элементов");
                continue;
            }

            string word = parts[0];

            try
            {
                // Парсинг вектора
                for (int j = 0; j < embeddingDim; j++)
                {
                    embeddings[loadedWords, j] = float.Parse(parts[j + 1]);
                }

                words.Add(word);
                loadedWords++;

                if (loadedWords % 10000 == 0)
                {
                    logger.LogDebug($"Загружено {loadedWords}/{vocabSize} слов...");
                }
            }
            catch (FormatException)
            {
                logger.LogWarning($"Пропуск слова '{word}': ошибка парсинга");
            }
        }

        var dictionary = new Dictionary.Dictionary(words, language);

        logger.LogInformation($"Успешно загружено {loadedWords} эмбеддингов GloVe");
        return (dictionary, embeddings);
    }

    /// <summary>
    /// Загрузка эмбеддингов в формате FastText.
    /// Аналогично Word2Vec, но с дополнительной обработкой метаданных.
    /// </summary>
    private static (Dictionary.Dictionary dictionary, MatrixFloat embeddings) LoadFastTextFormat(
        string filePath, int maxVocab, int embeddingDim, string language)
    {
        // FastText обычно совместим с форматом Word2Vec
        return LoadWord2VecFormat(filePath, maxVocab, embeddingDim, language);
    }

    /// <summary>
    /// Экспорт эмбеддингов в файл после обучения.
    /// </summary>
    /// <param name="embeddings">Матрица эмбеддингов для экспорта</param>
    /// <param name="dictionary">Словарь с соответствием индексов и слов</param>
    /// <param name="filePath">Путь к выходному файлу</param>
    /// <param name="format">Формат экспорта (txt/bin)</param>
    public static void ExportEmbeddings(MatrixFloat embeddings, Dictionary.Dictionary dictionary,
                                      string filePath, string format = "txt")
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;
        logger.LogInformation($"Экспорт эмбеддингов в {filePath} (формат: {format})");

        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        switch (format.ToLower())
        {
            case "txt":
                ExportTextFormat(embeddings, dictionary, filePath);
                break;
            case "bin":
                ExportBinaryFormat(embeddings, dictionary, filePath);
                break;
            default:
                throw new ArgumentException($"Неподдерживаемый формат экспорта: {format}");
        }

        logger.LogInformation("Экспорт завершен успешно");
    }

    /// <summary>
    /// Экспорт в текстовом формате (совместимом с word2vec).
    /// </summary>
    private static void ExportTextFormat(MatrixFloat embeddings, Dictionary.Dictionary dictionary, string filePath)
    {
        using var writer = new StreamWriter(filePath, false, Encoding.UTF8);

        int vocabSize = embeddings.Dimensions[0];
        int embeddingDim = embeddings.Dimensions[1];

        // Запись заголовка
        writer.WriteLine($"{vocabSize} {embeddingDim}");

        // Запись эмбеддингов
        for (int i = 0; i < vocabSize; i++)
        {
            string word = dictionary.GetWord(i);
            writer.Write(word);

            for (int j = 0; j < embeddingDim; j++)
            {
                writer.Write($" {embeddings[i, j]:G9}"); // G9 для точности
            }

            writer.WriteLine();
        }
    }

    /// <summary>
    /// Экспорт в бинарном формате для экономии места.
    /// </summary>
    private static void ExportBinaryFormat(MatrixFloat embeddings, Dictionary.Dictionary dictionary, string filePath)
    {
        using var stream = new FileStream(filePath, FileMode.Create);
        using var writer = new BinaryWriter(stream);

        int vocabSize = embeddings.Dimensions[0];
        int embeddingDim = embeddings.Dimensions[1];

        // Запись заголовка
        writer.Write(vocabSize);
        writer.Write(embeddingDim);

        // Запись слов и эмбеддингов
        for (int i = 0; i < vocabSize; i++)
        {
            string word = dictionary.GetWord(i);
            writer.Write(word);

            // Запись эмбеддинга как массива float
            for (int j = 0; j < embeddingDim; j++)
            {
                writer.Write(embeddings[i, j]);
            }
        }
    }

    /// <summary>
    /// Валидация загруженных эмбеддингов.
    /// Проверяет нормы векторов и наличие NaN/Inf значений.
    /// </summary>
    /// <param name="embeddings">Матрица эмбеддингов</param>
    /// <param name="dictionary">Словарь</param>
    /// <returns>True если эмбеддинги корректны</returns>
    public static bool ValidateEmbeddings(MatrixFloat embeddings, Dictionary.Dictionary dictionary)
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;
        logger.LogInformation("Валидация загруженных эмбеддингов...");

        int vocabSize = embeddings.Dimensions[0];
        int embeddingDim = embeddings.Dimensions[1];

        int invalidVectors = 0;
        int zeroNormVectors = 0;

        for (int i = 0; i < vocabSize; i++)
        {
            float normSquared = 0.0f;
            bool hasInvalidValue = false;

            for (int j = 0; j < embeddingDim; j++)
            {
                float value = embeddings[i, j];

                if (float.IsNaN(value) || float.IsInfinity(value))
                {
                    hasInvalidValue = true;
                    break;
                }

                normSquared += value * value;
            }

            if (hasInvalidValue)
            {
                invalidVectors++;
                logger.LogWarning($"Слово '{dictionary.GetWord(i)}' содержит NaN/Inf значения");
            }
            else if (normSquared < 1e-8f)
            {
                zeroNormVectors++;
                logger.LogWarning($"Слово '{dictionary.GetWord(i)}' имеет нулевую норму");
            }
        }

        logger.LogInformation($"Валидация завершена: {invalidVectors} некорректных, {zeroNormVectors} нулевых векторов");

        return invalidVectors == 0;
    }
}
