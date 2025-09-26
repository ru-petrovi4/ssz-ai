using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Models;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Training;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Evaluation;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Utils;
using Microsoft.Extensions.Logging;
using TorchSharp;
using static TorchSharp.torch;
using MathNet.Numerics.Statistics;
using static TorchSharp.torch.nn;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Evaluation;

/// <summary>
/// Высокопроизводительные утилиты для оценки кросс-лингвальных эмбеддингов
/// Реализация вспомогательных функций из utils.py, wordsim.py, word_translation.py, sent_translation.py
/// </summary>
public static class EvaluationUtils
{
    #region Constants

    /// <summary>
    /// Размер батча по умолчанию для GPU операций
    /// </summary>
    private const int DefaultBatchSize = 1024;

    #endregion

    #region Word Pair Loading

    /// <summary>
    /// Загружает пары слов и их оценки из файла
    /// Реализация функции get_word_pairs из wordsim.py
    /// </summary>
    /// <param name="filePath">Путь к файлу с парами слов</param>
    /// <param name="lower">Приводить ли к нижнему регистру</param>
    /// <returns>Список пар слов с оценками</returns>
    public static async Task<List<(string word1, string word2, double score)>> LoadWordPairsAsync(
        string filePath, bool lower = true)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Файл не найден: {filePath}");

        var wordPairs = new List<(string, string, double)>();

        await foreach (var line in File.ReadLinesAsync(filePath))
        {
            var trimmedLine = line.Trim();
            if (string.IsNullOrEmpty(trimmedLine)) continue;

            var processedLine = lower ? trimmedLine.ToLowerInvariant() : trimmedLine;
            var parts = processedLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            // Игнорируем фразы, рассматриваем только отдельные слова
            if (parts.Length != 3)
            {
                // Проверяем исключения для SEMEVAL17 и EN-IT_MWS353
                var fileName = Path.GetFileName(filePath);
                if (parts.Length > 3 && (fileName.Contains("SEMEVAL17") || fileName.Contains("EN-IT_MWS353")))
                    continue;
                else if (parts.Length != 3)
                    continue;
            }

            if (double.TryParse(parts[2], out var score))
            {
                wordPairs.Add((parts[0], parts[1], score));
            }
        }

        return wordPairs;
    }

    #endregion

    #region Word ID Lookup

    /// <summary>
    /// Получает ID слова в словаре с обработкой регистра
    /// Реализация функции get_word_id из wordsim.py
    /// </summary>
    /// <param name="word">Слово для поиска</param>
    /// <param name="dictionary">Словарь</param>
    /// <param name="lower">Использовать ли нижний регистр</param>
    /// <returns>ID слова или null если не найдено</returns>
    public static int? GetWordId(string word, Dictionary dictionary, bool lower)
    {
        if (dictionary.WordToId.TryGetValue(word, out var id))
            return id;

        if (!lower)
        {
            // Попробуем с заглавной буквы
            var capitalized = word.Length > 0 ? char.ToUpperInvariant(word[0]) + word.Substring(1) : word;
            if (dictionary.WordToId.TryGetValue(capitalized, out id))
                return id;

            // Попробуем Title Case
            var titleCase = System.Globalization.CultureInfo.CurrentCulture.TextInfo.ToTitleCase(word.ToLowerInvariant());
            if (dictionary.WordToId.TryGetValue(titleCase, out id))
                return id;
        }

        return null;
    }

    #endregion

    #region Cosine Similarity

    /// <summary>
    /// Вычисляет косинусное сходство между двумя векторами
    /// Использует System.Numerics.Tensors для SIMD оптимизации
    /// </summary>
    /// <param name="vector1">Первый вектор</param>
    /// <param name="vector2">Второй вектор</param>
    /// <returns>Косинусное сходство</returns>
    public static double ComputeCosineSimilarity(ReadOnlySpan<float> vector1, ReadOnlySpan<float> vector2)
    {
        if (vector1.Length != vector2.Length)
            throw new ArgumentException("Векторы должны быть одинаковой длины");

        if (vector1.Length == 0)
            return 0.0;

        // Используем System.Numerics.Tensors для SIMD операций
        var dotProduct = System.Numerics.Tensors.TensorPrimitives.Dot(vector1, vector2);
        var norm1 = Math.Sqrt(System.Numerics.Tensors.TensorPrimitives.SumOfSquares(vector1));
        var norm2 = Math.Sqrt(System.Numerics.Tensors.TensorPrimitives.SumOfSquares(vector2));

        var magnitude = norm1 * norm2;
        return magnitude > 0 ? dotProduct / magnitude : 0.0;
    }

    /// <summary>
    /// Батчевое вычисление косинусных сходств между векторами
    /// </summary>
    /// <param name="embeddings1">Первый набор эмбеддингов [n, dim]</param>
    /// <param name="embeddings2">Второй набор эмбеддингов [m, dim]</param>
    /// <param name="batchSize">Размер батча</param>
    /// <returns>Матрица сходств [n, m]</returns>
    public static async Task<Tensor> ComputeBatchCosineSimilaritiesAsync(
        Tensor embeddings1, Tensor embeddings2, int batchSize = DefaultBatchSize)
    {
        var n = (int)embeddings1.size(0);
        var m = (int)embeddings2.size(0);
        var similarities = zeros(n, m, dtype: ScalarType.Float32, device: embeddings1.device);

        // Нормализуем эмбеддинги один раз
        var normalizedEmb1 = functional.normalize(embeddings1, p: 2, dim: 1);
        var normalizedEmb2 = functional.normalize(embeddings2, p: 2, dim: 1);

        // Батчевое вычисление для экономии памяти
        for (int i = 0; i < n; i += batchSize)
        {
            var endI = Math.Min(n, i + batchSize);
            var batch1 = normalizedEmb1[TensorIndex.Slice(i, endI)];

            for (int j = 0; j < m; j += batchSize)
            {
                var endJ = Math.Min(m, j + batchSize);
                var batch2 = normalizedEmb2[TensorIndex.Slice(j, endJ)];

                // Скалярное произведение нормализованных векторов = косинусное сходство
                var batchSimilarities = batch1.mm(batch2.transpose(0, 1));
                similarities[TensorIndex.Slice(i, endI), TensorIndex.Slice(j, endJ)] = batchSimilarities;
            }

            // Даем контроль другим задачам
            if (i % (batchSize * 10) == 0)
                await Task.Yield();
        }

        return similarities;
    }

    #endregion

    #region Spearman Correlation

    /// <summary>
    /// Вычисляет корреляцию Спирмена для монолингвального или кросс-лингвального семантического сходства
    /// Реализация функции get_spearman_rho из wordsim.py
    /// </summary>
    /// <param name="wordPairs">Пары слов с золотыми оценками</param>
    /// <param name="dictionary1">Первый словарь</param>
    /// <param name="embeddings1">Первые эмбеддинги</param>
    /// <param name="lower">Использовать ли нижний регистр</param>
    /// <param name="dictionary2">Второй словарь (опционально для кросс-лингвального)</param>
    /// <param name="embeddings2">Второй набор эмбеддингов (опционально)</param>
    /// <returns>Корреляция, количество найденных и не найденных пар</returns>
    public static (double correlation, int found, int notFound) ComputeSpearmanCorrelation(
        IEnumerable<(string word1, string word2, double goldScore)> wordPairs,
        Dictionary dictionary1, Tensor embeddings1, bool lower,
        Dictionary? dictionary2 = null, Tensor? embeddings2 = null)
    {
        dictionary2 ??= dictionary1;
        embeddings2 ??= embeddings1;

        var predictions = new List<double>();
        var goldScores = new List<double>();
        int notFound = 0;

        var embData1 = embeddings1.data<float>().ToArray();
        var embData2 = embeddings2.data<float>().ToArray();
        var embDim = (int)embeddings1.size(1);

        foreach (var (word1, word2, goldScore) in wordPairs)
        {
            var id1 = GetWordId(word1, dictionary1, lower);
            var id2 = GetWordId(word2, dictionary2, lower);

            if (id1 == null || id2 == null)
            {
                notFound++;
                continue;
            }

            // Извлекаем векторы
            var vector1 = new ReadOnlySpan<float>(embData1, id1.Value * embDim, embDim);
            var vector2 = new ReadOnlySpan<float>(embData2, id2.Value * embDim, embDim);

            // Вычисляем косинусное сходство
            var similarity = ComputeCosineSimilarity(vector1, vector2);
            predictions.Add(similarity);
            goldScores.Add(goldScore);
        }

        if (predictions.Count < 2)
            return (0.0, predictions.Count, notFound);

        // Вычисляем корреляцию Спирмена
        var correlation = Correlation.Spearman(predictions, goldScores);
        return (correlation, predictions.Count, notFound);
    }

    #endregion

    #region Average Distance Computation

    /// <summary>
    /// Вычисляет средние расстояния до k ближайших соседей
    /// Реализация функции get_nn_avg_dist из utils.py с оптимизациями для .NET 9
    /// </summary>
    /// <param name="embeddings">Эмбеддинги-ключи</param>
    /// <param name="queries">Запросные эмбеддинги</param>
    /// <param name="k">Количество ближайших соседей</param>
    /// <param name="batchSize">Размер батча для обработки</param>
    /// <returns>Средние расстояния для каждого запроса</returns>
    public static async Task<Tensor> ComputeNearestNeighborAverageDistancesAsync(
        Tensor embeddings, Tensor queries, int k, int batchSize = DefaultBatchSize)
    {
        var queryCount = (int)queries.size(0);
        var keyCount = (int)embeddings.size(0);
        var actualK = Math.Min(k, keyCount);

        var avgDistances = zeros(queryCount, dtype: ScalarType.Float32, device: queries.device);

        // Нормализуем эмбеддинги для косинусного сходства
        var normalizedEmbeddings = functional.normalize(embeddings, p: 2, dim: 1);
        var normalizedQueries = functional.normalize(queries, p: 2, dim: 1);

        // Транспонируем ключи один раз для эффективного матричного умножения
        var embeddingsT = normalizedEmbeddings.transpose(0, 1);

        for (int i = 0; i < queryCount; i += batchSize)
        {
            var endIdx = Math.Min(queryCount, i + batchSize);
            var batchQueries = normalizedQueries[TensorIndex.Slice(i, endIdx)];

            // Вычисляем косинусные сходства для батча
            var similarities = batchQueries.mm(embeddingsT);

            // Находим k лучших сходств
            var (topSimilarities, _) = similarities.topk(actualK, dim: 1, largest: true, sorted: false);

            // Вычисляем среднее
            avgDistances[TensorIndex.Slice(i, endIdx)] = topSimilarities.mean(dimensions: [ 1 ]); // VALFIX

            // Периодически передаем управление для отзывчивости
            if (i % (batchSize * 5) == 0)
                await Task.Yield();
        }

        return avgDistances;
    }

    #endregion

    #region Dictionary Loading

    /// <summary>
    /// Загружает словарь идентичных символьных строк
    /// Реализация функции load_identical_char_dico из word_translation.py
    /// </summary>
    /// <param name="dictionary1">Первый словарь</param>
    /// <param name="dictionary2">Второй словарь</param>
    /// <returns>Тензор пар индексов</returns>
    public static Tensor LoadIdenticalCharacterDictionary(Dictionary dictionary1, Dictionary dictionary2)
    {
        var pairs = dictionary1.WordToId.Keys
            .Where(word => dictionary2.WordToId.ContainsKey(word))
            .Select(word => (dictionary1.WordToId[word], dictionary2.WordToId[word]))
            .OrderBy(pair => pair.Item1) // Сортируем по частоте исходных слов
            .ToList();

        if (pairs.Count == 0)
        {
            throw new InvalidOperationException(
                "Не найдено идентичных символьных строк. Пожалуйста, укажите словарь.");
        }

        var data = new long[pairs.Count * 2];
        for (int i = 0; i < pairs.Count; i++)
        {
            data[i * 2] = pairs[i].Item1;
            data[i * 2 + 1] = pairs[i].Item2;
        }

        return tensor(data, dtype: ScalarType.Int64).reshape(pairs.Count, 2);
    }

    /// <summary>
    /// Загружает билингвальный словарь из файла
    /// Реализация функции load_dictionary из word_translation.py
    /// </summary>
    /// <param name="filePath">Путь к файлу словаря</param>
    /// <param name="dictionary1">Первый словарь</param>
    /// <param name="dictionary2">Второй словарь</param>
    /// <param name="logger">Логгер для отчетности</param>
    /// <returns>Тензор пар индексов, отсортированных по частоте исходных слов</returns>
    public static async Task<Tensor> LoadBilingualDictionaryAsync(
        string filePath, Dictionary dictionary1, Dictionary dictionary2, ILogger? logger = null)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Файл словаря не найден: {filePath}");

        var pairs = new List<(string word1, string word2, int sourceId, int targetId)>();
        int notFound = 0;
        int notFound1 = 0;
        int notFound2 = 0;

        await foreach (var line in File.ReadLinesAsync(filePath))
        {
            var trimmedLine = line.Trim();
            
            // Проверяем что строка в нижнем регистре (требование MUSE)
            if (trimmedLine != trimmedLine.ToLowerInvariant())
            {
                logger?.LogWarning($"Строка не в нижнем регистре: {trimmedLine}");
            }

            var parts = trimmedLine.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2)
            {
                logger?.LogWarning($"Не удалось распарсить строку: {line}");
                continue;
            }

            var word1 = parts[0];
            var word2 = parts[1];

            var sourceId = GetWordId(word1, dictionary1, true);
            var targetId = GetWordId(word2, dictionary2, true);

            if (sourceId.HasValue && targetId.HasValue)
            {
                pairs.Add((word1, word2, sourceId.Value, targetId.Value));
            }
            else
            {
                notFound++;
                if (!sourceId.HasValue) notFound1++;
                if (!targetId.HasValue) notFound2++;
            }
        }

        logger?.LogInformation($"Найдено {pairs.Count} пар слов в словаре ({pairs.Select(p => p.word1).Distinct().Count()} уникальных). " +
                             $"{notFound} других пар содержали хотя бы одно неизвестное слово " +
                             $"({notFound1} в первом языке, {notFound2} во втором языке)");

        // Сортируем по частоте исходных слов (меньший индекс = более частое слово)
        var sortedPairs = pairs.OrderBy(p => p.sourceId).ToList();

        var data = new long[sortedPairs.Count * 2];
        for (int i = 0; i < sortedPairs.Count; i++)
        {
            data[i * 2] = sortedPairs[i].sourceId;
            data[i * 2 + 1] = sortedPairs[i].targetId;
        }

        return tensor(data, dtype: ScalarType.Int64).reshape(sortedPairs.Count, 2);
    }

    #endregion

    #region Bag-of-Words with IDF

    /// <summary>
    /// Создает представления предложений с взвешиванием IDF
    /// Реализация функции bow_idf из utils.py
    /// </summary>
    /// <param name="sentences">Массив предложений (каждое как массив слов)</param>
    /// <param name="wordVectors">Словарь векторов слов</param>
    /// <param name="idfWeights">IDF веса для слов (опционально)</param>
    /// <returns>Матрица векторов предложений</returns>
    public static float[][] CreateBagOfWordsIdf(
        string[][] sentences, 
        Dictionary<string, float[]> wordVectors,
        Dictionary<string, double>? idfWeights = null)
    {
        var sentenceVectors = new List<float[]>();

        foreach (var sentence in sentences)
        {
            // Получаем уникальные слова предложения, которые есть в словаре
            var validWords = sentence.Distinct()
                .Where(w => wordVectors.ContainsKey(w) && (idfWeights == null || idfWeights.ContainsKey(w)))
                .ToArray();

            if (validWords.Length > 0)
            {
                var embeddingDim = wordVectors.Values.First().Length;
                var sentenceVector = new float[embeddingDim];

                if (idfWeights != null)
                {
                    // Взвешенное суммирование с IDF весами
                    double totalWeight = 0.0;
                    foreach (var word in validWords)
                    {
                        var wordVector = wordVectors[word];
                        var weight = idfWeights[word];

                        for (int i = 0; i < embeddingDim; i++)
                        {
                            sentenceVector[i] += (float)(wordVector[i] * weight);
                        }
                        totalWeight += weight;
                    }

                    // Нормализуем на общий вес
                    if (totalWeight > 0)
                    {
                        var normFactor = (float)(1.0 / totalWeight);
                        for (int i = 0; i < embeddingDim; i++)
                        {
                            sentenceVector[i] *= normFactor;
                        }
                    }
                }
                else
                {
                    // Простое усреднение без IDF
                    foreach (var word in validWords)
                    {
                        var wordVector = wordVectors[word];
                        for (int i = 0; i < embeddingDim; i++)
                        {
                            sentenceVector[i] += wordVector[i];
                        }
                    }

                    // Нормализуем на количество слов
                    var normFactor = 1.0f / validWords.Length;
                    for (int i = 0; i < embeddingDim; i++)
                    {
                        sentenceVector[i] *= normFactor;
                    }
                }

                sentenceVectors.Add(sentenceVector);
            }
            else
            {
                // Если нет подходящих слов, используем случайный вектор
                var randomVector = wordVectors.Values.FirstOrDefault()?.ToArray() ?? new float[300];
                sentenceVectors.Add(randomVector);
            }
        }

        return sentenceVectors.ToArray();
    }

    #endregion

    #region IDF Computation

    /// <summary>
    /// Вычисляет IDF (Inverse Document Frequency) веса для корпуса
    /// Реализация функции get_idf из utils.py
    /// </summary>
    /// <param name="corpus">Корпус предложений по языкам</param>
    /// <param name="language1">Первый язык</param>
    /// <param name="language2">Второй язык</param>
    /// <param name="nIdf">Количество предложений для IDF вычислений</param>
    /// <returns>IDF веса по языкам</returns>
    public static Dictionary<string, Dictionary<string, double>> ComputeIdfWeights(
        Dictionary<string, string[][]> corpus, string language1, string language2, int nIdf = 300000)
    {
        var idf = new Dictionary<string, Dictionary<string, double>>
        {
            [language1] = new Dictionary<string, double>(),
            [language2] = new Dictionary<string, double>()
        };

        int k = 0;
        foreach (var language in new[] { language1, language2 })
        {
            if (!corpus.ContainsKey(language))
            {
                k++;
                continue;
            }

            var sentences = corpus[language];
            var startIdx = 200000 + k * nIdf; // Смещение как в оригинале
            var endIdx = Math.Min(200000 + (k + 1) * nIdf, sentences.Length);

            if (sentences.Length <= startIdx)
            {
                k++;
                continue;
            }

            // Подсчитываем вхождения слов в документах
            for (int i = startIdx; i < endIdx; i++)
            {
                var uniqueWords = new HashSet<string>(sentences[i]);
                foreach (var word in uniqueWords)
                {
                    idf[language][word] = idf[language].GetValueOrDefault(word, 0) + 1;
                }
            }

            // Вычисляем IDF веса
            var nDoc = endIdx - startIdx;
            var wordsToUpdate = idf[language].Keys.ToList();
            foreach (var word in wordsToUpdate)
            {
                var docFreq = idf[language][word];
                idf[language][word] = Math.Max(1.0, Math.Log10((double)nDoc / docFreq));
            }

            k++;
        }

        return idf;
    }

    #endregion

    #region Vector Normalization

    /// <summary>
    /// Нормализует тензор векторов
    /// </summary>
    /// <param name="vectors">Тензор векторов [n, dim]</param>
    /// <param name="p">Норма (по умолчанию L2)</param>
    /// <param name="dim">Размерность для нормализации</param>
    /// <returns>Нормализованный тензор</returns>
    public static Tensor NormalizeVectors(Tensor vectors, double p = 2.0, int dim = 1)
    {
        return functional.normalize(vectors, p: p, dim: dim);
    }

    /// <summary>
    /// Нормализует массив векторов используя System.Numerics.Tensors для SIMD оптимизации
    /// </summary>
    /// <param name="vectors">Массив векторов</param>
    /// <returns>Нормализованные векторы</returns>
    public static float[][] NormalizeVectorArray(float[][] vectors)
    {
        var normalizedVectors = new float[vectors.Length][];

        for (int i = 0; i < vectors.Length; i++)
        {
            var vector = vectors[i];
            var norm = Math.Sqrt(System.Numerics.Tensors.TensorPrimitives.SumOfSquares(vector));

            if (norm > 0)
            {
                normalizedVectors[i] = new float[vector.Length];
                var normFactor = (float)(1.0 / norm);
                System.Numerics.Tensors.TensorPrimitives.Multiply(vector, normFactor, normalizedVectors[i]);
            }
            else
            {
                normalizedVectors[i] = vector.ToArray();
            }
        }

        return normalizedVectors;
    }

    #endregion

    #region Matrix Operations

    /// <summary>
    /// Эффективное матричное умножение с использованием TorchSharp для больших матриц
    /// </summary>
    /// <param name="matrix1">Первая матрица [n, k]</param>
    /// <param name="matrix2">Вторая матрица [k, m]</param>
    /// <param name="batchSize">Размер батча для экономии памяти</param>
    /// <returns>Результат умножения [n, m]</returns>
    public static async Task<Tensor> BatchedMatrixMultiplyAsync(
        Tensor matrix1, Tensor matrix2, int batchSize = DefaultBatchSize)
    {
        var n = (int)matrix1.size(0);
        var m = (int)matrix2.size(1);
        var result = zeros(n, m, dtype: matrix1.dtype, device: matrix1.device);

        for (int i = 0; i < n; i += batchSize)
        {
            var endIdx = Math.Min(n, i + batchSize);
            var batch = matrix1[TensorIndex.Slice(i, endIdx)];
            var batchResult = batch.mm(matrix2);
            
            result[TensorIndex.Slice(i, endIdx)] = batchResult;

            // Периодически передаем управление
            if (i % (batchSize * 5) == 0)
                await Task.Yield();
        }

        return result;
    }

    #endregion

    #region Precision@K Computation

    /// <summary>
    /// Вычисляет Precision@K метрики
    /// </summary>
    /// <param name="scores">Матрица скоров [n_queries, n_targets]</param>
    /// <param name="goldTargets">Истинные целевые индексы [n_queries]</param>
    /// <param name="kValues">Значения K для вычисления</param>
    /// <param name="allowMultiple">Разрешить множественные правильные переводы</param>
    /// <returns>Precision@K значения</returns>
    public static Dictionary<int, double> ComputePrecisionAtK(
        Tensor scores, Tensor goldTargets, int[] kValues, bool allowMultiple = true)
    {
        var results = new Dictionary<int, double>();
        var maxK = kValues.Max();
        var (_, topMatches) = scores.topk(maxK, dim: 1, largest: true, sorted: true);

        var topMatchesData = topMatches.data<long>().ToArray();
        var goldTargetsData = goldTargets.data<long>().ToArray();
        var nQueries = goldTargets.size(0);

        foreach (var k in kValues)
        {
            if (allowMultiple)
            {
                // Разрешаем множественные правильные переводы (группируем по исходному слову)
                var matchingBySource = new Dictionary<long, int>();

                for (int i = 0; i < nQueries; i++)
                {
                    var goldTarget = goldTargetsData[i];
                    var isMatch = false;

                    for (int j = 0; j < k; j++)
                    {
                        if (topMatchesData[i * maxK + j] == goldTarget)
                        {
                            isMatch = true;
                            break;
                        }
                    }

                    if (isMatch)
                    {
                        // В MUSE исходные слова группируются, здесь упрощаем
                        matchingBySource[i] = Math.Min(matchingBySource.GetValueOrDefault(i, 0) + 1, 1);
                    }
                }

                results[k] = 100.0 * matchingBySource.Values.Average();
            }
            else
            {
                // Простой подсчет без группировки
                int correctCount = 0;
                for (int i = 0; i < nQueries; i++)
                {
                    var goldTarget = goldTargetsData[i];
                    for (int j = 0; j < k; j++)
                    {
                        if (topMatchesData[i * maxK + j] == goldTarget)
                        {
                            correctCount++;
                            break;
                        }
                    }
                }

                results[k] = 100.0 * correctCount / (double)nQueries;
            }
        }

        return results;
    }

    #endregion

    #region Logging Utilities

    /// <summary>
    /// Логирует результаты оценки в табличном формате как в оригинальном MUSE
    /// </summary>
    /// <param name="logger">Логгер</param>
    /// <param name="title">Заголовок таблицы</param>
    /// <param name="results">Результаты оценки</param>
    /// <param name="columns">Названия колонок</param>
    public static void LogEvaluationTable(ILogger logger, string title, 
        IEnumerable<(string name, int found, int notFound, double score)> results,
        (string col1, string col2, string col3, string col4) columns = default)
    {
        var actualColumns = columns == default 
            ? (col1: "Dataset", col2: "Found", col3: "Not found", col4: "Score") 
            : columns;

        var separator = new string('=', 30 + 1 + 10 + 1 + 13 + 1 + 12);
        var pattern = "{0,-30} {1,10} {2,13} {3,12}";

        logger.LogInformation($"\n{title}");
        logger.LogInformation(separator);
        logger.LogInformation(string.Format(pattern, actualColumns.col1, actualColumns.col2, actualColumns.col3, actualColumns.col4));
        logger.LogInformation(separator);

        foreach (var (name, found, notFound, score) in results)
        {
            logger.LogInformation(string.Format(pattern, name, found, notFound, $"{score:F4}"));
        }

        logger.LogInformation(separator);
    }

    #endregion
}

/// <summary>
/// Расширения для упрощения работы с оценочными утилитами
/// </summary>
public static class EvaluationExtensions
{
    /// <summary>
    /// Конвертирует Tensor в массив векторов для совместимости с утилитами
    /// </summary>
    /// <param name="tensor">Тензор [n, dim]</param>
    /// <returns>Массив векторов</returns>
    public static float[][] ToVectorArray(this Tensor tensor)
    {
        var data = tensor.cpu().data<float>().ToArray();
        var rows = (int)tensor.size(0);
        var cols = (int)tensor.size(1);

        var vectors = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            vectors[i] = new float[cols];
            Array.Copy(data, i * cols, vectors[i], 0, cols);
        }

        return vectors;
    }

    /// <summary>
    /// Создает словарь векторов слов из эмбеддингов и словаря
    /// </summary>
    /// <param name="embeddings">Эмбеддинги [vocab_size, emb_dim]</param>
    /// <param name="dictionary">Словарь</param>
    /// <returns>Словарь векторов слов</returns>
    public static Dictionary<string, float[]> CreateWordVectorDictionary(this Tensor embeddings, Dictionary dictionary)
    {
        var vectors = embeddings.ToVectorArray();
        var wordVectors = new Dictionary<string, float[]>();

        foreach (var kvp in dictionary.WordToId)
        {
            wordVectors[kvp.Key] = vectors[kvp.Value];
        }

        return wordVectors;
    }

    /// <summary>
    /// Логирует прогресс длительной операции
    /// </summary>
    /// <param name="logger">Логгер</param>
    /// <param name="current">Текущий прогресс</param>
    /// <param name="total">Общее количество</param>
    /// <param name="operation">Название операции</param>
    /// <param name="stepSize">Шаг для логирования</param>
    public static void LogProgress(this ILogger logger, int current, int total, string operation, int stepSize = 1000)
    {
        if (current % stepSize == 0 || current == total)
        {
            var percentage = 100.0 * current / total;
            logger.LogInformation($"{operation}: {current}/{total} ({percentage:F1}%)");
        }
    }
}