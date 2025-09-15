using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Models;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Training;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Evaluation;
using Microsoft.Extensions.Logging;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Evaluation
{
    /// <summary>
    /// Результаты оценки точности перевода слов
    /// </summary>
    public sealed record WordTranslationResults
    {
        /// <summary>
        /// Precision@1
        /// </summary>
        public double Precision1 { get; init; }
        
        /// <summary>
        /// Precision@5  
        /// </summary>
        public double Precision5 { get; init; }
        
        /// <summary>
        /// Precision@10
        /// </summary>
        public double Precision10 { get; init; }
        
        /// <summary>
        /// Количество найденных пар в тестовом словаре
        /// </summary>
        public int FoundPairs { get; init; }
        
        /// <summary>
        /// Количество не найденных пар
        /// </summary>
        public int NotFoundPairs { get; init; }
    }

    /// <summary>
    /// Результаты оценки семантического сходства слов
    /// </summary>
    public sealed record WordSimilarityResults
    {
        /// <summary>
        /// Корреляция Спирмена
        /// </summary>
        public double SpearmanCorrelation { get; init; }
        
        /// <summary>
        /// Количество найденных пар
        /// </summary>
        public int FoundPairs { get; init; }
        
        /// <summary>
        /// Количество не найденных пар
        /// </summary>
        public int NotFoundPairs { get; init; }
        
        /// <summary>
        /// Название датасета
        /// </summary>
        public string DatasetName { get; init; } = "";
    }

    /// <summary>
    /// Результаты оценки mean cosine
    /// </summary>
    public sealed record MeanCosineResults
    {
        /// <summary>
        /// Значение mean cosine
        /// </summary>
        public double MeanCosine { get; init; }
        
        /// <summary>
        /// Метод построения словаря
        /// </summary>
        public string Method { get; init; } = "";
        
        /// <summary>
        /// Режим построения словаря
        /// </summary>
        public string BuildMode { get; init; } = "";
        
        /// <summary>
        /// Размер словаря
        /// </summary>
        public int DictionarySize { get; init; }
    }

    /// <summary>
    /// Комплексная система оценки качества кросс-лингвальных эмбеддингов
    /// Аналог evaluator.py с высокопроизводительными оптимизациями для .NET 9
    /// </summary>
    public sealed class CrossLingualEvaluator : IDisposable
    {
        #region Private Fields

        /// <summary>
        /// Тренер для доступа к моделям и данным
        /// </summary>
        private readonly CrossLingualTrainer _trainer;
        
        /// <summary>
        /// Логгер
        /// </summary>
        private readonly ILogger? _logger;
        
        /// <summary>
        /// Флаг освобождения ресурсов
        /// </summary>
        private bool _disposed = false;

        #endregion

        #region Constructor

        /// <summary>
        /// Инициализирует новый экземпляр оценщика
        /// </summary>
        /// <param name="trainer">Тренер с моделями и данными</param>
        /// <param name="logger">Логгер</param>
        public CrossLingualEvaluator(CrossLingualTrainer trainer, ILogger? logger = null)
        {
            _trainer = trainer ?? throw new ArgumentNullException(nameof(trainer));
            _logger = logger;
        }

        #endregion

        #region Public Methods - Word Translation

        /// <summary>
        /// Оценивает точность перевода слов с использованием билингвального словаря
        /// </summary>
        /// <param name="sourceLang">Код исходного языка</param>
        /// <param name="targetLang">Код целевого языка</param>
        /// <param name="dictionaryPath">Путь к тестовому словарю или "default"</param>
        /// <param name="method">Метод поиска переводов (nn, csls_knn_10)</param>
        /// <returns>Результаты оценки точности</returns>
        public async Task<WordTranslationResults> EvaluateWordTranslationAsync(
            string sourceLang,
            string targetLang, 
            string dictionaryPath = "default",
            string method = "csls_knn_10")
        {
            _logger?.LogInformation($"Оценка точности перевода слов методом {method}");
            
            // Загружаем тестовый словарь
            var testDictionary = await LoadEvaluationDictionaryAsync(sourceLang, targetLang, dictionaryPath);
            
            if (testDictionary.size(0) == 0)
            {
                _logger?.LogWarning("Тестовый словарь пустой");
                return new WordTranslationResults();
            }
            
            // Получаем нормализованные эмбеддинги
            var (sourceEmbeddings, targetEmbeddings) = GetNormalizedEmbeddings();
            
            // Вычисляем точность для k = 1, 5, 10
            var results = await ComputeTranslationAccuracyAsync(
                sourceEmbeddings, targetEmbeddings, testDictionary, method);
            
            _logger?.LogInformation($"Точность перевода - P@1: {results.Precision1:F3}, " +
                                  $"P@5: {results.Precision5:F3}, P@10: {results.Precision10:F3}");
            
            return results;
        }

        /// <summary>
        /// Загружает словарь для оценки
        /// </summary>
        private async Task<Tensor> LoadEvaluationDictionaryAsync(string sourceLang, string targetLang, string dictionaryPath)
        {
            string actualPath;
            
            if (dictionaryPath == "default")
            {
                actualPath = Path.Combine("data", "crosslingual", "dictionaries", $"{sourceLang}-{targetLang}.5000-6500.txt");
            }
            else
            {
                actualPath = dictionaryPath;
            }
            
            if (!File.Exists(actualPath))
            {
                _logger?.LogWarning($"Файл словаря не найден: {actualPath}");
                return zeros(0, 2, dtype: ScalarType.Int64);
            }
            
            var pairs = new List<(int sourceIdx, int targetIdx)>();
            int notFound = 0;
            
            await foreach (var line in File.ReadLinesAsync(actualPath))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                
                var parts = line.Trim().ToLowerInvariant().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 2) continue;
                
                var sourceWord = parts[0];
                var targetWord = parts[1];
                
                // Здесь нужно получить словари из тренера
                // Это требует модификации интерфейса тренера
                // Временно возвращаем пустой тензор
                notFound++;
            }
            
            _logger?.LogInformation($"Загружен тестовый словарь: {pairs.Count} пар, {notFound} не найдено");
            
            if (pairs.Count == 0)
                return zeros(0, 2, dtype: ScalarType.Int64);
            
            var data = new long[pairs.Count * 2];
            for (int i = 0; i < pairs.Count; i++)
            {
                data[i * 2] = pairs[i].sourceIdx;
                data[i * 2 + 1] = pairs[i].targetIdx;
            }
            
            return tensor(data, dtype: ScalarType.Int64).reshape(pairs.Count, 2);
        }

        /// <summary>
        /// Вычисляет точность перевода для различных k
        /// </summary>
        private async Task<WordTranslationResults> ComputeTranslationAccuracyAsync(
            Tensor sourceEmbeddings,
            Tensor targetEmbeddings,
            Tensor testDictionary, 
            string method)
        {
            var testSize = testDictionary.size(0);
            var sourceIndices = testDictionary.select(1, 0);
            var targetIndices = testDictionary.select(1, 1);
            
            // Получаем запросные эмбеддинги
            var queryEmbeddings = sourceEmbeddings.index_select(0, sourceIndices);
            
            // Вычисляем скоры в зависимости от метода
            Tensor scores;
            
            if (method == "nn")
            {
                // Простое скалярное произведение
                scores = queryEmbeddings.mm(targetEmbeddings.transpose(0, 1));
            }
            else if (method.StartsWith("csls_knn_"))
            {
                // CSLS метод
                var kStr = method.Substring("csls_knn_".Length);
                if (!int.TryParse(kStr, out int k))
                    throw new ArgumentException($"Некорректное значение k в методе: {method}");
                
                scores = await ComputeCSLSScoresAsync(queryEmbeddings, targetEmbeddings, k);
            }
            else
            {
                throw new ArgumentException($"Неподдерживаемый метод: {method}");
            }
            
            // Находим топ-10 кандидатов для каждого запроса
            var (_, topIndices) = scores.topk(10, dim: 1, largest: true, sorted: true);
            
            // Вычисляем точность для k = 1, 5, 10
            var precision1 = ComputePrecisionAtK(topIndices, targetIndices, 1);
            var precision5 = ComputePrecisionAtK(topIndices, targetIndices, 5);
            var precision10 = ComputePrecisionAtK(topIndices, targetIndices, 10);
            
            return new WordTranslationResults
            {
                Precision1 = precision1,
                Precision5 = precision5,
                Precision10 = precision10,
                FoundPairs = (int)testSize,
                NotFoundPairs = 0
            };
        }

        /// <summary>
        /// Вычисляет CSLS скоры
        /// </summary>
        private async Task<Tensor> ComputeCSLSScoresAsync(Tensor queryEmbeddings, Tensor targetEmbeddings, int k)
        {
            // Вычисляем средние расстояния до k ближайших соседей
            var avgDistQueries = await ComputeAverageDistancesToNeighborsAsync(queryEmbeddings, targetEmbeddings, k);
            var avgDistTargets = await ComputeAverageDistancesToNeighborsAsync(targetEmbeddings, queryEmbeddings, k);
            
            // Базовые скоры сходства
            var baseScores = queryEmbeddings.mm(targetEmbeddings.transpose(0, 1));
            
            // Применяем CSLS: 2 * similarity - avg_dist_queries - avg_dist_targets
            var cslsScores = baseScores.mul(2);
            cslsScores = cslsScores.sub(avgDistQueries.unsqueeze(1));
            cslsScores = cslsScores.sub(avgDistTargets.unsqueeze(0));
            
            return cslsScores;
        }

        /// <summary>
        /// Вычисляет средние расстояния до k ближайших соседей
        /// </summary>
        private Task<Tensor> ComputeAverageDistancesToNeighborsAsync(Tensor queries, Tensor keys, int k)
        {
            var queryCount = queries.size(0);
            var avgDistances = zeros(queryCount, dtype: ScalarType.Float32, device: queries.device);
            
            const int batchSize = 128;
            
            for (int i = 0; i < queryCount; i += batchSize)
            {
                var endIdx = Math.Min(queryCount, i + batchSize);
                var batchQueries = queries[TensorIndex.Slice(i, endIdx)];
                
                var similarities = keys.mm(batchQueries.transpose(0, 1)).transpose(0, 1);
                var topK = Math.Min(k, (int)keys.size(0));
                var (topSimilarities, _) = similarities.topk(topK, dim: 1, largest: true);
                
                avgDistances[TensorIndex.Slice(i, endIdx)] = topSimilarities.mean(dimensions: [ 1 ]); // VALFIX
            }
            
            return Task.FromResult(avgDistances);
        }

        /// <summary>
        /// Вычисляет Precision@K
        /// </summary>
        private double ComputePrecisionAtK(Tensor topIndices, Tensor targetIndices, int k)
        {
            var topK = topIndices[TensorIndex.Ellipsis, TensorIndex.Slice(null, k)];
            var targets = targetIndices.unsqueeze(1).expand_as(topK);
            
            var matches = topK.eq(targets).sum(dim: 1).cpu();
            var correctPredictions = matches.gt(0).sum().item<long>();
            
            return (double)correctPredictions / topIndices.size(0);
        }

        #endregion

        #region Public Methods - Mean Cosine

        /// <summary>
        /// Вычисляет mean cosine критерий для выбора модели
        /// </summary>
        /// <param name="methods">Методы построения словаря для тестирования</param>
        /// <param name="maxSize">Максимальный размер словаря</param>
        /// <returns>Результаты mean cosine для каждого метода</returns>
        public async Task<Dictionary<string, MeanCosineResults>> EvaluateMeanCosineAsync(
            string[]? methods = null,
            int maxSize = 10000)
        {
            methods ??= new[] { "nn", "csls_knn_10" };
            
            _logger?.LogInformation("Оценка mean cosine критерия");
            
            var results = new Dictionary<string, MeanCosineResults>();
            var (sourceEmbeddings, targetEmbeddings) = GetNormalizedEmbeddings();
            
            foreach (var method in methods)
            {
                try
                {
                    var parameters = new DictionaryBuilderParameters
                    {
                        Method = method,
                        BuildMode = "S2T",
                        MaxSize = maxSize,
                        MaxRank = 10000,
                        Threshold = 0.0,
                        MinSize = 0
                    };
                    
                    // Строим словарь
                    var dictionary = await DictionaryBuilder.BuildDictionaryAsync(
                        sourceEmbeddings, targetEmbeddings, parameters, logger: _logger);
                    
                    double meanCosine;
                    if (dictionary is null || dictionary.size(0) == 0)
                    {
                        meanCosine = -1e9;
                    }
                    else
                    {
                        // Вычисляем mean cosine
                        var dictSize = Math.Min(maxSize, (int)dictionary.size(0));
                        var sourceIndices = dictionary[TensorIndex.Slice(null, dictSize), 0];
                        var targetIndices = dictionary[TensorIndex.Slice(null, dictSize), 1];
                        
                        var sourceVecs = sourceEmbeddings.index_select(0, sourceIndices);
                        var targetVecs = targetEmbeddings.index_select(0, targetIndices);
                        
                        var cosines = (sourceVecs * targetVecs).sum(dim: 1);
                        meanCosine = cosines.mean().item<double>();
                    }
                    
                    results[method] = new MeanCosineResults
                    {
                        MeanCosine = meanCosine,
                        Method = method,
                        BuildMode = "S2T",
                        DictionarySize = (int)(dictionary?.size(0) ?? 0)
                    };
                    
                    _logger?.LogInformation($"Mean cosine ({method}): {meanCosine:F5}");
                }
                catch (Exception ex)
                {
                    _logger?.LogError($"Ошибка при вычислении mean cosine для {method}: {ex.Message}");
                    results[method] = new MeanCosineResults
                    {
                        MeanCosine = -1e9,
                        Method = method,
                        BuildMode = "S2T",
                        DictionarySize = 0
                    };
                }
            }
            
            return results;
        }

        #endregion

        #region Public Methods - Comprehensive Evaluation

        /// <summary>
        /// Выполняет полную оценку модели
        /// </summary>
        /// <param name="sourceLang">Исходный язык</param>
        /// <param name="targetLang">Целевой язык</param>
        /// <param name="evaluationMethods">Методы оценки</param>
        /// <returns>Словарь с результатами оценки</returns>
        public async Task<Dictionary<string, object>> RunFullEvaluationAsync(
            string sourceLang,
            string targetLang,
            string[]? evaluationMethods = null)
        {
            evaluationMethods ??= new[] { "nn", "csls_knn_10" };
            
            _logger?.LogSeparator("ПОЛНАЯ ОЦЕНКА МОДЕЛИ");
            
            var results = new Dictionary<string, object>();
            
            // Оценка точности перевода слов
            foreach (var method in evaluationMethods)
            {
                try
                {
                    var translationResults = await EvaluateWordTranslationAsync(sourceLang, targetLang, "default", method);
                    results[$"word_translation_{method}"] = translationResults;
                }
                catch (Exception ex)
                {
                    _logger?.LogError($"Ошибка в оценке перевода слов ({method}): {ex.Message}");
                }
            }
            
            // Оценка mean cosine
            try
            {
                var meanCosineResults = await EvaluateMeanCosineAsync(evaluationMethods);
                results["mean_cosine"] = meanCosineResults;
            }
            catch (Exception ex)
            {
                _logger?.LogError($"Ошибка в оценке mean cosine: {ex.Message}");
            }
            
            // TODO: Добавить другие виды оценки
            // - Монолингвальная оценка семантического сходства
            // - Кросс-лингвальная оценка семантического сходства  
            // - Оценка аналогий
            // - Оценка перевода предложений
            
            _logger?.LogSeparator("ОЦЕНКА ЗАВЕРШЕНА");
            
            return results;
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Получает нормализованные эмбеддинги из тренера
        /// </summary>
        private (Tensor sourceEmbeddings, Tensor targetEmbeddings) GetNormalizedEmbeddings()
        {
            // Получаем эмбеддинги через мапинг
            using var _ = no_grad();
            
            // Здесь нужен доступ к весам эмбеддингов через тренер
            // Пока используем заглушку - нужно будет добавить публичные свойства в тренер
            
            throw new NotImplementedException("Необходимо добавить доступ к эмбеддингам через тренер");
        }

        #endregion

        #region IDisposable

        /// <summary>
        /// Освобождает ресурсы
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Освобождает ресурсы
        /// </summary>
        private void Dispose(bool disposing)
        {
            if (!_disposed && disposing)
            {
                // Освобождение ресурсов если необходимо
                _disposed = true;
            }
        }

        #endregion
    }

    /// <summary>
    /// Расширения для логгера для оценки
    /// </summary>
    public static class EvaluationLoggerExtensions
    {
        /// <summary>
        /// Логирует результаты оценки в табличном формате
        /// </summary>
        public static void LogEvaluationResults(this ILogger logger, string title, Dictionary<string, object> results)
        {
            logger.LogSeparator(title);
            
            foreach (var result in results)
            {
                switch (result.Value)
                {
                    case WordTranslationResults wtr:
                        logger.LogInformation($"{result.Key}:");
                        logger.LogInformation($"  Precision@1:  {wtr.Precision1:F3}");
                        logger.LogInformation($"  Precision@5:  {wtr.Precision5:F3}");
                        logger.LogInformation($"  Precision@10: {wtr.Precision10:F3}");
                        logger.LogInformation($"  Found pairs:  {wtr.FoundPairs}");
                        break;
                        
                    case Dictionary<string, MeanCosineResults> mcResults:
                        logger.LogInformation($"{result.Key}:");
                        foreach (var mcr in mcResults)
                        {
                            logger.LogInformation($"  {mcr.Key}: {mcr.Value.MeanCosine:F5} (size: {mcr.Value.DictionarySize})");
                        }
                        break;
                        
                    default:
                        logger.LogInformation($"{result.Key}: {result.Value}");
                        break;
                }
            }
            
            logger.LogSeparator($"Конец: {title}");
        }
    }
}