using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Models;
using Microsoft.Extensions.Logging;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Evaluation
{
    /// <summary>
    /// Параметры для построения словарей
    /// </summary>
    public interface IDictionaryBuilderParameters
    {
        /// <summary>
        /// Метод построения словаря (nn, csls_knn_10, invsm_beta_30)
        /// </summary>
        string DicoMethod { get; }
        
        /// <summary>
        /// Режим построения (SourceToTarget, TargetToSource, SourceToTarget|TargetToSource, SourceToTarget&TargetToSource)
        /// </summary>
        string DicoBuild { get; }

        /// <summary>
        /// Пороговое значение уверенности
        /// </summary>
        float DicoThreshold { get; }
        
        /// <summary>
        /// Максимальный ранг слов в словаре
        /// </summary>
        int DicoMaxRank { get; }
        
        /// <summary>
        /// Минимальный размер словаря
        /// </summary>
        int DicoMinSize { get; }
        
        /// <summary>
        /// Максимальный размер словаря
        /// </summary>
        int DicoMaxSize { get; }
    }

    /// <summary>
    /// Строитель словарей для кросс-лингвального выравнивания
    /// Аналог dico_builder.py с оптимизациями для .NET 9
    /// </summary>
    public static class DictionaryBuilder
    {
        #region Constants
        
        /// <summary>
        /// Размер батча для обработки кандидатов
        /// </summary>
        private const int BatchSize = 128;
        
        #endregion

        #region Public Methods

        /// <summary>
        /// Получает лучших кандидатов для перевода
        /// </summary>
        /// <param name="sourceEmbeddings_Device">Исходные эмбеддинги [vocab_size, emb_dim]</param>
        /// <param name="targetEmbeddings_Device">Целевые эмбеддинги [vocab_size, emb_dim]</param>
        /// <param name="parameters">Параметры построения словаря</param>
        /// <param name="logger">Логгер</param>
        /// <returns>Пары кандидатов [n_pairs, 2] (source_idx, target_idx)</returns>
        public static async Task<Tensor> GetCandidatesAsync(
            Tensor sourceEmbeddings_Device, 
            Tensor targetEmbeddings_Device,
            IDictionaryBuilderParameters parameters,
            ILogger? logger = null)
        {
            logger?.LogInformation($"Получение кандидатов методом {parameters.DicoMethod}...");
            
            var sourceVocabSize = sourceEmbeddings_Device.size(0);
            var targetVocabSize = targetEmbeddings_Device.size(0);
            var embeddingDim = sourceEmbeddings_Device.size(1);
            
            // Ограничиваем количество исходных слов если задан max_rank
            var nSourceWords = parameters.DicoMaxRank > 0 && !parameters.DicoMethod.StartsWith("invsm_beta_") 
                ? Math.Min(parameters.DicoMaxRank, (int)sourceVocabSize) 
                : (int)sourceVocabSize;
            
            return parameters.DicoMethod switch
            {
                "nn" => await GetNearestNeighborCandidatesAsync(sourceEmbeddings_Device, targetEmbeddings_Device, nSourceWords, logger),
                var method when method.StartsWith("invsm_beta_") => await GetInvertedSoftmaxCandidatesAsync(
                    sourceEmbeddings_Device, targetEmbeddings_Device, method, parameters, logger),
                var method when method.StartsWith("csls_knn_") => await GetCSLSCandidatesAsync(
                    sourceEmbeddings_Device, targetEmbeddings_Device, method, nSourceWords, parameters, logger),
                _ => throw new ArgumentException($"Неизвестный метод построения словаря: {parameters.DicoMethod}")
            };
        }

        /// <summary>
        /// Строит словарь из выровненных эмбеддингов
        /// </summary>
        /// <param name="mappedSourceEmbeddings_Device">Исходные эмбеддинги</param>
        /// <param name="targetEmbeddings_Device">Целевые эмбеддинги</param>
        /// <param name="parameters">Параметры</param>
        /// <param name="sourceToTargetCandidates">Кандидаты Source->Target (опционально)</param>
        /// <param name="targetToSourceCandidates">Кандидаты Target->Source (опционально)</param>
        /// <param name="logger">Логгер</param>
        /// <returns>Словарь пар [n_pairs, 2] или null если словарь пустой</returns>
        public static async Task<Tensor?> BuildDictionaryAsync(
            Tensor mappedSourceEmbeddings_Device,
            Tensor targetEmbeddings_Device,
            IDictionaryBuilderParameters parameters,
            Tensor? sourceToTargetCandidates = null,
            Tensor? targetToSourceCandidates = null,
            ILogger? logger = null)
        {
            logger?.LogInformation("Построение обучающего словаря...");
            
            var buildSourceToTarget = parameters.DicoBuild.Contains("SourceToTarget");
            var buildTargetToSource = parameters.DicoBuild.Contains("TargetToSource");
            
            if (!buildSourceToTarget && !buildTargetToSource)
                throw new ArgumentException("Должен быть указан хотя бы один режим построения (SourceToTarget или TargetToSource)");
            
            // Получаем кандидатов SourceToTarget
            if (buildSourceToTarget && sourceToTargetCandidates is null)
            {
                sourceToTargetCandidates = await GetCandidatesAsync(mappedSourceEmbeddings_Device, targetEmbeddings_Device, parameters, logger);
            }
            
            // Получаем кандидатов TargetToSource
            if (buildTargetToSource && targetToSourceCandidates is null)
            {
                targetToSourceCandidates = await GetCandidatesAsync(targetEmbeddings_Device, mappedSourceEmbeddings_Device, parameters, logger);
                // Меняем местами колонки для TargetToSource
                targetToSourceCandidates = cat(new[] { targetToSourceCandidates.select(1, 1).unsqueeze(1), targetToSourceCandidates.select(1, 0).unsqueeze(1) }, dim: 1);
            }
            
            // Строим финальный словарь в зависимости от режима
            Tensor? finalDictionary = parameters.DicoBuild switch
            {
                "SourceToTarget" => sourceToTargetCandidates,
                "TargetToSource" => targetToSourceCandidates,
                "SourceToTarget|TargetToSource" => CombineDictionaries(sourceToTargetCandidates!, targetToSourceCandidates!, union: true, logger),
                "SourceToTarget&TargetToSource" => CombineDictionaries(sourceToTargetCandidates!, targetToSourceCandidates!, union: false, logger),
                _ => throw new ArgumentException($"Неизвестный режим построения: {parameters.DicoBuild}")
            };
            
            if (finalDictionary is null || finalDictionary.size(0) == 0)
            {
                logger?.LogWarning("Пустое пересечение кандидатов...");
                return null;
            }
            
            // Применяем фильтры
            finalDictionary = await ApplyFiltersAsync(finalDictionary, parameters, logger);
            
            logger?.LogInformation($"Новый обучающий словарь из {finalDictionary?.size(0) ?? 0} пар");
            
            return finalDictionary;
        }

        #endregion

        #region Private Methods - Candidate Generation

        /// <summary>
        /// Получает кандидатов методом ближайших соседей
        /// </summary>
        private static Task<Tensor> GetNearestNeighborCandidatesAsync(
            Tensor sourceEmbeddings_Device, 
            Tensor targetEmbeddings_Device, 
            int nSourceWords,
            ILogger? logger)
        {
            logger?.LogDebug("Использование метода ближайших соседей");
            
            var allScores = new List<Tensor>();
            var allTargets = new List<Tensor>();
            
            // Обрабатываем батчами для оптимизации памяти
            for (int i = 0; i < nSourceWords; i += BatchSize)
            {
                using var disposeScope = torch.NewDisposeScope();

                var endIdx = Math.Min(nSourceWords, i + BatchSize);
                var batchSourceEmb = sourceEmbeddings_Device[TensorIndex.Slice(i, endIdx)];
                
                // Вычисляем скоры: target_emb * source_emb^T
                var scores = targetEmbeddings_Device.mm(batchSourceEmb.transpose(0, 1)).transpose(0, 1);
                var (bestScores, bestTargets) = scores.topk(2, dim: 1, largest: true, sorted: true);
                
                allScores.Add(bestScores.cpu().DetachFromDisposeScope());
                allTargets.Add(bestTargets.cpu().DetachFromDisposeScope());
                
                if (i % (BatchSize * 10) == 0)
                {
                    logger?.LogDebug($"Обработано {i}/{nSourceWords} исходных слов");
                }
            }
            
            var finalScores = cat(allScores.ToArray(), dim: 0);
            var finalTargets = cat(allTargets.ToArray(), dim: 0);
            
            return Task.FromResult(CreateCandidatePairs(finalScores, finalTargets, nSourceWords));
        }

        /// <summary>
        /// Получает кандидатов методом инвертированного softmax
        /// </summary>
        private static Task<Tensor> GetInvertedSoftmaxCandidatesAsync(
            Tensor sourceEmbeddings,
            Tensor targetEmbeddings,
            string method,
            IDictionaryBuilderParameters parameters,
            ILogger? logger)
        {
            var betaStr = method.Substring("invsm_beta_".Length);
            if (!float.TryParse(betaStr, out float beta))
                throw new ArgumentException($"Некорректное значение beta в методе: {method}");
            
            logger?.LogDebug($"Использование инвертированного softmax с beta={beta}");
            
            var allScores = new List<Tensor>();
            var allTargets = new List<Tensor>();
            var targetVocabSize = targetEmbeddings.size(0);
            
            // Для каждого целевого слова
            for (int i = 0; i < targetVocabSize; i += BatchSize)
            {
                var endIdx = Math.Min(targetVocabSize, i + BatchSize);
                var batchTargetEmb = targetEmbeddings[TensorIndex.Slice(i, endIdx)];
                
                // Вычисляем скоры: source_emb * target_emb^T
                var scores = sourceEmbeddings.mm(batchTargetEmb.transpose(0, 1));
                
                // Применяем softmax: exp(beta * scores) / sum(exp(beta * scores))
                scores = scores.mul(beta).exp_();
                scores = scores.div(scores.sum(0, keepdim: true).expand_as(scores));
                
                var (bestScores, bestTargets) = scores.topk(2, dim: 0, largest: true, sorted: true);
                
                allScores.Add(bestScores.cpu());
                allTargets.Add((bestTargets + i).cpu());
            }
            
            var finalScores = cat(allScores.ToArray(), dim: 1);
            var finalTargets = cat(allTargets.ToArray(), dim: 1);
            
            // Получаем лучшие пары для каждого исходного слова
            var (topScores, topIndices) = finalScores.topk(2, dim: 1, largest: true, sorted: true);
            var topTargets = finalTargets.gather(1, topIndices);
            
            return Task.FromResult(CreateCandidatePairs(topScores, topTargets, sourceEmbeddings.size(0)));
        }

        /// <summary>
        /// Получает кандидатов методом CSLS (Cross-domain Similarity Local Scaling)
        /// </summary>
        private static async Task<Tensor> GetCSLSCandidatesAsync(
            Tensor mappedSourceEmbeddings,
            Tensor targetEmbeddings,
            string method,
            int nSourceWords,
            IDictionaryBuilderParameters parameters,
            ILogger? logger)
        {
            var knnStr = method.Substring("csls_knn_".Length);
            if (!int.TryParse(knnStr, out int k))
                throw new ArgumentException($"Некорректное значение k в методе: {method}");
            
            logger?.LogDebug($"Использование CSLS с k={k}");
            
            // Вычисляем средние расстояния до k ближайших соседей
            logger?.LogDebug("Вычисление средних расстояний для CSLS...");
            var mappedSourceEmb_AvgDist = await EvaluationUtils.ComputeAverageDistancesAsync(emb: targetEmbeddings, query: mappedSourceEmbeddings, k);
            var targetEmb_AvgDist = await EvaluationUtils.ComputeAverageDistancesAsync(emb: mappedSourceEmbeddings, query: targetEmbeddings, k);
            
            var allScores = new List<Tensor>(nSourceWords);
            var allTargets = new List<Tensor>(nSourceWords);

            // Для каждого исходного слова вычисляем CSLS скоры            
            for (int i = 0; i < nSourceWords; i += BatchSize)
            {
                var endIdx = Math.Min(nSourceWords, i + BatchSize);
                var batchSourceEmb = mappedSourceEmbeddings[TensorIndex.Slice(i, endIdx)];
                
                // Базовые скоры сходства
                var scores = targetEmbeddings.mm(batchSourceEmb.transpose(0, 1)).transpose(0, 1);
                
                // Применяем CSLS: 2 * similarity - avg_dist1 - avg_dist2
                scores = scores.mul(2);
                scores = scores.sub(mappedSourceEmb_AvgDist[TensorIndex.Slice(i, endIdx)].unsqueeze(dim: 1));
                scores = scores.sub(targetEmb_AvgDist.unsqueeze(dim: 0));
                
                var (bestScores, bestTargets) = scores.topk(k: 2, dim: 1, largest: true, sorted: true);
                
                allScores.Add(bestScores.cpu());
                allTargets.Add(bestTargets.cpu());
            }
            
            var finalScores = cat(allScores.ToArray(), dim: 0);
            var finalTargets = cat(allTargets.ToArray(), dim: 0);
            
            return CreateCandidatePairs(finalScores, finalTargets, nSourceWords);
        }        

        #endregion

        #region Private Methods - Utilities

        /// <summary>
        /// Создает пары кандидатов из скоров и целевых индексов
        /// </summary>
        private static Tensor CreateCandidatePairs(Tensor scores, Tensor targets, long nSourceWords)
        {
            // Создаем пары (source_idx, target_idx) с лучшими скорами
            var sourceIndices = arange(start: 0, stop: nSourceWords, dtype: ScalarType.Int64, device: targets.device);
            var bestTargets = targets.select(dim: 1, index: 0); // Берем лучшие целевые индексы
            
            var pairs = cat(new[]
            {
                sourceIndices.unsqueeze(1),
                bestTargets.unsqueeze(1)
            }, dim: 1);
            
            // Сортируем по уверенности (разность между лучшим и вторым скором)
            var confidence = scores.select(dim: 1, index: 0) - scores.select(dim: 1, index: 1);
            var (_, sortedIndices) = confidence.sort(dim: 0, descending: true);
            
            return pairs[sortedIndices];
        }

        /// <summary>
        /// Объединяет два словаря (пересечение или объединение)
        /// </summary>
        private static Tensor? CombineDictionaries(Tensor sourceToTargetCandidates, Tensor targetToSourceCandidates, bool union, ILogger? logger)
        {
            // Конвертируем в HashSet для эффективного поиска
            var sourceToTargetPairs = new HashSet<(long, long)>();
            var sourceToTargetData = sourceToTargetCandidates.data<long>().ToArray();
            for (int i = 0; i < sourceToTargetData.Length; i += 2)
            {
                sourceToTargetPairs.Add((sourceToTargetData[i], sourceToTargetData[i + 1]));
            }
            
            var targetToSourcePairs = new HashSet<(long, long)>();
            var targetToSourceData = targetToSourceCandidates.data<long>().ToArray();
            for (int i = 0; i < targetToSourceData.Length; i += 2)
            {
                targetToSourcePairs.Add((targetToSourceData[i], targetToSourceData[i + 1]));
            }
            
            // Выполняем операцию объединения или пересечения
            var finalPairs = union ? sourceToTargetPairs.Union(targetToSourcePairs).ToList() : sourceToTargetPairs.Intersect(targetToSourcePairs).ToList();
            
            if (finalPairs.Count == 0)
            {
                logger?.LogWarning("Пустое множество пар после объединения/пересечения");
                return null;
            }
            
            // Создаем тензор результата
            var resultData = new long[finalPairs.Count * 2];
            for (int i = 0; i < finalPairs.Count; i++)
            {
                resultData[i * 2] = finalPairs[i].Item1;
                resultData[i * 2 + 1] = finalPairs[i].Item2;
            }
            
            return tensor(resultData, dtype: ScalarType.Int64).reshape(finalPairs.Count, 2);
        }

        /// <summary>
        /// Применяет фильтры к словарю (размер, ранг, порог уверенности)
        /// </summary>
        private static Task<Tensor> ApplyFiltersAsync(Tensor dictionary, IDictionaryBuilderParameters parameters, ILogger? logger)
        {
            var currentDict = dictionary;
            
            // Фильтр по максимальному рангу
            if (parameters.DicoMaxRank > 0)
            {
                var maxRankMask = currentDict.max(dim: 1).values <= parameters.DicoMaxRank;
                if (maxRankMask.sum().item<long>() < currentDict.size(0))
                {
                    currentDict = currentDict[maxRankMask];
                    logger?.LogDebug($"После фильтра по max_rank: {currentDict.size(0)} пар");
                }
            }
            
            // Фильтр по максимальному размеру
            if (parameters.DicoMaxSize > 0 && currentDict.size(0) > parameters.DicoMaxSize)
            {
                currentDict = currentDict[TensorIndex.Slice(null, parameters.DicoMaxSize)];
                logger?.LogDebug($"После фильтра по max_size: {currentDict.size(0)} пар");
            }
            
            // Примечание: фильтр по минимальному размеру и порогу уверенности 
            // требует дополнительной информации о скорах, которая не сохраняется в текущей реализации
            // Эти фильтры можно добавить, если передавать скоры отдельно
            
            return Task.FromResult(currentDict);
        }

        #endregion
    }
}



///// <summary>
///// Вычисляет средние расстояния до k ближайших соседей для CSLS
///// </summary>
//private static Task<Tensor> ComputeAverageDistancesAsync(
//    Tensor queryEmbeddings,
//    Tensor keyEmbeddings,
//    int k,
//    ILogger? logger)
//{
//    var queryCount = queryEmbeddings.size(0);
//    var keyCount = keyEmbeddings.size(0);
//    var avgDistances = zeros(queryCount, dtype: ScalarType.Float32, device: queryEmbeddings.device);

//    // Обрабатываем батчами
//    for (int i = 0; i < queryCount; i += BatchSize)
//    {
//        var endIdx = Math.Min(queryCount, i + BatchSize);
//        var batchQueries = queryEmbeddings[TensorIndex.Slice(i, endIdx)];

//        // Вычисляем косинусные сходства
//        var similarities = keyEmbeddings.mm(batchQueries.transpose(0, 1)).transpose(0, 1);

//        // Находим k лучших сходств для каждого запроса
//        var topK = Math.Min(k, (int)keyCount);
//        var (topSimilarities, _) = similarities.topk(topK, dim: 1, largest: true, sorted: true);

//        // Вычисляем среднее
//        avgDistances[TensorIndex.Slice(i, endIdx)] = topSimilarities.mean(dimensions: [1]); // VALFIX

//        if (i % (BatchSize * 5) == 0)
//        {
//            logger?.LogDebug($"CSLS: обработано {i}/{queryCount} запросов");
//        }
//    }

//    return Task.FromResult(avgDistances);
//}