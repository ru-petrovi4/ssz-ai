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
    public sealed record DictionaryBuilderParameters
    {
        /// <summary>
        /// Метод построения словаря (nn, csls_knn_10, invsm_beta_30)
        /// </summary>
        public string Method { get; init; } = "csls_knn_10";
        
        /// <summary>
        /// Режим построения (S2T, T2S, S2T|T2S, S2T&T2S)
        /// </summary>
        public string BuildMode { get; init; } = "S2T";
        
        /// <summary>
        /// Пороговое значение уверенности
        /// </summary>
        public double Threshold { get; init; } = 0.0;
        
        /// <summary>
        /// Максимальный ранг слов в словаре
        /// </summary>
        public int MaxRank { get; init; } = 15000;
        
        /// <summary>
        /// Минимальный размер словаря
        /// </summary>
        public int MinSize { get; init; } = 0;
        
        /// <summary>
        /// Максимальный размер словаря
        /// </summary>
        public int MaxSize { get; init; } = 0;
        
        /// <summary>
        /// Использовать GPU для вычислений
        /// </summary>
        public bool UseCuda { get; init; } = true;
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
        /// <param name="sourceEmbeddings">Исходные эмбеддинги [vocab_size, emb_dim]</param>
        /// <param name="targetEmbeddings">Целевые эмбеддинги [vocab_size, emb_dim]</param>
        /// <param name="parameters">Параметры построения словаря</param>
        /// <param name="logger">Логгер</param>
        /// <returns>Пары кандидатов [n_pairs, 2] (source_idx, target_idx)</returns>
        public static async Task<Tensor> GetCandidatesAsync(
            Tensor sourceEmbeddings, 
            Tensor targetEmbeddings, 
            DictionaryBuilderParameters parameters,
            ILogger? logger = null)
        {
            logger?.LogInformation($"Получение кандидатов методом {parameters.Method}...");
            
            var sourceVocabSize = sourceEmbeddings.size(0);
            var targetVocabSize = targetEmbeddings.size(0);
            var embeddingDim = sourceEmbeddings.size(1);
            
            // Ограничиваем количество исходных слов если задан max_rank
            var nSourceWords = parameters.MaxRank > 0 && !parameters.Method.StartsWith("invsm_beta_") 
                ? Math.Min(parameters.MaxRank, (int)sourceVocabSize) 
                : (int)sourceVocabSize;
            
            return parameters.Method switch
            {
                "nn" => await GetNearestNeighborCandidatesAsync(sourceEmbeddings, targetEmbeddings, nSourceWords, logger),
                var method when method.StartsWith("invsm_beta_") => await GetInvertedSoftmaxCandidatesAsync(
                    sourceEmbeddings, targetEmbeddings, method, parameters, logger),
                var method when method.StartsWith("csls_knn_") => await GetCSLSCandidatesAsync(
                    sourceEmbeddings, targetEmbeddings, method, nSourceWords, parameters, logger),
                _ => throw new ArgumentException($"Неизвестный метод построения словаря: {parameters.Method}")
            };
        }

        /// <summary>
        /// Строит словарь из выровненных эмбеддингов
        /// </summary>
        /// <param name="sourceEmbeddings">Исходные эмбеддинги</param>
        /// <param name="targetEmbeddings">Целевые эмбеддинги</param>
        /// <param name="parameters">Параметры</param>
        /// <param name="s2tCandidates">Кандидаты Source->Target (опционально)</param>
        /// <param name="t2sCandidates">Кандидаты Target->Source (опционально)</param>
        /// <param name="logger">Логгер</param>
        /// <returns>Словарь пар [n_pairs, 2] или null если словарь пустой</returns>
        public static async Task<Tensor?> BuildDictionaryAsync(
            Tensor sourceEmbeddings,
            Tensor targetEmbeddings,
            DictionaryBuilderParameters parameters,
            Tensor? s2tCandidates = null,
            Tensor? t2sCandidates = null,
            ILogger? logger = null)
        {
            logger?.LogInformation("Построение обучающего словаря...");
            
            var buildS2T = parameters.BuildMode.Contains("S2T");
            var buildT2S = parameters.BuildMode.Contains("T2S");
            
            if (!buildS2T && !buildT2S)
                throw new ArgumentException("Должен быть указан хотя бы один режим построения (S2T или T2S)");
            
            // Получаем кандидатов S2T
            if (buildS2T && s2tCandidates is null)
            {
                s2tCandidates = await GetCandidatesAsync(sourceEmbeddings, targetEmbeddings, parameters, logger);
            }
            
            // Получаем кандидатов T2S
            if (buildT2S && t2sCandidates is null)
            {
                t2sCandidates = await GetCandidatesAsync(targetEmbeddings, sourceEmbeddings, parameters, logger);
                // Меняем местами колонки для T2S
                t2sCandidates = cat(new[] { t2sCandidates.select(1, 1).unsqueeze(1), t2sCandidates.select(1, 0).unsqueeze(1) }, dim: 1);
            }
            
            // Строим финальный словарь в зависимости от режима
            Tensor? finalDictionary = parameters.BuildMode switch
            {
                "S2T" => s2tCandidates,
                "T2S" => t2sCandidates,
                "S2T|T2S" => CombineDictionaries(s2tCandidates!, t2sCandidates!, union: true, logger),
                "S2T&T2S" => CombineDictionaries(s2tCandidates!, t2sCandidates!, union: false, logger),
                _ => throw new ArgumentException($"Неизвестный режим построения: {parameters.BuildMode}")
            };
            
            if (finalDictionary is null || finalDictionary.size(0) == 0)
            {
                logger?.LogWarning("Пустое пересечение кандидатов...");
                return null;
            }
            
            // Применяем фильтры
            finalDictionary = await ApplyFiltersAsync(finalDictionary, parameters, logger);
            
            logger?.LogInformation($"Новый обучающий словарь из {finalDictionary?.size(0) ?? 0} пар");
            
            return finalDictionary?.to(parameters.UseCuda ? CUDA : CPU);
        }

        #endregion

        #region Private Methods - Candidate Generation

        /// <summary>
        /// Получает кандидатов методом ближайших соседей
        /// </summary>
        private static async Task<Tensor> GetNearestNeighborCandidatesAsync(
            Tensor sourceEmbeddings, 
            Tensor targetEmbeddings, 
            int nSourceWords,
            ILogger? logger)
        {
            logger?.LogDebug("Использование метода ближайших соседей");
            
            var allScores = new List<Tensor>();
            var allTargets = new List<Tensor>();
            
            // Обрабатываем батчами для оптимизации памяти
            for (int i = 0; i < nSourceWords; i += BatchSize)
            {
                var endIdx = Math.Min(nSourceWords, i + BatchSize);
                var batchSourceEmb = sourceEmbeddings[TensorIndex.Slice(i, endIdx)];
                
                // Вычисляем скоры: target_emb * source_emb^T
                var scores = targetEmbeddings.mm(batchSourceEmb.transpose(0, 1)).transpose(0, 1);
                var (bestScores, bestTargets) = scores.topk(2, dim: 1, largest: true, sorted: true);
                
                allScores.Add(bestScores.cpu());
                allTargets.Add(bestTargets.cpu());
                
                if (i % (BatchSize * 10) == 0)
                {
                    logger?.LogDebug($"Обработано {i}/{nSourceWords} исходных слов");
                }
            }
            
            var finalScores = cat(allScores.ToArray(), dim: 0);
            var finalTargets = cat(allTargets.ToArray(), dim: 0);
            
            return CreateCandidatePairs(finalScores, finalTargets, nSourceWords);
        }

        /// <summary>
        /// Получает кандидатов методом инвертированного softmax
        /// </summary>
        private static async Task<Tensor> GetInvertedSoftmaxCandidatesAsync(
            Tensor sourceEmbeddings,
            Tensor targetEmbeddings,
            string method,
            DictionaryBuilderParameters parameters,
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
            
            return CreateCandidatePairs(topScores, topTargets, sourceEmbeddings.size(0));
        }

        /// <summary>
        /// Получает кандидатов методом CSLS (Cross-domain Similarity Local Scaling)
        /// </summary>
        private static async Task<Tensor> GetCSLSCandidatesAsync(
            Tensor sourceEmbeddings,
            Tensor targetEmbeddings,
            string method,
            int nSourceWords,
            DictionaryBuilderParameters parameters,
            ILogger? logger)
        {
            var knnStr = method.Substring("csls_knn_".Length);
            if (!int.TryParse(knnStr, out int k))
                throw new ArgumentException($"Некорректное значение k в методе: {method}");
            
            logger?.LogDebug($"Использование CSLS с k={k}");
            
            // Вычисляем средние расстояния до k ближайших соседей
            logger?.LogDebug("Вычисление средних расстояний для CSLS...");
            var avgDist1 = await ComputeAverageDistancesAsync(targetEmbeddings, sourceEmbeddings, k, logger);
            var avgDist2 = await ComputeAverageDistancesAsync(sourceEmbeddings, targetEmbeddings, k, logger);
            
            var allScores = new List<Tensor>();
            var allTargets = new List<Tensor>();
            
            // Для каждого исходного слова вычисляем CSLS скоры
            for (int i = 0; i < nSourceWords; i += BatchSize)
            {
                var endIdx = Math.Min(nSourceWords, i + BatchSize);
                var batchSourceEmb = sourceEmbeddings[TensorIndex.Slice(i, endIdx)];
                
                // Базовые скоры сходства
                var scores = targetEmbeddings.mm(batchSourceEmb.transpose(0, 1)).transpose(0, 1);
                
                // Применяем CSLS: 2 * similarity - avg_dist1 - avg_dist2
                scores = scores.mul(2);
                scores = scores.sub(avgDist1[TensorIndex.Slice(i, endIdx)].unsqueeze(1));
                scores = scores.sub(avgDist2.unsqueeze(0));
                
                var (bestScores, bestTargets) = scores.topk(2, dim: 1, largest: true, sorted: true);
                
                allScores.Add(bestScores.cpu());
                allTargets.Add(bestTargets.cpu());
            }
            
            var finalScores = cat(allScores.ToArray(), dim: 0);
            var finalTargets = cat(allTargets.ToArray(), dim: 0);
            
            return CreateCandidatePairs(finalScores, finalTargets, nSourceWords);
        }

        /// <summary>
        /// Вычисляет средние расстояния до k ближайших соседей для CSLS
        /// </summary>
        private static async Task<Tensor> ComputeAverageDistancesAsync(
            Tensor queryEmbeddings,
            Tensor keyEmbeddings, 
            int k,
            ILogger? logger)
        {
            var queryCount = queryEmbeddings.size(0);
            var keyCount = keyEmbeddings.size(0);
            var avgDistances = zeros(queryCount, dtype: ScalarType.Float32, device: queryEmbeddings.device);
            
            // Обрабатываем батчами
            for (int i = 0; i < queryCount; i += BatchSize)
            {
                var endIdx = Math.Min(queryCount, i + BatchSize);
                var batchQueries = queryEmbeddings[TensorIndex.Slice(i, endIdx)];
                
                // Вычисляем косинусные сходства
                var similarities = keyEmbeddings.mm(batchQueries.transpose(0, 1)).transpose(0, 1);
                
                // Находим k лучших сходств для каждого запроса
                var topK = Math.Min(k, (int)keyCount);
                var (topSimilarities, _) = similarities.topk(topK, dim: 1, largest: true, sorted: true);
                
                // Вычисляем среднее
                avgDistances[TensorIndex.Slice(i, endIdx)] = topSimilarities.mean(dimensions: [ 1 ]); // VALFIX
                
                if (i % (BatchSize * 5) == 0)
                {
                    logger?.LogDebug($"CSLS: обработано {i}/{queryCount} запросов");
                }
            }
            
            return avgDistances;
        }

        #endregion

        #region Private Methods - Utilities

        /// <summary>
        /// Создает пары кандидатов из скоров и целевых индексов
        /// </summary>
        private static Tensor CreateCandidatePairs(Tensor scores, Tensor targets, long nSourceWords)
        {
            // Создаем пары (source_idx, target_idx) с лучшими скорами
            var sourceIndices = arange(0, nSourceWords, dtype: ScalarType.Int64, device: targets.device);
            var bestTargets = targets.select(1, 0); // Берем лучшие целевые индексы
            
            var pairs = cat(new[]
            {
                sourceIndices.unsqueeze(1),
                bestTargets.unsqueeze(1)
            }, dim: 1);
            
            // Сортируем по уверенности (разность между лучшим и вторым скором)
            var confidence = scores.select(1, 0) - scores.select(1, 1);
            var (_, sortedIndices) = confidence.sort(0, descending: true);
            
            return pairs[sortedIndices];
        }

        /// <summary>
        /// Объединяет два словаря (пересечение или объединение)
        /// </summary>
        private static Tensor? CombineDictionaries(Tensor s2tCandidates, Tensor t2sCandidates, bool union, ILogger? logger)
        {
            // Конвертируем в HashSet для эффективного поиска
            var s2tPairs = new HashSet<(long, long)>();
            var s2tData = s2tCandidates.data<long>().ToArray();
            for (int i = 0; i < s2tData.Length; i += 2)
            {
                s2tPairs.Add((s2tData[i], s2tData[i + 1]));
            }
            
            var t2sPairs = new HashSet<(long, long)>();
            var t2sData = t2sCandidates.data<long>().ToArray();
            for (int i = 0; i < t2sData.Length; i += 2)
            {
                t2sPairs.Add((t2sData[i], t2sData[i + 1]));
            }
            
            // Выполняем операцию объединения или пересечения
            var finalPairs = union ? s2tPairs.Union(t2sPairs).ToList() : s2tPairs.Intersect(t2sPairs).ToList();
            
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
        private static async Task<Tensor> ApplyFiltersAsync(Tensor dictionary, DictionaryBuilderParameters parameters, ILogger? logger)
        {
            var currentDict = dictionary;
            
            // Фильтр по максимальному рангу
            if (parameters.MaxRank > 0)
            {
                var maxRankMask = currentDict.max(dim: 1).values <= parameters.MaxRank;
                if (maxRankMask.sum().item<long>() < currentDict.size(0))
                {
                    currentDict = currentDict[maxRankMask];
                    logger?.LogDebug($"После фильтра по max_rank: {currentDict.size(0)} пар");
                }
            }
            
            // Фильтр по максимальному размеру
            if (parameters.MaxSize > 0 && currentDict.size(0) > parameters.MaxSize)
            {
                currentDict = currentDict[TensorIndex.Slice(null, parameters.MaxSize)];
                logger?.LogDebug($"После фильтра по max_size: {currentDict.size(0)} пар");
            }
            
            // Примечание: фильтр по минимальному размеру и порогу уверенности 
            // требует дополнительной информации о скорах, которая не сохраняется в текущей реализации
            // Эти фильтры можно добавить, если передавать скоры отдельно
            
            return currentDict;
        }

        #endregion
    }
}