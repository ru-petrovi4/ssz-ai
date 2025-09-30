using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
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
using Tensor = TorchSharp.torch.Tensor;
using static TorchSharp.torch.nn;
using Tensorflow;
using Ssz.Utils;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Evaluation
{
    /// <summary>
    /// Полная реализация класса Evaluator из оригинального проекта MUSE на Python
    /// Содержит все оценочные методы: word similarity, word analogy, word translation, 
    /// sentence translation, discriminator evaluation и mean cosine
    /// Оптимизировано для .NET 9 с высокопроизводительными алгоритмами
    /// </summary>
    public sealed class Evaluator : IDisposable
    {
        #region construction and destruction

        /// <summary>
        /// Инициализирует новый экземпляр MUSE Evaluator
        /// </summary>
        /// <param name="trainer">Тренер с моделями и данными</param>
        /// <param name="logger">Логгер</param>
        public Evaluator(Trainer trainer, ILogger? logger = null)
        {
            _trainer = trainer ?? throw new ArgumentNullException(nameof(trainer));
            _logger = logger;
        }

        #endregion

        #region Constants

        /// <summary>
        /// Путь к монолингвальным датасетам оценки
        /// </summary>
        private const string MonolingualEvalPath = "data/monolingual";

        /// <summary>
        /// Путь к кросс-лингвальным датасетам SEMEVAL
        /// </summary>
        private const string SemEval17EvalPath = "data/crosslingual/wordsim";

        /// <summary>
        /// Путь к Europarl корпусу для оценки перевода предложений
        /// </summary>
        private const string EuroparlDir = "data/crosslingual/europarl";

        /// <summary>
        /// Путь к словарям для оценки
        /// </summary>
        private const string DictionariesPath = "data/crosslingual/dictionaries";

        /// <summary>
        /// Размер батча для обработки
        /// </summary>
        private const int BatchSize = 128;

        #endregion

        #region Private Fields

        /// <summary>
        /// Тренер с доступом к моделям и данным
        /// </summary>
        private readonly Trainer _trainer;

        /// <summary>
        /// Логгер
        /// </summary>
        private readonly ILogger? _logger;

        /// <summary>
        /// Кэш для Europarl данных
        /// </summary>
        private Dictionary<string, Dictionary<string, string[][]>>? _europarlData;

        /// <summary>
        /// Флаг освобождения ресурсов
        /// </summary>
        private bool _disposed = false;

        #endregion        

        #region Public Methods - Monolingual Word Similarity

        /// <summary>
        /// Оценка монолингвального семантического сходства слов
        /// Реализация метода monolingual_wordsim из оригинального evaluator.py
        /// Использует файлы данных.
        /// </summary>
        /// <param name="results">Словарь для записи результатов</param>
        public async Task EvaluateMonolingualWordSimilarityAsync(TrainingStats stats)
        {
            _logger?.LogInformation("Оценка монолингвального семантического сходства...");

            // Получаем выровненные эмбеддинги исходного языка
            var sourceEmbeddings = GetMappedSourceEmbeddings();            

            var sourceScores = await GetWordSimilarityScoresAsync(
                _trainer.SourceDictionary.Language, 
                _trainer.SourceDictionary, 
                sourceEmbeddings);

            // Оцениваем целевой язык если доступен
            Dictionary<string, float>? targetScores = null;
            if (!string.IsNullOrEmpty(_trainer.TargetDictionary?.Language))
            {
                var targetEmbeddings = _trainer.TargetEmbeddings.weight!.cpu();
                targetScores = await GetWordSimilarityScoresAsync(
                    _trainer.TargetDictionary.Language, 
                    _trainer.TargetDictionary, 
                    targetEmbeddings);
            }

            // Логируем и записываем результаты для исходного языка
            if (sourceScores != null && sourceScores.Count > 0)
            {
                var avgScore = sourceScores.Values.Average();
                _logger?.LogInformation($"Средний балл семантического сходства (исходный язык): {avgScore:F5}");

                stats.ToLog["src_ws_monolingual_scores"] = avgScore;
                foreach (var kvp in sourceScores)
                {
                    stats.ToLog[$"src_{kvp.Key}"] = kvp.Value;
                }
            }

            // Логируем и записываем результаты для целевого языка
            if (targetScores != null && targetScores.Count > 0)
            {
                var avgScore = targetScores.Values.Average();
                _logger?.LogInformation($"Средний балл семантического сходства (целевой язык): {avgScore:F5}");

                stats.ToLog["tgt_ws_monolingual_scores"] = avgScore;
                foreach (var kvp in targetScores)
                {
                    stats.ToLog[$"tgt_{kvp.Key}"] = kvp.Value;
                }
            }

            // Общий балл если есть оба языка
            if (sourceScores?.Count > 0 && targetScores?.Count > 0)
            {
                var overallScore = (sourceScores.Values.Average() + targetScores.Values.Average()) / 2;
                _logger?.LogInformation($"Общий средний балл семантического сходства: {overallScore:F5}");
                stats.ToLog["ws_monolingual_scores"] = overallScore;
            }
        }

        /// <summary>
        /// Получает баллы семантического сходства для языка
        /// </summary>
        private async Task<Dictionary<string, float>?> GetWordSimilarityScoresAsync(
            string language, Dictionary dictionary, Tensor embeddings)
        {
            var datasetDir = Path.Combine(MonolingualEvalPath, language);
            if (!Directory.Exists(datasetDir))
            {
                _logger?.LogWarning($"Датасеты для языка {language} не найдены в {datasetDir}");
                return null;
            }

            var scores = new Dictionary<string, float>();
            var separator = new string('=', 30 + 1 + 10 + 1 + 13 + 1 + 12);
            var pattern = "{0,-30} {1,10} {2,13} {3,12}";

            _logger?.LogInformation(separator);
            _logger?.LogInformation(string.Format(pattern, "Dataset", "Found", "Not found", "Rho"));
            _logger?.LogInformation(separator);

            var files = Directory.GetFiles(datasetDir, $"{language.ToUpper()}_*.txt");
            foreach (var filePath in files)
            {
                try
                {
                    var fileName = Path.GetFileNameWithoutExtension(filePath);
                    var (correlation, found, notFound) = await ComputeSpearmanRhoAsync(
                        dictionary, embeddings, filePath);

                    _logger?.LogInformation(string.Format(pattern, fileName, found, notFound, $"{correlation:F4}"));
                    scores[fileName] = correlation;
                }
                catch (Exception ex)
                {
                    _logger?.LogWarning($"Ошибка при обработке {filePath}: {ex.Message}");
                }
            }

            _logger?.LogInformation(separator);
            return scores.Count > 0 ? scores : null;
        }

        #endregion

        #region Public Methods - Monolingual Word Analogy

        /// <summary>
        /// Оценка монолингвальных аналогий слов (только для английского языка)
        /// Реализация метода monolingual_wordanalogy из оригинального evaluator.py
        /// </summary>
        /// <param name="results">Словарь для записи результатов</param>
        public async Task EvaluateMonolingualWordAnalogyAsync(TrainingStats stats)
        {
            _logger?.LogInformation("Оценка монолингвальных аналогий слов...");

            // Оцениваем исходный язык
            var sourceLanguage = _trainer.SourceDictionary.Language;
            var sourceEmbeddings = GetMappedSourceEmbeddings();
            var sourceAnalogies = await GetWordAnalogyScoresAsync(
                sourceLanguage, _trainer.SourceDictionary, sourceEmbeddings);

            // Оцениваем целевой язык если доступен
            Dictionary<string, float>? targetAnalogies = null;
            if (!string.IsNullOrEmpty(_trainer.TargetDictionary?.Language))
            {
                var targetEmbeddings = _trainer.TargetEmbeddings.weight!.cpu();
                targetAnalogies = await GetWordAnalogyScoresAsync(
                    _trainer.TargetDictionary.Language, _trainer.TargetDictionary, targetEmbeddings);
            }

            // Записываем результаты исходного языка
            if (sourceAnalogies != null && sourceAnalogies.Count > 0)
            {
                var avgScore = sourceAnalogies.Values.Average();
                _logger?.LogInformation($"Средний балл аналогий (исходный язык): {avgScore:F5}");

                stats.ToLog["src_analogy_monolingual_scores"] = avgScore;
                foreach (var kvp in sourceAnalogies)
                {
                    stats.ToLog[$"src_{kvp.Key}"] = kvp.Value;
                }
            }

            // Записываем результаты целевого языка
            if (targetAnalogies != null && targetAnalogies.Count > 0)
            {
                var avgScore = targetAnalogies.Values.Average();
                _logger?.LogInformation($"Средний балл аналогий (целевой язык): {avgScore:F5}");

                stats.ToLog["tgt_analogy_monolingual_scores"] = avgScore;
                foreach (var kvp in targetAnalogies)
                {
                    stats.ToLog[$"tgt_{kvp.Key}"] = kvp.Value;
                }
            }
        }

        /// <summary>
        /// Вычисляет баллы аналогий для языка (только английский поддерживается)
        /// </summary>
        private async Task<Dictionary<string, float>?> GetWordAnalogyScoresAsync(
            string language, Dictionary dictionary, Tensor embeddings)
        {
            var datasetDir = Path.Combine(MonolingualEvalPath, language);
            if (!Directory.Exists(datasetDir) || !language.Equals("en", StringComparison.OrdinalIgnoreCase))
            {
                return null;
            }

            var questionsFile = Path.Combine(datasetDir, "questions-words.txt");
            if (!File.Exists(questionsFile))
            {
                _logger?.LogWarning($"Файл questions-words.txt не найден в {datasetDir}");
                return null;
            }

            // Нормализуем эмбеддинги
            var normalizedEmbeddings = NormalizeEmbeddingsForAnalogy(embeddings);

            var scores = new Dictionary<string, Dictionary<string, int>>();
            var wordIds = new Dictionary<string, List<int[]>>();
            var queries = new Dictionary<string, List<float[]>>();

            // Парсим файл аналогий
            await foreach (var line in File.ReadLinesAsync(questionsFile))
            {
                var trimmedLine = line.Trim().ToLowerInvariant();
                if (string.IsNullOrEmpty(trimmedLine)) continue;

                // Новая категория
                if (trimmedLine.Contains(':'))
                {
                    var category = trimmedLine.Substring(2);
                    if (!scores.ContainsKey(category))
                    {
                        scores[category] = new Dictionary<string, int>
                        {
                            ["n_found"] = 0,
                            ["n_not_found"] = 0,
                            ["n_correct"] = 0
                        };
                        wordIds[category] = new List<int[]>();
                        queries[category] = new List<float[]>();
                    }
                    continue;
                }

                // Обрабатываем аналогию
                var parts = trimmedLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length != 4) continue;

                var currentCategory = scores.Keys.LastOrDefault();
                if (currentCategory == null) continue;

                var word1Id = EvaluationUtils.GetWordId(parts[0], dictionary);
                var word2Id = EvaluationUtils.GetWordId(parts[1], dictionary);
                var word3Id = EvaluationUtils.GetWordId(parts[2], dictionary);
                var word4Id = EvaluationUtils.GetWordId(parts[3], dictionary);

                if (word1Id == null || word2Id == null || word3Id == null || word4Id == null)
                {
                    scores[currentCategory]["n_not_found"]++;
                    continue;
                }

                scores[currentCategory]["n_found"]++;
                wordIds[currentCategory].Add(new[] { word1Id.Value, word2Id.Value, word3Id.Value, word4Id.Value });

                // Генерируем запросный вектор: word1 - word2 + word4
                var query = GenerateAnalogyQuery(normalizedEmbeddings, word1Id.Value, word2Id.Value, word4Id.Value);
                queries[currentCategory].Add(query);
            }

            // Вычисляем точность для каждой категории
            var accuracies = await ComputeAnalogyAccuracies(
                queries, wordIds, scores, normalizedEmbeddings);

            LogAnalogyResults(accuracies, scores);

            return accuracies;
        }

        #endregion

        #region Public Methods - Cross-lingual Word Similarity

        /// <summary>
        /// Оценка кросс-лингвального семантического сходства слов
        /// Реализация метода crosslingual_wordsim из оригинального evaluator.py
        /// Использует файлы данных.
        /// </summary>
        /// <param name="results">Словарь для записи результатов</param>
        public async Task EvaluateCrossLingualWordSimilarityAsync(TrainingStats stats)
        {
            _logger?.LogInformation("Оценка кросс-лингвального семантического сходства...");

            var sourceEmbeddings = GetMappedSourceEmbeddings();
            var targetEmbeddings = _trainer.TargetEmbeddings.weight!.cpu();

            var crossLingualScores = await GetCrossLingualWordSimilarityScoresAsync(
                _trainer.SourceDictionary.Language, _trainer.SourceDictionary, sourceEmbeddings,
                _trainer.TargetDictionary.Language, _trainer.TargetDictionary, targetEmbeddings);

            if (crossLingualScores == null || crossLingualScores.Count == 0)
            {
                _logger?.LogWarning("Кросс-лингвальные датасеты семантического сходства не найдены");
                return;
            }

            var avgScore = crossLingualScores.Values.Average();
            _logger?.LogInformation($"Средний балл кросс-лингвального семантического сходства: {avgScore:F5}");

            stats.ToLog["ws_crosslingual_scores"] = avgScore;
            foreach (var kvp in crossLingualScores)
            {
                stats.ToLog[$"src_tgt_{kvp.Key}"] = kvp.Value;
            }
        }

        /// <summary>
        /// Получает баллы кросс-лингвального семантического сходства
        /// </summary>
        private async Task<Dictionary<string, float>?> GetCrossLingualWordSimilarityScoresAsync(
            string lang1, Dictionary dict1, Tensor emb1,
            string lang2, Dictionary dict2, Tensor emb2)
        {
            var file1 = Path.Combine(SemEval17EvalPath, $"{lang1}-{lang2}-SEMEVAL17.txt");
            var file2 = Path.Combine(SemEval17EvalPath, $"{lang2}-{lang1}-SEMEVAL17.txt");

            string? actualFile = null;
            if (File.Exists(file1))
                actualFile = file1;
            else if (File.Exists(file2))
                actualFile = file2;

            if (actualFile == null)
            {
                return null;
            }

            var (correlation, found, notFound) = await ComputeSpearmanRhoCrossLingualAsync(
                dict1, emb1, dict2, emb2, actualFile);

            var scores = new Dictionary<string, float>();
            var separator = new string('=', 30 + 1 + 10 + 1 + 13 + 1 + 12);
            var pattern = "{0,-30} {1,10} {2,13} {3,12}";

            _logger?.LogInformation(separator);
            _logger?.LogInformation(string.Format(pattern, "Dataset", "Found", "Not found", "Rho"));
            _logger?.LogInformation(separator);

            var taskName = $"{lang1.ToUpper()}_{lang2.ToUpper()}_SEMEVAL17";
            _logger?.LogInformation(string.Format(pattern, taskName, found, notFound, $"{correlation:F4}"));
            scores[taskName] = correlation;

            _logger?.LogInformation(separator);
            return scores.Count > 0 ? scores : null;
        }

        #endregion

        #region Public Methods - Word Translation

        /// <summary>
        /// Оценка точности перевода слов
        /// Реализация метода word_translation из оригинального evaluator.py
        /// Использует файлы данных.
        /// </summary>
        /// <param name="results">Словарь для записи результатов</param>
        /// <param name="dictionaryPath">Путь к тестовому словарю</param>
        public async Task EvaluateWordTranslationAsync(TrainingStats stats, string dictionaryPath = "default")
        {
            _logger?.LogInformation("Оценка точности перевода слов...");

            // Загружаем тестовый словарь
            var testDictionary = await LoadEvaluationDictionaryAsync(
                    _trainer.SourceDictionary.Language, 
                    _trainer.TargetDictionary.Language, 
                    dictionaryPath, 
                    _trainer.SourceDictionary, 
                    _trainer.TargetDictionary);            
            testDictionary = testDictionary.to(_trainer.Device);

            var mappedSourceEmbeddings = _trainer.Mapping.forward(_trainer.SourceEmbeddings.weight!);
            var targetEmbeddings = _trainer.TargetEmbeddings.weight!;

            var methods = new[] { "nn", "csls_knn_10" };
            foreach (var method in methods)
            {
                var translationResults = await GetWordTranslationAccuracyAsync(                    
                        mappedSourceEmbeddings,
                        targetEmbeddings,
                        method, 
                        testDictionary);

                foreach (var result in translationResults)
                {
                    var key = $"{result.Key}-{method}";
                    stats.ToLog[key] = result.Value;
                }
            }
        }

        /// <summary>
        /// Вычисляет точность перевода слов для заданного метода
        /// </summary>
        private async Task<Dictionary<string, float>> GetWordTranslationAccuracyAsync(            
            Tensor mappedSourceEmb,
            Tensor targetEmb,
            string method, 
            torch.Tensor testDictionary)
        { 
            // Нормализуем эмбеддинги
            mappedSourceEmb = functional.normalize(mappedSourceEmb, p: 2, dim: 1);
            targetEmb = functional.normalize(targetEmb, p: 2, dim: 1);

            bool saveDictionary = false;
            Tensor scores;
            if (method == "nn")
            {
                // Простое скалярное произведение
                var testDictionary_SourceIds = testDictionary.select(dim: 1, index: 0);
                var queryMappedSourceEmb = mappedSourceEmb.index_select(dim: 0, testDictionary_SourceIds);
                scores = queryMappedSourceEmb.mm(targetEmb.transpose(dim0: 0, dim1: 1));

                saveDictionary = true;
            }
            else if (method.StartsWith("csls_knn_"))
            {
                // CSLS метод
                var k = int.Parse(method.Substring("csls_knn_".Length));
                var testDictionary_SourceIds = testDictionary.select(dim: 1, index: 0);
                scores = await ComputeCSLSScoresForTranslationAsync(mappedSourceEmb, targetEmb, testDictionary_SourceIds, k);
            }
            else
            {
                throw new ArgumentException($"Неподдерживаемый метод: {method}");
            }

            // Вычисляем Precision@K
            var results = ComputePrecisionAtK(scores, testDictionary, new[] { 1, 5, 10, 20, 40, 80 }, saveDictionary);

            foreach (var result in results)
            {
                _logger?.LogInformation($"{testDictionary.size(0)} исходных слов - {method} - Precision at k = {result.Key}: {result.Value:F3}");
            }

            return results;
        }

        #endregion

        #region Public Methods - Sentence Translation

        /// <summary>
        /// Оценка точности перевода предложений на корпусе Europarl
        /// Реализация метода sent_translation из оригинального evaluator.py
        /// Использует файлы данных.
        /// </summary>
        /// <param name="results">Словарь для записи результатов</param>
        public async Task EvaluateSentenceTranslationAsync(TrainingStats stats)
        {
            _logger?.LogInformation("Оценка точности перевода предложений...");

            var sourceLang = _trainer.SourceDictionary.Language;
            var targetLang = _trainer.TargetDictionary.Language;

            // Параметры для оценки предложений
            const int nKeys = 200000;
            const int nQueries = 2000;
            const int nIdf = 300000;

            // Загружаем данные Europarl
            if (_europarlData == null)
            {
                _europarlData = await LoadEuroparlDataAsync(sourceLang, targetLang, nKeys + 2 * nIdf);
            }

            if (_europarlData == null || _europarlData.Count == 0)
            {
                _logger?.LogWarning($"Europarl данные для пары {sourceLang}-{targetLang} не найдены");
                return;
            }

            var sourceEmbeddings = _trainer.Mapping.forward(_trainer.SourceEmbeddings.weight!);
            var targetEmbeddings = _trainer.TargetEmbeddings.weight!;

            // Вычисляем IDF веса
            var idf = ComputeIdfWeights(_europarlData, sourceLang, targetLang, nIdf);

            var methods = new[] { "nn", "csls_knn_10" };
            foreach (var method in methods)
            {
                // Перевод Target -> Source
                var tgtToSrcResults = await GetSentenceTranslationAccuracyAsync(
                    _europarlData, sourceLang, _trainer.SourceDictionary, sourceEmbeddings,
                    targetLang, _trainer.TargetDictionary, targetEmbeddings,
                    nKeys, nQueries, method, idf);

                foreach (var result in tgtToSrcResults)
                {
                    stats.ToLog[$"tgt_to_src_{result.Key}-{method}"] = result.Value;
                }

                // Перевод Source -> Target
                var srcToTgtResults = await GetSentenceTranslationAccuracyAsync(
                    _europarlData, targetLang, _trainer.TargetDictionary, targetEmbeddings,
                    sourceLang, _trainer.SourceDictionary, sourceEmbeddings,
                    nKeys, nQueries, method, idf);

                foreach (var result in srcToTgtResults)
                {
                    stats.ToLog[$"src_to_tgt_{result.Key}-{method}"] = result.Value;
                }
            }
        }

        #endregion

        #region Public Methods - Mean Cosine

        /// <summary>
        /// Вычисление критерия выбора модели Mean Cosine
        /// Реализация метода dist_mean_cosine из оригинального evaluator.py
        /// </summary>
        /// <param name="results">Словарь для записи результатов</param>
        public async Task EvaluateMeanCosineAsync(TrainingStats stats)
        {
            _logger?.LogInformation("Оценка критерия выбора модели Mean Cosine...");

            // Получаем нормализованные эмбеддинги
            using var _ = no_grad();

            torch.Tensor mappedSourceEmbeddings = _trainer.Mapping.forward(_trainer.SourceEmbeddings.weight!);
            torch.Tensor targetEmbeddings = _trainer.TargetEmbeddings.weight!;

            mappedSourceEmbeddings = functional.normalize(mappedSourceEmbeddings, p: 2, dim: 1);
            targetEmbeddings = functional.normalize(targetEmbeddings, p: 2, dim: 1);

            var methods = new[] { "nn", "csls_knn_10" };
            foreach (var method in methods)
            {
                const string dicoBuild = "SourceToTarget";
                const int dicoMaxSize = 10000;

                // Создаем временные параметры для построения словаря
                var parameters = new UnsupervisedParameters
                {
                    DicoMethod = method,
                    DicoBuild = dicoBuild,
                    DicoThreshold = 0,
                    DicoMaxRank = 10000,
                    DicoMinSize = 0,
                    DicoMaxSize = dicoMaxSize
                };

                // Строим словарь
                var dictionary = await DictionaryBuilder.BuildDictionaryAsync(
                    mappedSourceEmbeddings, 
                    targetEmbeddings, 
                    parameters, 
                    logger: _logger);

                float meanCosine;
                if (dictionary is null || dictionary.size(0) == 0)
                {
                    meanCosine = -1e9f;
                }
                else
                {
                    // Вычисляем mean cosine
                    var actualSize = Math.Min(dicoMaxSize, (int)dictionary.size(0));
                    var sourceIndices = dictionary.select(1, 0).narrow(0, 0, actualSize);
                    var targetIndices = dictionary.select(1, 1).narrow(0, 0, actualSize);

                    var sourceVectors = mappedSourceEmbeddings.index_select(0, sourceIndices);
                    var targetVectors = targetEmbeddings.index_select(0, targetIndices);

                    var cosines = (sourceVectors * targetVectors).sum(dim: 1);
                    meanCosine = cosines.mean().item<float>();
                }

                var key = $"mean_cosine-{method}-{dicoBuild}-{dicoMaxSize}";
                stats.ToLog[key] = meanCosine;

                _logger?.LogInformation($"Mean cosine ({method} метод, {dicoBuild} построение, {dicoMaxSize} макс размер): {meanCosine:F5}");
            }
        }

        #endregion

        #region Public Methods - Discriminator Evaluation

        /// <summary>
        /// Оценка дискриминатора
        /// Реализация метода eval_dis из оригинального evaluator.py
        /// </summary>
        /// <param name="results">Словарь для записи результатов</param>
        public Task EvaluateDiscriminatorAsync(TrainingStats stats)
        {
            if (_trainer.Discriminator == null)
            {
                _logger?.LogWarning("Дискриминатор не найден, пропускаем оценку");
                return Task.CompletedTask;
            }

            _logger?.LogInformation("Оценка дискриминатора...");

            using var _ = no_grad();
            _trainer.Discriminator.eval();

            var sourcePredictions = new List<float>();
            var targetPredictions = new List<float>();

            // Оценка предсказаний для исходных эмбеддингов
            var sourceVocabSize = _trainer.SourceEmbeddings.weight!.shape[0];
            for (int i = 0; i < sourceVocabSize; i += BatchSize)
            {
                var endIdx = Math.Min(sourceVocabSize, i + BatchSize);
                var indices = arange(i, endIdx, dtype: ScalarType.Int64, device: _trainer.SourceEmbeddings.weight!.device);

                var embeddings = _trainer.SourceEmbeddings.forward(indices);
                var mappedEmbeddings = _trainer.Mapping.forward(embeddings);
                var predictions = _trainer.Discriminator.forward(mappedEmbeddings);

                sourcePredictions.AddRange(predictions.data<float>().ToArray());
            }

            // Оценка предсказаний для целевых эмбеддингов
            var targetVocabSize = _trainer.TargetEmbeddings.weight!.shape[0];
            for (int i = 0; i < targetVocabSize; i += BatchSize)
            {
                var endIdx = Math.Min(targetVocabSize, i + BatchSize);
                var indices = arange(i, endIdx, dtype: ScalarType.Int64, device: _trainer.TargetEmbeddings.weight!.device);

                var embeddings = _trainer.TargetEmbeddings.forward(indices);
                var predictions = _trainer.Discriminator.forward(embeddings);

                targetPredictions.AddRange(predictions.data<float>().ToArray());
            }

            // Вычисляем метрики
            var sourceMeanPred = sourcePredictions.Average();
            var targetMeanPred = targetPredictions.Average();

            var sourceAccuracy = sourcePredictions.Count(x => x >= 0.5) / (float)sourcePredictions.Count;
            var targetAccuracy = targetPredictions.Count(x => x < 0.5) / (float)targetPredictions.Count;

            var discriminatorAccuracy = (sourceAccuracy * sourcePredictions.Count + targetAccuracy * targetPredictions.Count) /
                                      (sourcePredictions.Count + targetPredictions.Count);

            _logger?.LogInformation($"Дискриминатор исходные / целевые предсказания: {sourceMeanPred:F5} / {targetMeanPred:F5}");
            _logger?.LogInformation($"Дискриминатор исходная / целевая / общая точность: {sourceAccuracy:F5} / {targetAccuracy:F5} / {discriminatorAccuracy:F5}");

            stats.ToLog["dis_accu"] = (float)discriminatorAccuracy;
            stats.ToLog["dis_src_pred"] = sourceMeanPred;
            stats.ToLog["dis_tgt_pred"] = targetMeanPred;

            return Task.CompletedTask;
        }

        #endregion

        #region Public Methods - All Evaluation

        /// <summary>
        /// Запуск всех оценок
        /// Реализация метода all_eval из оригинального evaluator.py
        /// </summary>
        /// <param name="results">Словарь для записи результатов</param>
        /// <param name="dictionaryPath">Путь к тестовому словарю</param>
        public async Task RunAllEvaluationsAsync(TrainingStats stats, string dictionaryPath = "default")
        {
            _logger?.LogInformation("Запуск полного набора оценок...");

            //await EvaluateMonolingualWordSimilarityAsync(stats);
            //await EvaluateCrossLingualWordSimilarityAsync(stats);
            //await EvaluateWordTranslationAsync(stats, dictionaryPath);
            //await EvaluateSentenceTranslationAsync(stats);
            await EvaluateMeanCosineAsync(stats);
        }

        #endregion

        #region Private Helper Methods

        /// <summary>
        /// Получает выровненные исходные эмбеддинги
        /// </summary>
        private Tensor GetMappedSourceEmbeddings()
        {
            using var _ = no_grad();
            return _trainer.Mapping.forward(_trainer.SourceEmbeddings.weight!).cpu();
        }

        /// <summary>
        /// Вычисляет корреляцию Спирмена для семантического сходства
        /// </summary>
        private async Task<(float correlation, int found, int notFound)> ComputeSpearmanRhoAsync(
            Dictionary dictionary, Tensor embeddings, string filePath,
            Dictionary? dictionary2 = null, Tensor? embeddings2 = null)
        {
            dictionary2 ??= dictionary;
            embeddings2 ??= embeddings;

            var wordPairs = await GetWordPairsFromFileAsync(filePath);
            var predictions = new List<double>();
            var goldScores = new List<double>();
            int notFound = 0;

            var embeddingData = embeddings.data<float>().ToArray();
            var embedding2Data = embeddings2.data<float>().ToArray();
            int embDim = (int)embeddings.size(1);

            foreach (var (word1, word2, goldScore) in wordPairs)
            {
                var id1 = EvaluationUtils.GetWordId(word1, dictionary);
                var id2 = EvaluationUtils.GetWordId(word2, dictionary2);

                if (id1 == null || id2 == null)
                {
                    notFound++;
                    continue;
                }

                // Извлекаем векторы и вычисляем косинусное сходство
                var vector1 = new ReadOnlySpan<float>(embeddingData, id1.Value * embDim, embDim);
                var vector2 = new ReadOnlySpan<float>(embedding2Data, id2.Value * embDim, embDim);

                var similarity = ComputeCosineSimilarity(vector1, vector2);
                predictions.Add(similarity);
                goldScores.Add(goldScore);
            }

            if (predictions.Count < 2)
            {
                return (0.0f, predictions.Count, notFound);
            }

            var correlation = (float)Correlation.Spearman(predictions, goldScores);
            return (correlation, predictions.Count, notFound);
        }

        /// <summary>
        /// Вычисляет кросс-лингвальную корреляцию Спирмена
        /// </summary>
        private async Task<(float correlation, int found, int notFound)> ComputeSpearmanRhoCrossLingualAsync(
            Dictionary dict1, Tensor emb1, Dictionary dict2, Tensor emb2, string filePath)
        {
            // Определяем правильный порядок словарей и эмбеддингов
            var fileName = Path.GetFileName(filePath);
            if (fileName.StartsWith($"{dict2.Language}-{dict1.Language}"))
            {
                // Меняем местами если файл имеет обратный порядок языков
                (dict1, dict2) = (dict2, dict1);
                (emb1, emb2) = (emb2, emb1);
            }

            return await ComputeSpearmanRhoAsync(dict1, emb1, filePath, dict2, emb2);
        }

        /// <summary>
        /// Парсит файл с парами слов и их оценками сходства
        /// </summary>
        private async Task<List<(string word1, string word2, float score)>> GetWordPairsFromFileAsync(
            string filePath)
        {
            var pairs = new List<(string, string, float)>();

            await foreach (var line in File.ReadLinesAsync(filePath))
            {
                var trimmedLine = line.Trim();
                if (string.IsNullOrEmpty(trimmedLine)) continue;

                var processedLine = trimmedLine;
                var parts = processedLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);

                if (parts.Length < 3) continue;

                // Игнорируем фразы, рассматриваем только отдельные слова
                if (parts.Length != 3)
                {
                    // Проверяем исключения для SEMEVAL17 и EN-IT_MWS353
                    var fileName = Path.GetFileName(filePath);
                    if (fileName.Contains("SEMEVAL17") || fileName.Contains("EN-IT_MWS353"))
                        continue;
                }

                if (float.TryParse(parts[2], out var score))
                {
                    pairs.Add((parts[0], parts[1], score));
                }
            }

            return pairs;
        }        

        /// <summary>
        /// Вычисляет косинусное сходство между двумя векторами
        /// </summary>
        private float ComputeCosineSimilarity(ReadOnlySpan<float> vector1, ReadOnlySpan<float> vector2)
        {
            if (vector1.Length != vector2.Length)
                throw new ArgumentException("Векторы должны быть одинаковой длины");

            float dotProduct = 0.0f;
            float norm1 = 0.0f;
            float norm2 = 0.0f;

            for (int i = 0; i < vector1.Length; i++)
            {
                dotProduct += vector1[i] * vector2[i];
                norm1 += vector1[i] * vector1[i];
                norm2 += vector2[i] * vector2[i];
            }

            var magnitude = MathF.Sqrt(norm1) * MathF.Sqrt(norm2);
            return magnitude > 0.0f ? dotProduct / magnitude : 0.0f;
        }

        /// <summary>
        /// Нормализует эмбеддинги для аналогий
        /// </summary>
        private Tensor NormalizeEmbeddingsForAnalogy(Tensor embeddings)
        {
            using var _ = no_grad();
            var embData = embeddings.data<float>().ToArray();
            var rows = (int)embeddings.size(0);
            var cols = (int)embeddings.size(1);

            // Нормализуем каждый вектор
            for (int i = 0; i < rows; i++)
            {
                float sumSquares = 0.0f;
                var startIdx = i * cols;

                for (int j = 0; j < cols; j++)
                {
                    var val = embData[startIdx + j];
                    sumSquares += val * val;
                }

                var norm = Math.Sqrt(sumSquares);
                if (norm > 0)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        embData[startIdx + j] = (float)(embData[startIdx + j] / norm);
                    }
                }
            }

            return tensor(embData).reshape(rows, cols);
        }

        /// <summary>
        /// Генерирует запросный вектор для аналогии
        /// </summary>
        private float[] GenerateAnalogyQuery(Tensor embeddings, int word1Id, int word2Id, int word4Id)
        {
            var embData = embeddings.data<float>().ToArray();
            var embDim = (int)embeddings.size(1);

            var query = new float[embDim];
            var startIdx1 = word1Id * embDim;
            var startIdx2 = word2Id * embDim;
            var startIdx4 = word4Id * embDim;

            // query = word1 - word2 + word4
            for (int i = 0; i < embDim; i++)
            {
                query[i] = embData[startIdx1 + i] - embData[startIdx2 + i] + embData[startIdx4 + i];
            }

            // Нормализуем запросный вектор
            float norm = 0.0f;
            for (int i = 0; i < embDim; i++)
            {
                norm += query[i] * query[i];
            }

            norm = MathF.Sqrt(norm);
            if (norm > 0)
            {
                for (int i = 0; i < embDim; i++)
                {
                    query[i] = (float)(query[i] / norm);
                }
            }

            return query;
        }

        /// <summary>
        /// Вычисляет точности аналогий по категориям
        /// </summary>
        private async Task<Dictionary<string, float>> ComputeAnalogyAccuracies(
            Dictionary<string, List<float[]>> queries,
            Dictionary<string, List<int[]>> wordIds,
            Dictionary<string, Dictionary<string, int>> scores,
            Tensor normalizedEmbeddings)
        {
            var accuracies = new Dictionary<string, float>();
            var embData = normalizedEmbeddings.data<float>().ToArray();
            var vocabSize = (int)normalizedEmbeddings.size(0);
            var embDim = (int)normalizedEmbeddings.size(1);

            foreach (var category in queries.Keys)
            {
                if (queries[category].Count == 0) continue;

                var categoryQueries = queries[category];
                var categoryWordIds = wordIds[category];
                int correctCount = 0;

                for (int i = 0; i < categoryQueries.Count; i++)
                {
                    var query = categoryQueries[i];
                    var wordIdSet = categoryWordIds[i];

                    // Находим наиболее похожее слово
                    int bestWordId = -1;
                    float bestSimilarity = float.MinValue;

                    for (int wordId = 0; wordId < vocabSize; wordId++)
                    {
                        // Исключаем входные слова
                        if (wordIdSet.Contains(wordId)) continue;

                        var startIdx = wordId * embDim;
                        var wordVector = new ReadOnlySpan<float>(embData, startIdx, embDim);

                        var similarity = ComputeCosineSimilarity(query.AsSpan(), wordVector);
                        if (similarity > bestSimilarity)
                        {
                            bestSimilarity = similarity;
                            bestWordId = wordId;
                        }
                    }

                    // Проверяем правильность ответа (ожидаем word3Id)
                    if (bestWordId == wordIdSet[2])
                    {
                        correctCount++;
                    }
                }

                scores[category]["n_correct"] = correctCount;
                accuracies[category] = (float)correctCount / Math.Max(scores[category]["n_found"], 1);
            }

            await Task.CompletedTask; // Для асинхронности
            return accuracies;
        }

        /// <summary>
        /// Логирует результаты аналогий
        /// </summary>
        private void LogAnalogyResults(Dictionary<string, float> accuracies, 
            Dictionary<string, Dictionary<string, int>> scores)
        {
            var separator = new string('=', 30 + 1 + 10 + 1 + 13 + 1 + 12);
            var pattern = "{0,-30} {1,10} {2,13} {3,12}";

            _logger?.LogInformation(separator);
            _logger?.LogInformation(string.Format(pattern, "Category", "Found", "Not found", "Accuracy"));
            _logger?.LogInformation(separator);

            foreach (var category in accuracies.Keys.OrderBy(x => x))
            {
                var stats = scores[category];
                _logger?.LogInformation(string.Format(pattern, 
                    category, 
                    stats["n_found"], 
                    stats["n_not_found"], 
                    $"{accuracies[category]:F4}"));
            }

            _logger?.LogInformation(separator);
        }

        /// <summary>
        /// Загружает тестовый словарь для оценки перевода
        /// </summary>
        private async Task<Tensor> LoadEvaluationDictionaryAsync(
            string sourceLang, string targetLang, string dictionaryPath,
            Dictionary sourceDict, Dictionary targetDict)
        {
            string actualPath;
            if (dictionaryPath == "default")
            {
                actualPath = Path.Combine(DictionariesPath, $"{sourceLang}-{targetLang}.5000-6500.txt");
            }
            else
            {
                actualPath = dictionaryPath;
            }

            if (!File.Exists(actualPath))
            {
                _logger?.LogWarning($"Тестовый словарь не найден: {actualPath}");
                return zeros(0, 2, dtype: ScalarType.Int64);
            }

            var pairs = new List<(int sourceId, int targetId)>();
            int notFound = 0;
            int notFound1 = 0;
            int notFound2 = 0;

            if (String.Equals(Path.GetExtension(actualPath), @".csv", StringComparison.InvariantCultureIgnoreCase))
            {
                await foreach (var line in File.ReadLinesAsync(actualPath))
                {
                    var trimmedLine = line.Trim().ToLowerInvariant();
                    if (string.IsNullOrEmpty(trimmedLine)) continue;

                    var parts = trimmedLine.Split(',', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length < 2)
                    {
                        _logger?.LogWarning($"Не удалось распарсить строку: {line}");
                        continue;
                    }

                    var sourceWord = parts[0].Trim();
                    var targetWord = parts[1].Trim();

                    var sourceId = EvaluationUtils.GetWordId(sourceWord, sourceDict);
                    var targetId = EvaluationUtils.GetWordId(targetWord, targetDict);

                    if (sourceId.HasValue && targetId.HasValue)
                    {
                        pairs.Add((sourceId.Value, targetId.Value));
                    }
                    else
                    {
                        notFound++;
                        if (!sourceId.HasValue) notFound1++;
                        if (!targetId.HasValue) notFound2++;
                    }
                }
            }
            else
            {
                await foreach (var line in File.ReadLinesAsync(actualPath))
                {
                    var trimmedLine = line.Trim().ToLowerInvariant();
                    if (string.IsNullOrEmpty(trimmedLine)) continue;

                    var parts = trimmedLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length < 2)
                    {
                        _logger?.LogWarning($"Не удалось распарсить строку: {line}");
                        continue;
                    }

                    var sourceWord = parts[0];
                    var targetWord = parts[1];

                    var sourceId = EvaluationUtils.GetWordId(sourceWord, sourceDict);
                    var targetId = EvaluationUtils.GetWordId(targetWord, targetDict);

                    if (sourceId.HasValue && targetId.HasValue)
                    {
                        pairs.Add((sourceId.Value, targetId.Value));
                    }
                    else
                    {
                        notFound++;
                        if (!sourceId.HasValue) notFound1++;
                        if (!targetId.HasValue) notFound2++;
                    }
                }
            }

            _logger?.LogInformation($"Найдено {pairs.Count} пар слов в словаре ({pairs.Select(p => p.sourceId).Distinct().Count()} уникальных). " +
                                  $"{notFound} других пар содержали хотя бы одно неизвестное слово " +
                                  $"({notFound1} в исходном языке, {notFound2} в целевом языке)");

            if (pairs.Count == 0)
                return zeros(0, 2, dtype: ScalarType.Int64);

            var data = new long[pairs.Count * 2];
            for (int i = 0; i < pairs.Count; i++)
            {
                data[i * 2] = pairs[i].sourceId;
                data[i * 2 + 1] = pairs[i].targetId;
            }

            return tensor(data, dtype: ScalarType.Int64).reshape(pairs.Count, 2);
        }

        /// <summary>
        /// Вычисляет CSLS скоры для перевода слов
        /// </summary>
        private async Task<Tensor> ComputeCSLSScoresForTranslationAsync(
            Tensor mappedSourceEmb, Tensor targetEmb, Tensor testDictionary_SourceIds, int k)
        {
            // Вычисляем средние расстояния до k ближайших соседей
            var mappedSourceEmb_AvgDist = await EvaluationUtils.ComputeAverageDistancesAsync(emb: targetEmb, query: mappedSourceEmb, k);
            var targetEmb_AvgDist = await EvaluationUtils.ComputeAverageDistancesAsync(emb: mappedSourceEmb, query: targetEmb, k);

            // Получаем запросные эмбеддинги            
            var queryMappedSourceEmb = mappedSourceEmb.index_select(dim: 0, testDictionary_SourceIds);

            // Базовые скоры
            var queryScores = queryMappedSourceEmb.mm(targetEmb.transpose(dim0: 0, dim1: 1));

            // Применяем CSLS
            queryScores = queryScores.mul(2);
            queryScores = queryScores.sub(mappedSourceEmb_AvgDist.index_select(dim: 0, testDictionary_SourceIds).unsqueeze(dim: 1));
            queryScores = queryScores.sub(targetEmb_AvgDist.unsqueeze(dim: 0));

            return queryScores;
        }        

        /// <summary>
        /// Вычисляет Precision@K для перевода слов
        /// </summary>
        private Dictionary<string, float> ComputePrecisionAtK(Tensor scores, Tensor testDictionary, int[] kValues, bool saveDictionary)
        {
            var results = new Dictionary<string, float>();
            var (_, topMatches) = scores.topk(k: kValues.Max(), dim: 1, largest: true, sorted: true);
            
            var testDictionary_SourceIds_Array = testDictionary.select(dim: 1, index: 0).data<long>().ToArray();
            var testDictionary_TargetIds_Array = testDictionary.select(dim: 1, index: 1).data<long>().ToArray();

            foreach (var k in kValues)
            {
                var topKMatches = topMatches[TensorIndex.Ellipsis, TensorIndex.Slice(null, k)];                

                var i_Matching_Dictionary = new Dictionary<int, int>();
                var topKData = topKMatches.data<long>().ToArray();                

                for (int i = 0; i < testDictionary_TargetIds_Array.Length; i++)
                {
                    //var sourceId = (int)testDictionary_SourceIds_Array[i];
                    var targetId = (int)testDictionary_TargetIds_Array[i];

                    var isMatch = false;
                    for (int j = 0; j < k; j++)
                    {
                        int index = i * k + j;
                        if (index < topKData.Length && topKData[index] == targetId)
                        {
                            isMatch = true;
                            break;
                        }
                    }

                    if (isMatch)
                    {
                        i_Matching_Dictionary[i] = 1;
                    }
                    else
                    {
                        i_Matching_Dictionary[i] = 0;
                    }
                }

                // TEMPCODE
                if (saveDictionary && k > 5)
                {
                    List<string?[]> data = new();
                    foreach (var kvp in i_Matching_Dictionary)
                    {   
                        if (kvp.Value == 1)
                        {
                            data.Add([
                                _trainer.SourceDictionary.IdToWord[(int)testDictionary_SourceIds_Array[kvp.Key]],
                                _trainer.TargetDictionary.IdToWord[(int)testDictionary_TargetIds_Array[kvp.Key]],
                            ]);
                        }                        
                    }
                    CsvHelper.SaveCsvFile(Path.Combine("Data", "PrimaryWords_RU_EN_Linear.csv"), data);
                }

                float precisionAtK;
                if (i_Matching_Dictionary.Count > 0)
                    precisionAtK = (float)i_Matching_Dictionary.Values.Sum() / (float)i_Matching_Dictionary.Count;
                else
                    precisionAtK = 0.0f;

                results[$"precision_at_{k}"] = precisionAtK;
            }

            return results;
        }

        /// <summary>
        /// Загружает данные Europarl для оценки перевода предложений
        /// </summary>
        private async Task<Dictionary<string, Dictionary<string, string[][]>>?> LoadEuroparlDataAsync(
            string lang1, string lang2, int maxSentences, bool lower = true)
        {
            var file1 = Path.Combine(EuroparlDir, $"europarl-v7.{lang1}-{lang2}.{lang1}");
            var file2 = Path.Combine(EuroparlDir, $"europarl-v7.{lang2}-{lang1}.{lang1}");

            if (!File.Exists(file1) && !File.Exists(file2))
            {
                return null;
            }

            if (File.Exists(file2) && !File.Exists(file1))
            {
                (lang1, lang2) = (lang2, lang1);
            }

            var data = new Dictionary<string, List<string[]>>
            {
                [lang1] = new List<string[]>(),
                [lang2] = new List<string[]>()
            };

            // Загружаем предложения для каждого языка
            foreach (var lang in new[] { lang1, lang2 })
            {
                var filePath = Path.Combine(EuroparlDir, $"europarl-v7.{lang1}-{lang2}.{lang}");
                if (!File.Exists(filePath)) continue;

                var sentenceCount = 0;
                await foreach (var line in File.ReadLinesAsync(filePath))
                {
                    if (sentenceCount >= maxSentences) break;

                    var processedLine = lower ? line.ToLowerInvariant() : line;
                    var words = processedLine.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    data[lang].Add(words);
                    sentenceCount++;
                }
            }

            // Обеспечиваем одинаковое количество предложений
            if (data[lang1].Count != data[lang2].Count)
            {
                var minCount = Math.Min(data[lang1].Count, data[lang2].Count);
                data[lang1] = data[lang1].Take(minCount).ToList();
                data[lang2] = data[lang2].Take(minCount).ToList();
            }

            // Удаляем дубликаты и перемешиваем
            var uniqueData = RemoveDuplicatesAndShuffle(data, lang1, lang2);

            _logger?.LogInformation($"Загружено europarl {lang1}-{lang2} ({uniqueData[lang1].Count} предложений)"); // VALFIX

            return uniqueData;
        }

        /// <summary>
        /// Удаляет дубликаты и перемешивает данные Europarl
        /// </summary>
        private Dictionary<string, Dictionary<string, string[][]>> RemoveDuplicatesAndShuffle(
            Dictionary<string, List<string[]>> data, string lang1, string lang2)
        {
            var sentences1 = data[lang1].ToArray();
            var sentences2 = data[lang2].ToArray();

            // Удаляем дубликаты для lang1 и соответствующие предложения из lang2
            var uniqueIndices1 = new List<int>();
            var seen1 = new HashSet<string>();

            for (int i = 0; i < sentences1.Length; i++)
            {
                var sentenceStr = string.Join(" ", sentences1[i]);
                if (seen1.Add(sentenceStr))
                {
                    uniqueIndices1.Add(i);
                }
            }

            var filteredSentences1 = uniqueIndices1.Select(i => sentences1[i]).ToArray();
            var filteredSentences2 = uniqueIndices1.Select(i => sentences2[i]).ToArray();

            // Удаляем дубликаты для lang2
            var uniqueIndices2 = new List<int>();
            var seen2 = new HashSet<string>();

            for (int i = 0; i < filteredSentences2.Length; i++)
            {
                var sentenceStr = string.Join(" ", filteredSentences2[i]);
                if (seen2.Add(sentenceStr))
                {
                    uniqueIndices2.Add(i);
                }
            }

            var finalSentences1 = uniqueIndices2.Select(i => filteredSentences1[i]).ToArray();
            var finalSentences2 = uniqueIndices2.Select(i => filteredSentences2[i]).ToArray();

            // Перемешиваем с фиксированным seed
            var random = new Random(1234);
            var indices = Enumerable.Range(0, finalSentences1.Length).ToArray();
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            var shuffled1 = indices.Select(i => finalSentences1[i]).ToArray();
            var shuffled2 = indices.Select(i => finalSentences2[i]).ToArray();

            return new Dictionary<string, Dictionary<string, string[][]>>
            {
                [lang1] = new Dictionary<string, string[][]> { [lang1] = shuffled1 },
                [lang2] = new Dictionary<string, string[][]> { [lang2] = shuffled2 }
            };
        }

        /// <summary>
        /// Вычисляет IDF веса для предложений
        /// </summary>
        private Dictionary<string, Dictionary<string, float>> ComputeIdfWeights(
            Dictionary<string, Dictionary<string, string[][]>> europarlData, 
            string lang1, string lang2, int nIdf)
        {
            var idf = new Dictionary<string, Dictionary<string, float>>
            {
                [lang1] = new Dictionary<string, float>(),
                [lang2] = new Dictionary<string, float>()
            };

            int k = 0;
            foreach (var lang in new[] { lang1, lang2 })
            {
                var startIdx = 200000 + k * nIdf;
                var endIdx = 200000 + (k + 1) * nIdf;

                var sentences = europarlData[lang][lang];
                if (sentences.Length <= startIdx)
                {
                    k++;
                    continue;
                }

                var actualEndIdx = Math.Min(endIdx, sentences.Length);
                var sentencesToProcess = sentences[startIdx..actualEndIdx];

                foreach (var sentence in sentencesToProcess)
                {
                    var uniqueWords = new HashSet<string>(sentence);
                    foreach (var word in uniqueWords)
                    {
                        idf[lang][word] = idf[lang].GetValueOrDefault(word, 0) + 1;
                    }
                }

                var nDoc = sentencesToProcess.Length;
                var wordsToUpdate = idf[lang].Keys.ToList();
                foreach (var word in wordsToUpdate)
                {
                    idf[lang][word] = MathF.Max(1, MathF.Log10((float)nDoc / idf[lang][word]));
                }

                k++;
            }

            return idf;
        }

        /// <summary>
        /// Получает точность перевода предложений
        /// </summary>
        private async Task<Dictionary<string, float>> GetSentenceTranslationAccuracyAsync(
            Dictionary<string, Dictionary<string, string[][]>> data,
            string queryLang, Dictionary queryDict, Tensor queryEmb,
            string keyLang, Dictionary keyDict, Tensor keyEmb,
            int nKeys, int nQueries, string method,
            Dictionary<string, Dictionary<string, float>> idf)
        {
            // Создаем словари векторов слов
            var queryEmbData = queryEmb.cpu().data<float>().ToArray();
            var keyEmbData = keyEmb.cpu().data<float>().ToArray();
            var embDim = (int)queryEmb.size(1);

            var queryWordVectors = new Dictionary<string, float[]>();
            var keyWordVectors = new Dictionary<string, float[]>();

            foreach (var word in queryDict.WordToId.Keys)
            {
                var id = queryDict.WordToId[word];
                var vector = new float[embDim];
                Array.Copy(queryEmbData, id * embDim, vector, 0, embDim);
                queryWordVectors[word] = vector;
            }

            foreach (var word in keyDict.WordToId.Keys)
            {
                var id = keyDict.WordToId[word];
                var vector = new float[embDim];
                Array.Copy(keyEmbData, id * embDim, vector, 0, embDim);
                keyWordVectors[word] = vector;
            }

            // Получаем предложения
            var keySentences = data[keyLang][keyLang].Take(nKeys).ToArray();
            var keys = BagOfWordsIdf(keySentences, keyWordVectors, idf[keyLang]);

            // Выбираем случайные запросы
            var random = new Random(1234);
            var queryIndices = Enumerable.Range(0, nKeys).OrderBy(x => random.Next()).Take(nQueries).ToArray();
            var querySentences = queryIndices.Select(i => data[queryLang][queryLang][i]).ToArray();
            var queries = BagOfWordsIdf(querySentences, queryWordVectors, idf[queryLang]);

            // Нормализуем векторы предложений
            var normalizedQueries = NormalizeSentenceVectors(queries);
            var normalizedKeys = NormalizeSentenceVectors(keys);

            // Вычисляем скоры в зависимости от метода
            var scores = method switch
            {
                "nn" => ComputeNeuralNetworkScores(normalizedQueries, normalizedKeys),
                var m when m.StartsWith("csls_knn_") => await ComputeCSLSScoresForSentencesAsync(normalizedQueries, normalizedKeys, m),
                _ => throw new ArgumentException($"Неподдерживаемый метод: {method}")
            };

            // Вычисляем Precision@K
            var results = new Dictionary<string, float>();
            var (_, topMatches) = scores.topk(10, dim: 1, largest: true, sorted: true);

            foreach (var k in new[] { 1, 5, 10 })
            {
                var topKMatches = topMatches[TensorIndex.Ellipsis, TensorIndex.Slice(null, k)];
                var targetIndices = tensor(queryIndices.Select(i => (long)i).ToArray());

                var matches = topKMatches.eq(targetIndices.unsqueeze(1).expand_as(topKMatches)).sum(dim: 1);
                var precisionAtK = 100.0f * matches.to_type(ScalarType.Float32).mean().item<float>();

                _logger?.LogInformation($"{nQueries} запросов ({queryLang.ToUpper()}) - {method} - Precision at k = {k}: {precisionAtK:F3}");
                results[$"sent-precision_at_{k}"] = precisionAtK;
            }

            return results;
        }

        /// <summary>
        /// Создает bag-of-words представления с IDF весами
        /// </summary>
        private float[][] BagOfWordsIdf(string[][] sentences, Dictionary<string, float[]> wordVectors, 
            Dictionary<string, float> idfWeights)
        {
            var sentenceVectors = new List<float[]>();

            foreach (var sentence in sentences)
            {
                var uniqueWords = sentence.Distinct().Where(w => wordVectors.ContainsKey(w) && idfWeights.ContainsKey(w)).ToArray();

                if (uniqueWords.Length > 0)
                {
                    var embDim = wordVectors.Values.First().Length;
                    var sentenceVector = new float[embDim];
                    var totalWeight = 0.0;

                    foreach (var word in uniqueWords)
                    {
                        var wordVector = wordVectors[word];
                        var weight = idfWeights[word];

                        for (int i = 0; i < embDim; i++)
                        {
                            sentenceVector[i] += (float)(wordVector[i] * weight);
                        }
                        totalWeight += weight;
                    }

                    if (totalWeight > 0)
                    {
                        for (int i = 0; i < embDim; i++)
                        {
                            sentenceVector[i] /= (float)totalWeight;
                        }
                    }

                    sentenceVectors.Add(sentenceVector);
                }
                else
                {
                    // Используем случайный вектор слова если ничего не найдено
                    var randomVector = wordVectors.Values.First().ToArray();
                    sentenceVectors.Add(randomVector);
                }
            }

            return sentenceVectors.ToArray();
        }

        /// <summary>
        /// Нормализует векторы предложений
        /// </summary>
        private Tensor NormalizeSentenceVectors(float[][] vectors)
        {
            var flatData = new List<float>();
            foreach (var vector in vectors)
            {
                flatData.AddRange(vector);
            }

            var tensor = torch.tensor(flatData.ToArray()).reshape(vectors.Length, vectors[0].Length);
            return functional.normalize(tensor, p: 2, dim: 1);
        }

        /// <summary>
        /// Вычисляет скоры для нейронной сети
        /// </summary>
        private Tensor ComputeNeuralNetworkScores(Tensor queries, Tensor keys)
        {
            return keys.mm(queries.transpose(0, 1)).transpose(0, 1);
        }

        /// <summary>
        /// Вычисляет CSLS скоры для предложений
        /// </summary>
        private async Task<Tensor> ComputeCSLSScoresForSentencesAsync(Tensor queries, Tensor keys, string method)
        {
            var k = int.Parse(method.Substring("csls_knn_".Length));

            var avgDistKeys = await EvaluationUtils.ComputeAverageDistancesAsync(queries, keys, k);
            var avgDistQueries = await EvaluationUtils.ComputeAverageDistancesAsync(keys, queries, k);

            var scores = keys.mm(queries.transpose(0, 1)).transpose(0, 1);
            scores = scores.mul(2);
            scores = scores.sub(avgDistQueries.unsqueeze(1)); // TODO verify
            scores = scores.sub(avgDistKeys.unsqueeze(0)); // TODO verify

            return scores;
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
        /// Освобождает управляемые ресурсы
        /// </summary>
        private void Dispose(bool disposing)
        {
            if (!_disposed && disposing)
            {
                _europarlData?.Clear();
                _europarlData = null;
                _disposed = true;
            }
        }

        #endregion
    }
}