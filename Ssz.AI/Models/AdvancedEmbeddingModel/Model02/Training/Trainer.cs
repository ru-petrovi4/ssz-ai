using System.Numerics.Tensors;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02.Models;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02.Utils;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02.Dictionary;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02.Evaluation;
using System.Collections.Generic;
using System;
using System.Linq;
using Ssz.Utils.Logging;
using Microsoft.Extensions.Logging;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02.Training;

/// <summary>
/// Основной класс тренера для несупервизированного обучения MUSE.
/// Реализует adversarial обучение с итеративным построением словаря и Procrustes решением.
/// Оптимизирован для максимальной производительности через векторизованные операции.
/// </summary>
public class Trainer
{
    private readonly MatrixFloat _sourceEmbeddings;
    private readonly MatrixFloat _targetEmbeddings;
    private readonly MatrixFloat _mappingMatrix;
    private readonly Models.Discriminator? _discriminator;
    private readonly Parameters _parameters;
    private readonly Dictionary.Dictionary _sourceDictionary;
    private readonly Dictionary.Dictionary? _targetDictionary;
    private readonly IUserFriendlyLogger _logger;

    // Оптимизаторы (простая реализация SGD)
    private readonly Utils.SGDOptimizer _mapOptimizer;
    private readonly Utils.SGDOptimizer? _discriminatorOptimizer;

    // Статистика обучения
    private readonly List<float> _mapLosses = new();
    private readonly List<float> _discriminatorLosses = new();
    private readonly List<float> _validationScores = new();

    /// <summary>
    /// Инициализация тренера с компонентами модели.
    /// </summary>
    /// <param name="sourceEmbeddings">Исходные эмбеддинги</param>
    /// <param name="targetEmbeddings">Целевые эмбеддинги</param>
    /// <param name="mappingMatrix">Отображающая матрица</param>
    /// <param name="discriminator">Дискриминатор (может быть null)</param>
    /// <param name="sourceDictionary">Словарь исходного языка</param>
    /// <param name="targetDictionary">Словарь целевого языка</param>
    /// <param name="parameters">Параметры обучения</param>
    public Trainer(MatrixFloat sourceEmbeddings, MatrixFloat targetEmbeddings,
                   MatrixFloat mappingMatrix, Models.Discriminator? discriminator,
                   Dictionary.Dictionary sourceDictionary, Dictionary.Dictionary? targetDictionary,
                   Parameters parameters)
    {
        _sourceEmbeddings = sourceEmbeddings;
        _targetEmbeddings = targetEmbeddings;
        _mappingMatrix = mappingMatrix;
        _discriminator = discriminator;
        _sourceDictionary = sourceDictionary;
        _targetDictionary = targetDictionary;
        _parameters = parameters;
        _logger = LoggersSet.Default.UserFriendlyLogger;

        // Инициализация оптимизаторов
        _mapOptimizer = new Utils.SGDOptimizer(_parameters.MapLearningRate, _parameters.MapWeightDecay);

        if (_discriminator != null)
        {
            _discriminatorOptimizer = new Utils.SGDOptimizer(_parameters.DiscriminatorLearningRate,
                                                     _parameters.DiscriminatorWeightDecay);
        }

        _logger.LogInformation("Тренер инициализирован успешно");
    }

    /// <summary>
    /// Основной цикл обучения MUSE.
    /// Чередует обучение дискриминатора и оптимизацию отображения.
    /// </summary>
    public void Train()
    {
        _logger.LogInformation("Начало обучения MUSE...");

        // Инициализация словаря с наиболее частотными словами
        var dictionary = InitializeFrequentWordDictionary();

        for (int epoch = 0; epoch < _parameters.Epochs; epoch++)
        {
            _logger.LogInformation($"Эпоха {epoch + 1}/{_parameters.Epochs}");

            // Adversarial обучение
            if (_discriminator != null && _discriminatorOptimizer != null)
            {
                TrainDiscriminatorEpoch();
                TrainMappingAdversarial();
            }

            // Procrustes решение
            var newDictionary = BuildDictionaryAndRefine(dictionary);
            dictionary = newDictionary;

            // Валидация
            if ((epoch + 1) % _parameters.ValidationMetricStep == 0)
            {
                var score = ValidateModel();
                _validationScores.Add(score);
                _logger.LogInformation($"Validation score: {score:F4}");
            }

            // Ортогонализация отображающей матрицы
            OrthogonalizeMapping();
        }

        _logger.LogInformation("Обучение завершено успешно");
    }

    /// <summary>
    /// Инициализация словаря наиболее частотными словами для стабильного старта.
    /// Использует наиболее частые слова из обоих языков для создания начального словаря.
    /// </summary>
    /// <returns>Начальный словарь переводов</returns>
    private List<(int sourceId, int targetId)> InitializeFrequentWordDictionary()
    {
        _logger.LogInformation("Инициализация словаря наиболее частотными словами...");

        int dictionarySize = Math.Min(_parameters.MostFrequentValidation,
                                    Math.Min(_sourceDictionary.Length,
                                           _targetDictionary?.Length ?? _sourceDictionary.Length));

        var dictionary = new List<(int, int)>();

        // Простая стратегия: сопоставляем по порядку частотности
        for (int i = 0; i < dictionarySize; i++)
        {
            dictionary.Add((i, i));
        }

        _logger.LogInformation($"Инициализирован словарь с {dictionary.Count} парами");
        return dictionary;
    }

    /// <summary>
    /// Обучение дискриминатора для различения исходных и целевых эмбеддингов.
    /// Использует adversarial подход для улучшения качества отображения.
    /// </summary>
    private void TrainDiscriminatorEpoch()
    {
        if (_discriminator == null || _discriminatorOptimizer == null) return;

        _logger.LogDebug("Обучение дискриминатора...");

        var batchSize = 128; // Размер мини-батча
        var sourceSize = _sourceEmbeddings.Dimensions[0];
        var targetSize = _targetEmbeddings.Dimensions[0];
        var random = new Random();

        float totalLoss = 0.0f;
        int numBatches = _parameters.DiscriminatorSteps;

        for (int step = 0; step < numBatches; step++)
        {
            // Формирование батча из исходных эмбеддингов (класс 0)
            var sourceBatch = CreateBatch(_sourceEmbeddings, batchSize, random);

            // Применение текущего отображения к исходным эмбеддингам
            var mappedSourceBatch = ApplyMapping(sourceBatch);

            // Формирование батча из целевых эмбеддингов (класс 1)
            var targetBatch = CreateBatch(_targetEmbeddings, batchSize, random);

            // Объединение батчей
            var combinedBatch = CombineBatches(mappedSourceBatch, targetBatch);
            var labels = CreateLabels(batchSize); // [0,0,...,0,1,1,...,1]

            // Прямое прохождение
            var predictions = _discriminator.Forward(combinedBatch, isTraining: true);

            // Вычисление loss (binary cross-entropy)
            var loss = ComputeBinaryCrossEntropyLoss(predictions, labels);
            totalLoss += loss;

            // Обратное распространение (упрощенная версия)
            var gradients = ComputeDiscriminatorGradients(combinedBatch, predictions, labels);

            // Обновление весов дискриминатора
            _discriminatorOptimizer.UpdateWeights(_discriminator, gradients);
        }

        var avgLoss = totalLoss / numBatches;
        _discriminatorLosses.Add(avgLoss);
        _logger.LogDebug($"Discriminator loss: {avgLoss:F6}");
    }

    /// <summary>
    /// Обучение отображающей матрицы через adversarial подход.
    /// Пытается обмануть дискриминатор, улучшая качество отображения.
    /// </summary>
    private void TrainMappingAdversarial()
    {
        if (_discriminator == null) return;

        _logger.LogDebug("Adversarial обучение отображения...");

        var batchSize = 128;
        var random = new Random();
        float totalLoss = 0.0f;
        int numSteps = _parameters.MapOptimizerSteps;

        for (int step = 0; step < numSteps; step++)
        {
            // Формирование батча исходных эмбеддингов
            var sourceBatch = CreateBatch(_sourceEmbeddings, batchSize, random);

            // Применение отображения
            var mappedBatch = ApplyMapping(sourceBatch);

            // Получение предсказаний дискриминатора
            var predictions = _discriminator.Forward(mappedBatch, isTraining: false);

            // Adversarial loss: хотим, чтобы дискриминатор предсказывал класс 1 (целевой язык)
            var targetLabels = Enumerable.Repeat(1.0f, batchSize).ToArray();
            var loss = ComputeBinaryCrossEntropyLoss(predictions, targetLabels);
            totalLoss += loss;

            // Вычисление градиентов для отображающей матрицы
            var mappingGradients = ComputeMappingGradients(sourceBatch, predictions, targetLabels);

            // Обновление отображающей матрицы
            _mapOptimizer.UpdateMappingMatrix(_mappingMatrix, mappingGradients);
        }

        var avgLoss = totalLoss / numSteps;
        _mapLosses.Add(avgLoss);
        _logger.LogDebug($"Mapping adversarial loss: {avgLoss:F6}");
    }

    /// <summary>
    /// Построение словаря и уточнение отображения методом Procrustes.
    /// Ключевой этап MUSE, использующий CSLS для поиска ближайших соседей.
    /// </summary>
    /// <param name="currentDictionary">Текущий словарь</param>
    /// <returns>Обновленный словарь</returns>
    private List<(int sourceId, int targetId)> BuildDictionaryAndRefine(
        List<(int sourceId, int targetId)> currentDictionary)
    {
        _logger.LogDebug("Построение словаря и Procrustes уточнение...");

        // Применение текущего отображения к исходным эмбеддингам
        var mappedSource = ApplyMapping(_sourceEmbeddings);

        // Построение нового словаря через CSLS
        var newDictionary = DictionaryBuilder.BuildDictionary(
            mappedSource, _targetEmbeddings, _parameters);

        // Получение соответствующих пар эмбеддингов для Procrustes
        var (sourceMatrix, targetMatrix) = ExtractDictionaryEmbeddings(newDictionary);

        // Procrustes решение: нахождение оптимальной ортогональной матрицы
        var newMapping = Utils.MathUtils.ProcrustesAlignment(sourceMatrix, targetMatrix);

        // Обновление отображающей матрицы
        CopyMatrix(newMapping, _mappingMatrix);

        _logger.LogDebug($"Словарь обновлен: {newDictionary.Count} пар");
        return newDictionary;
    }

    /// <summary>
    /// Валидация модели на контрольном наборе.
    /// Использует метрику точности перевода слов.
    /// </summary>
    /// <returns>Оценка качества модели</returns>
    private float ValidateModel()
    {
        _logger.LogDebug("Валидация модели...");

        var evaluator = new WordTranslationEvaluator();
        var mappedSource = ApplyMapping(_sourceEmbeddings);

        // Вычисление точности перевода на валидационном наборе
        var accuracy = evaluator.EvaluateAccuracy(
            mappedSource, _targetEmbeddings,
            _sourceDictionary, _targetDictionary,
            _parameters.MostFrequentValidation);

        return accuracy;
    }

    /// <summary>
    /// Ортогонализация отображающей матрицы для сохранения геометрических свойств.
    /// Использует SVD разложение через MathNet.
    /// </summary>
    private void OrthogonalizeMapping()
    {
        _logger.LogDebug("Ортогонализация отображающей матрицы...");
        Utils.MathUtils.OrthogonalizeMatrix(_mappingMatrix);
    }

    /// <summary>
    /// Применение отображающей матрицы к эмбеддингам.
    /// Высокопроизводительная операция через матричное умножение.
    /// </summary>
    /// <param name="embeddings">Исходные эмбеддинги</param>
    /// <returns>Отображенные эмбеддинги</returns>
    private MatrixFloat ApplyMapping(MatrixFloat embeddings)
    {
        var result = new MatrixFloat(embeddings.Dimensions);
        Utils.MathUtils.MatrixMultiply(embeddings, _mappingMatrix, result);
        return result;
    }

    /// <summary>
    /// Создание случайного батча из матрицы эмбеддингов.
    /// </summary>
    /// <param name="embeddings">Матрица эмбеддингов</param>
    /// <param name="batchSize">Размер батча</param>
    /// <param name="random">Генератор случайных чисел</param>
    /// <returns>Батч эмбеддингов</returns>
    private MatrixFloat CreateBatch(MatrixFloat embeddings, int batchSize, Random random)
    {
        int vocabSize = embeddings.Dimensions[0];
        int embeddingDim = embeddings.Dimensions[1];
        var batch = new MatrixFloat(new[] { batchSize, embeddingDim });

        for (int i = 0; i < batchSize; i++)
        {
            int randomIndex = random.Next(vocabSize);
            for (int j = 0; j < embeddingDim; j++)
            {
                batch[i, j] = embeddings[randomIndex, j];
            }
        }

        return batch;
    }

    /// <summary>
    /// Вычисление binary cross-entropy loss.
    /// </summary>
    /// <param name="predictions">Предсказания модели</param>
    /// <param name="labels">Истинные метки</param>
    /// <returns>Значение loss функции</returns>
    private float ComputeBinaryCrossEntropyLoss(float[] predictions, float[] labels)
    {
        float loss = 0.0f;
        const float epsilon = 1e-7f; // Для численной стабильности

        for (int i = 0; i < predictions.Length; i++)
        {
            var p = Math.Clamp(predictions[i], epsilon, 1.0f - epsilon);
            loss -= labels[i] * MathF.Log(p) + (1 - labels[i]) * MathF.Log(1 - p);
        }

        return loss / predictions.Length;
    }

    // Дополнительные вспомогательные методы...

    /// <summary>
    /// Получение статистики обучения.
    /// </summary>
    /// <returns>Статистика loss функций и валидации</returns>
    public (List<float> mapLosses, List<float> discLosses, List<float> validationScores) GetTrainingStats()
    {
        return (_mapLosses, _discriminatorLosses, _validationScores);
    }
}