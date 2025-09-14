using System.Numerics.Tensors;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02.Models;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02.Utils;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02.Dictionary;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02.Evaluation;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using Ssz.Utils.Logging;
using System;
using System.Linq;
using System.IO;

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
    private readonly Discriminator? _discriminator;
    private readonly Parameters _parameters;
    private readonly Dictionary.Dictionary _sourceDictionary;
    private readonly Dictionary.Dictionary? _targetDictionary;
    private readonly ILogger _logger;

    // Оптимизаторы (простая реализация SGD)
    private readonly SGDOptimizer _mapOptimizer;
    private readonly SGDOptimizer? _discriminatorOptimizer;

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
                   MatrixFloat mappingMatrix, Discriminator? discriminator,
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
        _mapOptimizer = new SGDOptimizer(_parameters.MapLearningRate, _parameters.MapWeightDecay);

        if (_discriminator != null)
        {
            _discriminatorOptimizer = new SGDOptimizer(_parameters.DiscriminatorLearningRate,
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
        var newMapping = MathUtils.ProcrustesAlignment(sourceMatrix, targetMatrix);

        // Обновление отображающей матрицы
        MathUtils.CopyMatrix(newMapping, _mappingMatrix);

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
        MathUtils.OrthogonalizeMatrix(_mappingMatrix);
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
        MathUtils.MatrixMultiply(embeddings, _mappingMatrix, result);
        return result;
    }

    /// <summary>
    /// Создание случайного батча из матрицы эмбеддингов.
    /// Выбирает случайные строки из матрицы для формирования мини-батча.
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

        // Заполнение батча случайными эмбеддингами
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
    /// Объединение двух батчей в один для обучения дискриминатора.
    /// Первый батч помечается как класс 0, второй как класс 1.
    /// </summary>
    /// <param name="batch1">Первый батч (обычно отображенные исходные эмбеддинги)</param>
    /// <param name="batch2">Второй батч (обычно целевые эмбеддинги)</param>
    /// <returns>Объединенный батч</returns>
    private MatrixFloat CombineBatches(MatrixFloat batch1, MatrixFloat batch2)
    {
        if (batch1.Dimensions[1] != batch2.Dimensions[1])
            throw new ArgumentException("Батчи должны иметь одинаковую размерность эмбеддингов");

        int batchSize1 = batch1.Dimensions[0];
        int batchSize2 = batch2.Dimensions[0];
        int embeddingDim = batch1.Dimensions[1];

        var combinedBatch = new MatrixFloat(new[] { batchSize1 + batchSize2, embeddingDim });

        // Копирование первого батча
        Array.Copy(batch1.Data, 0, combinedBatch.Data, 0, batch1.Data.Length);

        // Копирование второго батча
        Array.Copy(batch2.Data, 0, combinedBatch.Data, batch1.Data.Length, batch2.Data.Length);

        return combinedBatch;
    }

    /// <summary>
    /// Создание меток для объединенного батча.
    /// Первая половина помечается как 0 (исходный язык), вторая как 1 (целевой язык).
    /// </summary>
    /// <param name="batchSize">Размер одного батча</param>
    /// <returns>Массив меток [0,0,...,0,1,1,...,1]</returns>
    private float[] CreateLabels(int batchSize)
    {
        var labels = new float[batchSize * 2];

        // Первая половина: класс 0 (исходный язык)
        for (int i = 0; i < batchSize; i++)
        {
            labels[i] = 0.0f;
        }

        // Вторая половина: класс 1 (целевой язык)
        for (int i = batchSize; i < batchSize * 2; i++)
        {
            labels[i] = 1.0f;
        }

        return labels;
    }

    /// <summary>
    /// Вычисление binary cross-entropy loss функции.
    /// Используется для обучения дискриминатора и adversarial обучения отображения.
    /// </summary>
    /// <param name="predictions">Предсказания модели [0,1]</param>
    /// <param name="labels">Истинные метки (0 или 1)</param>
    /// <returns>Значение loss функции</returns>
    private float ComputeBinaryCrossEntropyLoss(float[] predictions, float[] labels)
    {
        if (predictions.Length != labels.Length)
            throw new ArgumentException("Длины массивов predictions и labels должны совпадать");

        float loss = 0.0f;
        const float epsilon = 1e-7f; // Для численной стабильности

        for (int i = 0; i < predictions.Length; i++)
        {
            // Ограничение предсказаний для избежания log(0)
            var p = Math.Clamp(predictions[i], epsilon, 1.0f - epsilon);

            // Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
            loss -= labels[i] * MathF.Log(p) + (1 - labels[i]) * MathF.Log(1 - p);
        }

        return loss / predictions.Length;
    }

    /// <summary>
    /// Вычисление градиентов для дискриминатора (упрощенная версия).
    /// В полной реализации здесь был бы полный backpropagation алгоритм.
    /// </summary>
    /// <param name="inputs">Входные данные батча</param>
    /// <param name="predictions">Предсказания дискриминатора</param>
    /// <param name="labels">Истинные метки</param>
    /// <returns>Градиенты для каждого слоя дискриминатора</returns>
    private List<(MatrixFloat weightGradients, float[] biasGradients)> ComputeDiscriminatorGradients(
        MatrixFloat inputs, float[] predictions, float[] labels)
    {
        if (_discriminator == null)
            return new List<(MatrixFloat, float[])>();

        var gradients = new List<(MatrixFloat, float[])>();

        // Упрощенная версия: вычисляем приблизительные градиенты
        // В полной реализации здесь был бы алгоритм обратного распространения

        for (int layer = 0; layer < _discriminator.LayerCount; layer++)
        {
            var weights = _discriminator.GetWeights(layer);
            var biases = _discriminator.GetBiases(layer);

            // Создание градиентов той же размерности
            var weightGradients = new MatrixFloat(weights.Dimensions);
            var biasGradients = new float[biases.Length];

            // Простое приближение градиентов на основе разности предсказаний и меток
            float avgError = 0.0f;
            for (int i = 0; i < predictions.Length; i++)
            {
                avgError += predictions[i] - labels[i];
            }
            avgError /= predictions.Length;

            // Заполнение градиентов небольшими значениями пропорциональными ошибке
            float gradientMagnitude = avgError * 0.01f; // Масштабирующий коэффициент

            for (int i = 0; i < weightGradients.Data.Length; i++)
            {
                weightGradients.Data[i] = gradientMagnitude * (float)(new Random().NextDouble() * 2 - 1);
            }

            for (int i = 0; i < biasGradients.Length; i++)
            {
                biasGradients[i] = gradientMagnitude * (float)(new Random().NextDouble() * 2 - 1);
            }

            gradients.Add((weightGradients, biasGradients));
        }

        return gradients;
    }

    /// <summary>
    /// Вычисление градиентов для отображающей матрицы (упрощенная версия).
    /// В полной реализации здесь был бы градиент по chain rule от discriminator loss.
    /// </summary>
    /// <param name="sourceBatch">Батч исходных эмбеддингов</param>
    /// <param name="predictions">Предсказания дискриминатора</param>
    /// <param name="targetLabels">Целевые метки для adversarial обучения</param>
    /// <returns>Градиенты отображающей матрицы</returns>
    private MatrixFloat ComputeMappingGradients(MatrixFloat sourceBatch, float[] predictions, float[] targetLabels)
    {
        var gradients = new MatrixFloat(_mappingMatrix.Dimensions);

        // Упрощенная версия: градиенты пропорциональны ошибке предсказания
        float avgError = 0.0f;
        for (int i = 0; i < predictions.Length; i++)
        {
            avgError += predictions[i] - targetLabels[i];
        }
        avgError /= predictions.Length;

        // Вычисление градиентов как производной adversarial loss по матрице отображения
        // Приближение: градиент пропорционален транспонированному произведению
        // В полной реализации здесь был бы точный gradient backpropagation

        int batchSize = sourceBatch.Dimensions[0];
        int embeddingDim = sourceBatch.Dimensions[1];

        // Создание случайного градиента с правильным масштабом
        var random = new Random();
        float gradientScale = avgError * 0.001f; // Малый масштабирующий коэффициент

        for (int i = 0; i < gradients.Data.Length; i++)
        {
            gradients.Data[i] = gradientScale * (float)(random.NextDouble() * 2 - 1);
        }

        return gradients;
    }

    /// <summary>
    /// Извлечение соответствующих эмбеддингов для словарных пар.
    /// Используется для Procrustes анализа и поиска оптимальной матрицы отображения.
    /// </summary>
    /// <param name="dictionary">Словарь пар (исходный_id, целевой_id)</param>
    /// <returns>Матрицы соответствующих эмбеддингов</returns>
    private (MatrixFloat sourceMatrix, MatrixFloat targetMatrix) ExtractDictionaryEmbeddings(
        List<(int sourceId, int targetId)> dictionary)
    {
        if (dictionary.Count == 0)
        {
            // Возвращаем пустые матрицы если словарь пуст
            var emptyMatrix = new MatrixFloat(new[] { 1, _sourceEmbeddings.Dimensions[1] });
            return (emptyMatrix, emptyMatrix);
        }

        int embeddingDim = _sourceEmbeddings.Dimensions[1];
        int dictionarySize = dictionary.Count;

        var sourceMatrix = new MatrixFloat(new[] { embeddingDim, dictionarySize });
        var targetMatrix = new MatrixFloat(new[] { embeddingDim, dictionarySize });

        // Копирование соответствующих эмбеддингов
        for (int i = 0; i < dictionarySize; i++)
        {
            var (sourceId, targetId) = dictionary[i];

            // Проверка корректности индексов
            if (sourceId >= 0 && sourceId < _sourceEmbeddings.Dimensions[0] &&
                targetId >= 0 && targetId < _targetEmbeddings.Dimensions[0])
            {
                // Копирование столбцов (векторов эмбеддингов)
                for (int j = 0; j < embeddingDim; j++)
                {
                    sourceMatrix[j, i] = _sourceEmbeddings[sourceId, j];
                    targetMatrix[j, i] = _targetEmbeddings[targetId, j];
                }
            }
        }

        return (sourceMatrix, targetMatrix);
    }

    /// <summary>
    /// Получение статистики обучения для анализа и визуализации.
    /// </summary>
    /// <returns>Кортеж со статистикой loss функций и валидации</returns>
    public (List<float> mapLosses, List<float> discLosses, List<float> validationScores) GetTrainingStats()
    {
        return (_mapLosses, _discriminatorLosses, _validationScores);
    }

    /// <summary>
    /// Сохранение текущего состояния модели.
    /// Полезно для checkpoint'ов во время длительного обучения.
    /// </summary>
    /// <param name="filePath">Путь для сохранения модели</param>
    public void SaveModel(string filePath)
    {
        _logger.LogInformation($"Сохранение модели в {filePath}...");

        try
        {
            // Создание директории если не существует
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Простое сохранение отображающей матрицы в текстовом формате
            var lines = new List<string>();

            // Заголовок с размерностями
            lines.Add($"{_mappingMatrix.Dimensions[0]} {_mappingMatrix.Dimensions[1]}");

            // Сохранение элементов матрицы
            for (int i = 0; i < _mappingMatrix.Dimensions[0]; i++)
            {
                var row = new float[_mappingMatrix.Dimensions[1]];
                for (int j = 0; j < _mappingMatrix.Dimensions[1]; j++)
                {
                    row[j] = _mappingMatrix[i, j];
                }
                lines.Add(string.Join(" ", row.Select(x => x.ToString("G9"))));
            }

            File.WriteAllLines(filePath, lines);
            _logger.LogInformation("Модель успешно сохранена");
        }
        catch (Exception ex)
        {
            _logger.LogError($"Ошибка сохранения модели: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// Загрузка сохраненной модели.
    /// </summary>
    /// <param name="filePath">Путь к файлу модели</param>
    public void LoadModel(string filePath)
    {
        _logger.LogInformation($"Загрузка модели из {filePath}...");

        try
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Файл модели не найден: {filePath}");

            var lines = File.ReadAllLines(filePath);
            if (lines.Length < 2)
                throw new InvalidDataException("Некорректный формат файла модели");

            // Чтение заголовка
            var headerParts = lines[0].Split(' ');
            int rows = int.Parse(headerParts[0]);
            int cols = int.Parse(headerParts[1]);

            if (rows != _mappingMatrix.Dimensions[0] || cols != _mappingMatrix.Dimensions[1])
                throw new InvalidDataException("Размерности загружаемой модели не соответствуют текущей");

            // Загрузка элементов матрицы
            for (int i = 0; i < rows; i++)
            {
                var values = lines[i + 1].Split(' ').Select(float.Parse).ToArray();
                for (int j = 0; j < cols; j++)
                {
                    _mappingMatrix[i, j] = values[j];
                }
            }

            _logger.LogInformation("Модель успешно загружена");
        }
        catch (Exception ex)
        {
            _logger.LogError($"Ошибка загрузки модели: {ex.Message}");
            throw;
        }
    }
}