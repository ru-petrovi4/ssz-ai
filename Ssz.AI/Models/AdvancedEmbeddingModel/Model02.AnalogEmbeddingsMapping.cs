using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics.Tensors;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Models;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Training;
using MathNet.Numerics;
using MathNet.Numerics.Providers.LinearAlgebra;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Ssz.AI.Helpers;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

/// <summary>
///     Нахождение маппинга 
/// </summary>
public partial class Model02
{
    #region construction and destruction

    public Model02()
    {
        _loggersSet = new LoggersSet(NullLogger.Instance, new UserFriendlyLogger((l, id, s) => DebugWindow.Instance.AddLine(s)));
    }            

    #endregion

    #region public functions

    public const int OldVectorLength = 300;

    /// <summary>
    ///     RusVectores        
    /// </summary>
    public readonly LanguageInfo LanguageInfo_RU = new();

    /// <summary>
    ///     GloVe (Stanford)        
    /// </summary>
    public readonly LanguageInfo LanguageInfo_EN = new();        

    public async Task<int> ExecuteUnsupervisedTrainingAsync()
    {
        WordsHelper.InitializeWords_RU(LanguageInfo_RU, _loggersSet);
        var ruDictionary = GetDictionary(LanguageInfo_RU.Words.Select(w => w.Name).ToList(), "RU");
        var d = WordsHelper.OldVectorLength_RU;
        var ruEmb = new MatrixFloat_RowMajor(LanguageInfo_RU.Words.Count, d);
        for (int i = 0; i < LanguageInfo_RU.Words.Count; i += 1)
        {
            var row = LanguageInfo_RU.Words[i];
            for (int j = 0; j < d; j += 1)
            {
                ruEmb[i, j] = row.OldVectorNormalized[j];
            }
        }

        WordsHelper.InitializeWords_EN(LanguageInfo_EN, _loggersSet);
        var enDictionary = GetDictionary(LanguageInfo_EN.Words.Select(w => w.Name).ToList(), "EN");
        d = WordsHelper.OldVectorLength_EN;
        var enEmb = new MatrixFloat_RowMajor(LanguageInfo_EN.Words.Count, d);
        for (int i = 0; i < LanguageInfo_EN.Words.Count; i += 1)
        {
            var row = LanguageInfo_EN.Words[i];
            for (int j = 0; j < d; j += 1)
            {
                enEmb[i, j] = row.OldVectorNormalized[j];
            }
        }

        // Инициализация логгера
        LoggersSet.Default = _loggersSet;
        var logger = _loggersSet.UserFriendlyLogger;
        logger.LogInformation("Запуск MUSE обучения...");

        try
        {
            // Парсинг аргументов командной строки
            UnsupervisedParameters parameters = new();

            // Валидация параметров
            ValidateParameters(parameters);

            // Устанавливаем seed если указан
            if (parameters.Seed >= 0)
            {
                manual_seed(parameters.Seed);
                if (parameters.Cuda)
                {
                    cuda.manual_seed(parameters.Seed);
                }
            }

            // Устройство для размещения тензоров
            var device = parameters.Cuda ? CUDA : CPU;

            //// Загружаем исходные эмбеддинги            
            //var (sourceDictionary, sourceEmbeddingMatrix) = await EmbeddingUtils.LoadTextEmbeddingsAsync(
            //    parameters.SrcEmb, parameters.MaxVocab, parameters.EmbDim, true, logger);
            var (sourceDictionary, sourceEmbeddingMatrix) = (ruDictionary, ruEmb);

            //// Загружаем целевые эмбеддинги
            //var (targetDictionary, targetEmbeddingMatrix) = await EmbeddingUtils.LoadTextEmbeddingsAsync(
            //    parameters.TgtEmb, parameters.MaxVocab, parameters.EmbDim, true, logger);
            var (targetDictionary, targetEmbeddingMatrix) = (enDictionary, enEmb);

            // Создаем тензоры эмбеддингов
            var sourceTensor = tensor(sourceEmbeddingMatrix.Data)
                .reshape(sourceEmbeddingMatrix.RowsCount, sourceEmbeddingMatrix.ColumnsCount)
                .to(device);
            var targetTensor = tensor(targetEmbeddingMatrix.Data)
                .reshape(targetEmbeddingMatrix.RowsCount, targetEmbeddingMatrix.ColumnsCount)
                .to(device);

            // Создаем embedding слои
            var sourceEmbeddings = nn.Embedding(sourceDictionary.Count, parameters.EmbDim, sparse: true);
            var targetEmbeddings = nn.Embedding(targetDictionary.Count, parameters.EmbDim, sparse: true);

            using (no_grad())
            {
                sourceEmbeddings.weight!.copy_(sourceTensor);
                targetEmbeddings.weight!.copy_(targetTensor);
            }

            sourceEmbeddings.to(device);
            targetEmbeddings.to(device);

            // Применяем нормализацию если указана
            //if (!string.IsNullOrEmpty(parameters.NormalizeEmbeddings))
            //{
            //    EmbeddingUtils.NormalizeEmbeddings(sourceEmbeddingMatrix, parameters.NormalizeEmbeddings);
            //    EmbeddingUtils.NormalizeEmbeddings(targetEmbeddingMatrix, parameters.NormalizeEmbeddings);

            //    // Обновляем веса после нормализации
            //    sourceEmbeddings.weight.copy_(tensor(sourceEmbeddingMatrix.Data)
            //        .reshape(sourceEmbeddingMatrix.RowsCount, sourceEmbeddingMatrix.ColumnsCount));
            //    targetEmbeddings.weight.copy_(tensor(targetEmbeddingMatrix.Data)
            //        .reshape(targetEmbeddingMatrix.RowsCount, targetEmbeddingMatrix.ColumnsCount));
            //}            

            // Создаем модели
            var mapping = new EmbeddingMapping(parameters);
            var discriminator = new Discriminator(parameters);

            // Перемещаем на устройство
            mapping = mapping.to(device);
            discriminator = discriminator.to(device);

            // Инициализируем веса
            mapping.InitializeWeights();
            discriminator.InitializeWeights();

            logger.LogInformation($"Созданы модели на устройстве {device.type}:");
            logger.LogInformation(mapping.GetModelInfo());
            logger.LogInformation(discriminator.GetArchitectureInfo());

            logger.LogInformation("Модели успешно созданы и инициализированы");
            
            using var trainer = new CrossLingualTrainer(
                sourceEmbeddings, targetEmbeddings, mapping, discriminator,
                sourceDictionary, targetDictionary, 
                parameters, 
                device,
                GetExperimentPath(parameters),
                logger);

            // Настраиваем оптимизаторы
            trainer.SetupOptimizers(parameters.MapOptimizer, parameters.DisOptimizer);           

            // Состязательное обучение
            if (parameters.Adversarial)
            {
                await RunAdversarialTrainingAsync(trainer, parameters, logger);
            }

            // Procrustes refinement
            if (parameters.NRefinement > 0)
            {
                await RunProcrustesRefinementAsync(trainer, parameters, logger);
            }

            // Экспорт финальных эмбеддингов
            if (!string.IsNullOrEmpty(parameters.Export))
            {
                await ExportFinalEmbeddingsAsync(trainer, parameters, logger);
            }

            logger.LogInformation("Обучение успешно завершено!");
            return 0;
        }
        catch (Exception ex)
        {   
            logger.LogError(ex, "Критическая ошибка");
            return 1;
        }        
    }

    #endregion

    #region private functions    

    /// <summary>
    /// Валидирует параметры обучения
    /// </summary>
    /// <param name="parameters">Параметры для валидации</param>
    /// <exception cref="ArgumentException">При некорректных параметрах</exception>
    private static void ValidateParameters(UnsupervisedParameters parameters)
    {
        // Проверяем CUDA доступность
        if (parameters.Cuda && !cuda.is_available())
            throw new ArgumentException("CUDA не доступна, но была запрошена");

        // Проверяем параметры dropout
        if (parameters.DisDropout < 0 || parameters.DisDropout >= 1)
            throw new ArgumentException("dis-dropout должен быть в диапазоне [0, 1)");

        if (parameters.DisInputDropout < 0 || parameters.DisInputDropout >= 1)
            throw new ArgumentException("dis-input-dropout должен быть в диапазоне [0, 1)");

        if (parameters.DisSmooth < 0 || parameters.DisSmooth >= 0.5)
            throw new ArgumentException("dis-smooth должен быть в диапазоне [0, 0.5)");

        // Проверяем параметры дискриминатора
        if (parameters.DisLambda <= 0 || parameters.DisSteps <= 0)
            throw new ArgumentException("dis-lambda и dis-steps должны быть положительными");

        // Проверяем learning rate параметры
        if (parameters.LrShrink <= 0 || parameters.LrShrink > 1)
            throw new ArgumentException("lr-shrink должен быть в диапазоне (0, 1]");

        //// Проверяем файлы эмбеддингов
        //if (string.IsNullOrEmpty(parameters.SrcEmb) || !File.Exists(parameters.SrcEmb))
        //    throw new ArgumentException($"Файл исходных эмбеддингов не найден: {parameters.SrcEmb}");

        //if (string.IsNullOrEmpty(parameters.TgtEmb) || !File.Exists(parameters.TgtEmb))
        //    throw new ArgumentException($"Файл целевых эмбеддингов не найден: {parameters.TgtEmb}");

        // Проверяем словарь для оценки
        if (parameters.DicoEval != "default" && !File.Exists(parameters.DicoEval))
            throw new ArgumentException($"Файл словаря для оценки не найден: {parameters.DicoEval}");

        // Проверяем формат экспорта
        if (!string.IsNullOrEmpty(parameters.Export) &&
            parameters.Export != "txt" && parameters.Export != "pth")
            throw new ArgumentException("Формат экспорта должен быть 'txt' или 'pth'");
    }

    private static Dictionary GetDictionary(List<string> wordsList, string language)
    {
        return new Dictionary(
            new SortedDictionary<int, string>(wordsList.Select((w, i) => (w, i)).ToDictionary(it => it.i, it => it.w)),
            wordsList.Select((w, i) => (w, i)).ToDictionary(it => it.w, it => it.i),
            language
            );
    }

    /// <summary>
    /// Получает путь для сохранения эксперимента
    /// </summary>
    /// <param name="parameters">Параметры эксперимента</param>
    /// <returns>Путь к директории эксперимента</returns>
    private static string GetExperimentPath(UnsupervisedParameters parameters)
    {
        var basePath = string.IsNullOrEmpty(parameters.ExpBasePath) ? "./dumped" : parameters.ExpBasePath;

        if (!Directory.Exists(basePath))
            Directory.CreateDirectory(basePath);

        var expFolder = Path.Combine(basePath, parameters.ExpName);
        if (!Directory.Exists(expFolder))
            Directory.CreateDirectory(expFolder);

        string expPath;
        if (string.IsNullOrEmpty(parameters.ExpId))
        {
            // Генерируем случайный ID
            var chars = "abcdefghijklmnopqrstuvwxyz0123456789";
            var random = new Random();
            string expId;
            do
            {
                expId = new string(Enumerable.Repeat(chars, 10)
                    .Select(s => s[random.Next(s.Length)]).ToArray());
                expPath = Path.Combine(expFolder, expId);
            } while (Directory.Exists(expPath));
        }
        else
        {
            expPath = Path.Combine(expFolder, parameters.ExpId);
            if (Directory.Exists(expPath))
                throw new ArgumentException($"Директория эксперимента уже существует: {expPath}");
        }

        Directory.CreateDirectory(expPath);
        return expPath;
    }    

    /// <summary>
    /// Запускает состязательное обучение
    /// </summary>
    private static Task RunAdversarialTrainingAsync(CrossLingualTrainer trainer,
        UnsupervisedParameters parameters, ILogger logger)
    {
        logger.LogSeparator("СОСТЯЗАТЕЛЬНОЕ ОБУЧЕНИЕ");

        var stats = new TrainingStats();
        var startTime = DateTime.UtcNow;

        for (int epoch = 0; epoch < parameters.NEpochs; epoch++)
        {
            logger.LogInformation($"Начало эпохи состязательного обучения {epoch}...");

            var epochStartTime = DateTime.UtcNow;
            long processedWords = 0;
            stats.DiscriminatorLosses.Clear();

            for (int iteration = 0; iteration < parameters.NIterationsInEpoch; iteration += parameters.BatchSize)
            {
                // Обучение дискриминатора
                for (int disStep = 0; disStep < parameters.DisSteps; disStep++)
                {
                    trainer.DiscriminatorStep(stats);
                }

                // Обучение маппинга (обман дискриминатора)
                processedWords += trainer.MappingStep(stats);

                // Логирование прогресса каждые 500 итераций
                if (iteration % 500 == 0)
                {
                    var elapsedTime = DateTime.UtcNow - epochStartTime;
                    var avgDiscLoss = stats.DiscriminatorLosses.Count > 0 ?
                        stats.DiscriminatorLosses.Average() : 0.0;
                    var rate = processedWords / elapsedTime.TotalSeconds;

                    logger.LogInformation($"{iteration:000000} - Потери дискриминатора: {avgDiscLoss:F4} - " +
                                        $"{rate:F0} образцов/с");

                    // Сброс статистик
                    epochStartTime = DateTime.UtcNow;
                    processedWords = 0;
                    stats.DiscriminatorLosses.Clear();
                }
            }

            // TODO: Добавить оценку и сохранение лучшей модели
            // evaluator.all_eval(to_log);
            // trainer.save_best(to_log, VALIDATION_METRIC);

            logger.LogInformation($"Конец эпохи {epoch}");

            // Обновление learning rate
            // trainer.UpdateLearningRate(validationMetric, ValidationMetric, 
            //     parameters.LrDecay, parameters.LrShrink, parameters.MinLr);

            // Проверка минимального learning rate
            // if (currentLr < parameters.MinLr)
            // {
            //     logger.LogInformation("Learning rate < 1e-6. Прерывание обучения.");
            //     break;
            // }
        }

        logger.LogInformation("Состязательное обучение завершено");

        return Task.CompletedTask;
    }

    /// <summary>
    /// Запускает Procrustes refinement
    /// </summary>
    private static async Task RunProcrustesRefinementAsync(CrossLingualTrainer trainer,
        UnsupervisedParameters parameters, ILogger logger)
    {
        logger.LogSeparator("ИТЕРАТИВНОЕ PROCRUSTES REFINEMENT");

        // Загружаем лучшую модель
        await trainer.ReloadBestModelAsync();

        for (int iteration = 0; iteration < parameters.NRefinement; iteration++)
        {
            logger.LogInformation($"Начало итерации refinement {iteration}...");

            // TODO: Построение словаря из выровненных эмбеддингов
            // trainer.BuildDictionary();

            // Применение решения Прокруста
            trainer.ApplyProcrustesAlignment();

            // TODO: Оценка эмбеддингов
            // evaluator.all_eval(to_log);
            // trainer.save_best(to_log, VALIDATION_METRIC);

            logger.LogInformation($"Конец итерации refinement {iteration}");
        }
    }

    /// <summary>
    /// Экспортирует финальные эмбеддинги
    /// </summary>
    private static async Task ExportFinalEmbeddingsAsync(CrossLingualTrainer trainer,
        UnsupervisedParameters parameters, ILogger logger)
    {
        logger.LogInformation("Экспорт финальных эмбеддингов...");

        // Загружаем лучшую модель
        await trainer.ReloadBestModelAsync();

        // TODO: Реализовать экспорт эмбеддингов
        // trainer.Export();

        logger.LogInformation("Экспорт завершен");
    }

    #endregion

    #region private fields

    private readonly ILoggersSet _loggersSet;

    /// <summary>
    /// Метрика валидации для unsupervised обучения
    /// </summary>
    private const string ValidationMetric = "mean_cosine-csls_knn_10-S2T-10000";

    #endregion

    /// <summary>
    /// Параметры unsupervised обучения
    /// </summary>
    public sealed record UnsupervisedParameters : IDiscriminatorParameters, IMappingParameters, ITrainerParameters
    {
        /// <summary>
        /// Seed для инициализации (-1 для случайного)
        /// </summary>
        public int Seed { get; init; } = 5;
        /// <summary>
        /// Уровень подробности (2:debug, 1:info, 0:warning)
        /// </summary>
        public int Verbose { get; init; }
        /// <summary>
        /// Базовый путь для сохранения экспериментов
        /// </summary>
        public string ExpBasePath { get; init; } = @"Data\Ssz.AI.AdvancedEmbedding\MUSE";
        /// <summary>
        /// Название эксперимента
        /// </summary>
        public string ExpName { get; init; } = "debug";
        /// <summary>
        /// ID эксперимента
        /// </summary>
        public string ExpId { get; init; } = "";
        /// <summary>
        /// Использовать GPU
        /// </summary>
        public bool Cuda { get; init; } = true;
        /// <summary>
        /// Формат экспорта эмбеддингов (txt / pth)
        /// </summary>
        public string Export { get; init; } = "txt";

        // Data parameters
        /// <summary>
        /// Исходный язык
        /// </summary>
        public string SrcLang { get; init; } = "ru";
        /// <summary>
        /// Целевой язык
        /// </summary>
        public string TgtLang { get; init; } = "en";
        /// <summary>
        /// Размерность эмбеддингов
        /// </summary>
        public int EmbDim { get; init; } = 300;
        /// <summary>
        /// Максимальный размер словаря (-1 для неограниченного)
        /// </summary>
        public int MaxVocab { get; init; } = 200000;

        // Mapping parameters
        /// <summary>
        /// Инициализировать маппинг как единичную матрицу
        /// </summary>
        public bool MapIdInit { get; init; } = true;
        /// <summary>
        /// Параметр бета для ортогонализации
        /// </summary>
        public double MapBeta { get; init; } = 0.001;

        // Discriminator parameters
        /// <summary>
        /// Количество скрытых слоев дискриминатора
        /// </summary>
        public int DisHidLayers { get; init; } = 2;
        /// <summary>
        /// Размерность скрытых слоев дискриминатора
        /// </summary>
        public int DisHidDim { get; init; } = 2048;
        /// <summary>
        /// Dropout для скрытых слоев дискриминатора
        /// </summary>
        public double DisDropout { get; init; } = 0.0;
        /// <summary>
        /// Input dropout дискриминатора
        /// </summary>
        public double DisInputDropout { get; init; } = 0.1;
        /// <summary>
        /// Количество шагов дискриминатора
        /// </summary>
        public int DisSteps { get; init; } = 5;
        /// <summary>
        /// Коэффициент потерь дискриминатора
        /// </summary>
        public double DisLambda { get; init; } = 1.0;
        /// <summary>
        /// Количество наиболее частых слов для дискриминации
        /// </summary>
        public int DisMostFrequent { get; init; } = 75000;
        /// <summary>
        /// Сглаживание предсказаний дискриминатора
        /// </summary>
        public double DisSmooth { get; init; } = 0.1;
        /// <summary>
        /// Обрезание весов дискриминатора"
        /// </summary>
        public double DisClipWeights { get; init; } = 0.0;

        // Training parameters
        /// <summary>
        /// Использовать состязательное обучение
        /// </summary>
        public bool Adversarial { get; init; } = true;
        /// <summary>
        /// Количество эпох
        /// </summary>
        public int NEpochs { get; init; } = 5;
        /// <summary>
        /// Итераций на эпоху
        /// </summary>
        public int NIterationsInEpoch { get; init; } = 1000000;
        /// <summary>
        /// Размер батча
        /// </summary>
        public int BatchSize { get; init; } = 32;
        /// <summary>
        /// Оптимизатор маппинга
        /// </summary>
        public string MapOptimizer { get; init; } = "sgd&lr=0.1";
        /// <summary>
        /// Оптимизатор дискриминатора
        /// </summary>
        public string DisOptimizer { get; init; } = "sgd&lr=0.1";
        /// <summary>
        /// Уменьшение learning rate (только SGD)
        /// </summary>
        public double LrDecay { get; init; } = 0.98;
        /// <summary>
        /// Минимальный learning rate (только SGD)
        /// </summary>
        public double MinLr { get; init; } = 1e-6;
        /// <summary>
        /// Сжатие learning rate при ухудшении метрики
        /// </summary>
        public double LrShrink { get; init; } = 0.5;

        // Refinement parameters
        public int NRefinement { get; init; }

        // Dictionary parameters
        /// <summary>
        /// Путь к словарю для оценки
        /// </summary>
        public string DicoEval { get; init; } = "default";
        /// <summary>
        /// Метод построения словаря
        /// </summary>
        public string DicoMethod { get; init; } = "csls_knn_10";
        /// <summary>
        /// "Режим построения словаря
        /// </summary>
        public string DicoBuild { get; init; } = "S2T";
        /// <summary>
        /// Порог уверенности для построения словаря
        /// </summary>
        public double DicoThreshold { get; init; } = 0.0;
        /// <summary>
        /// Максимальный ранг слов в словаре
        /// </summary>
        public int DicoMaxRank { get; init; } = 15000;
        /// <summary>
        /// Минимальный размер создаваемого словаря
        /// </summary>
        public int DicoMinSize { get; init; } = 0;
        /// <summary>
        /// Максимальный размер создаваемого словаря
        /// </summary>
        public int DicoMaxSize { get; init; } = 0;

        // Embeddings parameters
        /// <summary>
        /// Путь к исходным эмбеддингам
        /// </summary>
        public string SrcEmb { get; init; } = "";
        /// <summary>
        /// Путь к целевым эмбеддингам
        /// </summary>
        public string TgtEmb { get; init; } = "";
        /// <summary>
        /// Нормализация эмбеддингов перед обучением
        /// </summary>
        public string NormalizeEmbeddings { get; init; } = "";

        /// <summary>
        /// Использовать ли bias в линейном преобразовании
        /// </summary>
        public bool UseBias { get; init; } = false;

        public Device Device { get; } = CPU;

        /// <summary>
        /// Путь для сохранения экспериментов
        /// </summary>
        public string ExperimentPath { get; } = "./experiments";
    }
}