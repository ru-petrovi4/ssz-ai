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
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Evaluation;
using Avalonia.Controls.Documents;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;

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

    public const string FileName_MUSE_Adversarial_RU_EN = "AdvancedEmbedding_MUSE_Adversarial_RU_EN.bin";

    public const string FileName_MUSE_Procrustes_RU_EN = "AdvancedEmbedding_MUSE_Procrustes_RU_EN.bin";

    public const string FileName_MUSE_Best_Mapping_RU_EN_Temp = "best_mapping.pt";

    public const string VALIDATION_METRIC = "mean_cosine-csls_knn_10-SourceToTarget-10000";

    /// <summary>
    ///     FastText         
    /// </summary>
    public readonly LanguageInfo LanguageInfo_RU = new();

    /// <summary>
    ///     FastText       
    /// </summary>
    public readonly LanguageInfo LanguageInfo_EN = new();        

    public async Task<int> ExecuteUnsupervisedTrainingAsync()
    {
        int wordsCount = 40000;
        WordsHelper.InitializeWords_RU(LanguageInfo_RU, wordsMaxCount: wordsCount, _loggersSet);
        var ruDictionary = GetDictionary(LanguageInfo_RU.Words.Take(wordsCount).Select(w => w.Name).ToList(), "ru");
        var d = WordsHelper.OldVectorLength_RU;
        var ruEmb = new MatrixFloat(wordsCount, d);
        for (int i = 0; i < wordsCount; i += 1)
        {
            var row = LanguageInfo_RU.Words[i];
            for (int j = 0; j < d; j += 1)
            {
                ruEmb[i, j] = row.OldVectorNormalized[j];
            }
        }

        WordsHelper.InitializeWords_EN(LanguageInfo_EN, wordsMaxCount: wordsCount, _loggersSet);
        var enDictionary = GetDictionary(LanguageInfo_EN.Words.Take(wordsCount).Select(w => w.Name).ToList(), "en");
        d = WordsHelper.OldVectorLength_EN;
        var enEmb = new MatrixFloat(wordsCount, d);
        for (int i = 0; i < wordsCount; i += 1)
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
                //if (parameters.Cuda) // Excessive
                //{
                //    cuda.manual_seed(parameters.Seed);
                //}
            }

            // Устройство для размещения тензоров
            var device = parameters.UseCuda ? CUDA : CPU;

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
                .reshape(sourceEmbeddingMatrix.RowsCount, sourceEmbeddingMatrix.ColumnsCount) // Row major
                .to(device);
            var targetTensor = tensor(targetEmbeddingMatrix.Data)
                .reshape(targetEmbeddingMatrix.RowsCount, targetEmbeddingMatrix.ColumnsCount) // Row major
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
            var mapping = new Mapping(parameters);
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
            
            using var trainer = new Trainer(
                sourceEmbeddings, targetEmbeddings, mapping, discriminator,
                sourceDictionary, targetDictionary, 
                parameters, 
                device,
                @"Data",
                //GetExperimentPath(parameters),
                logger);

            // Настраиваем оптимизаторы
            trainer.SetupOptimizers(parameters.MapOptimizerConfig, parameters.DisOptimizerConfig);

            bool runTraining = false;
            if (runTraining)
            {
                // Состязательное обучение
                if (parameters.Adversarial)
                {
                    await RunAdversarialTrainingAsync(trainer, parameters, logger);

                    var weightsToSave = trainer.Mapping.MappingLinear.weight.cpu();
                    weightsToSave.save(Path.Combine(@"Data", FileName_MUSE_Adversarial_RU_EN));
                }
            }

            bool runNRefinement = false;
            if (runNRefinement)
            {                
                // Procrustes refinement
                if (parameters.NRefinement > 0)
                {   
                    await RunProcrustesRefinementAsync(trainer, parameters, logger);

                    var weightsToSave = trainer.Mapping.MappingLinear.weight.cpu();
                    weightsToSave.save(Path.Combine(@"Data", FileName_MUSE_Procrustes_RU_EN));
                }
            }

            bool evaluateWordTranslation = true;
            if (evaluateWordTranslation)
            {
                using (var _ = no_grad())
                {
                    var loadedWeights = load(Path.Combine(@"Data", FileName_MUSE_Procrustes_RU_EN));
                    trainer.Mapping.MappingLinear.weight!.copy_(loadedWeights);
                }

                TrainingStats stats = new();
                var evaluator = new Evaluator(trainer, logger);
                await evaluator.EvaluateWordTranslationAsync(stats, Path.Combine("Data", "Words_RU_EN.csv"));
            }

            //// Экспорт финальных эмбеддингов
            //if (!string.IsNullOrEmpty(parameters.Export))
            //{
            //    await ExportFinalEmbeddingsAsync(trainer, parameters, logger);
            //}

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
        if (parameters.UseCuda && !cuda.is_available())
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
            wordsList.Select((w, i) => (w, i)).ToDictionary(it => it.w.ToLowerInvariant(), it => it.i),
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
    private static async Task RunAdversarialTrainingAsync(Trainer trainer,
        UnsupervisedParameters parameters, ILogger logger)
    {
        logger.LogSeparator("СОСТЯЗАТЕЛЬНОЕ ОБУЧЕНИЕ");

        var stats = new TrainingStats();
        var startTime = DateTime.UtcNow;

        var evaluator = new Evaluator(trainer, logger);

        for (int n_epoch = 0; n_epoch < parameters.NEpochs; n_epoch += 1)
        {
            logger.LogInformation($"Начало эпохи состязательного обучения {n_epoch}...");

            var epochStartTime = DateTime.UtcNow;
            long processedWords = 0;
            stats.DiscriminatorLosses.Clear();
            
            for (int iteration = 0; iteration < parameters.NIterationsInEpoch; iteration += parameters.BatchSize)                        
            {
                // Обучение дискриминатора
                for (int disStep = 0; disStep < parameters.DisSteps; disStep += 1)
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

            stats.ToLog.Clear();
            stats.ToLog["n_epoch"] = n_epoch;
            
            await evaluator.RunAllEvaluationsAsync(stats);
            await evaluator.EvaluateDiscriminatorAsync(stats);
            await trainer.SaveBestMappingWeightsAsync(stats, VALIDATION_METRIC);

            logger.LogInformation($"Конец эпохи {n_epoch}");

            // Обновление learning rate
            trainer.UpdateLearningRate(stats, "mean_cosine-csls_knn_10-SourceToTarget-10000",
                 parameters.LrDecay, parameters.LrShrink, parameters.MinLr);

            // Проверка минимального learning rate
            float currentLr = (float)trainer.MappingOptimizer!.ParamGroups.First().LearningRate;
            if (currentLr < parameters.MinLr)
            {
                logger.LogInformation("Learning rate < 1e-6. Прерывание обучения.");
                break;
            }
        }

        logger.LogInformation("Состязательное обучение завершено");

        //return Task.CompletedTask;
    }

    /// <summary>
    /// Запускает Procrustes refinement
    /// </summary>
    private static async Task RunProcrustesRefinementAsync(Trainer trainer,
        UnsupervisedParameters parameters, ILogger logger)
    {
        logger.LogSeparator("ИТЕРАТИВНОЕ PROCRUSTES REFINEMENT");

        // Загружаем лучшую модель
        await trainer.ReloadBestMappingWeightsAsync();
        var evaluator = new Evaluator(trainer, logger);
        var stats = new TrainingStats();

        for (int iteration = 0; iteration < parameters.NRefinement; iteration++)
        {
            logger.LogInformation($"Начало итерации refinement {iteration}...");

            await trainer.BuildDictionaryAsync(parameters);

            // Применение решения Прокруста
            trainer.ApplyProcrustesAlignment();

            await evaluator.RunAllEvaluationsAsync(stats);            
            await trainer.SaveBestMappingWeightsAsync(stats, VALIDATION_METRIC);            

            logger.LogInformation($"Конец итерации refinement {iteration}");
        }
    }

    /// <summary>
    /// Экспортирует финальные эмбеддинги
    /// </summary>
    private static async Task ExportFinalEmbeddingsAsync(Trainer trainer,
        UnsupervisedParameters parameters, ILogger logger)
    {
        logger.LogInformation("Экспорт финальных эмбеддингов...");

        // Загружаем лучшую модель
        await trainer.ReloadBestMappingWeightsAsync();

        // TODO: Реализовать экспорт эмбеддингов
        // trainer.Export();

        logger.LogInformation("Экспорт завершен");
    }

    #endregion

    #region private fields

    private readonly ILoggersSet _loggersSet;    

    #endregion    
}