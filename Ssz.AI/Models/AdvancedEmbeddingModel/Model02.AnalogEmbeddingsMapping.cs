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
using MathNet.Numerics;
using MathNet.Numerics.Providers.LinearAlgebra;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Ssz.AI.Helpers;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Dictionary;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Evaluation;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Models;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Training;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Utils;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

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

    public int Initialize()
    {
        WordsHelper.InitializeWords_RU(LanguageInfo_RU, _loggersSet);
        var ruDictionary = new Dictionary(LanguageInfo_RU.Words.Select(w => w.Name).ToList(), "RU");
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
        var enDictionary = new Dictionary(LanguageInfo_EN.Words.Select(w => w.Name).ToList(), "EN");
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
            Parameters parameters = new Parameters();            

            // Валидация параметров
            if (!parameters.Validate())
            {
                logger.LogError("Ошибки в параметрах конфигурации");
                return 1;
            }

            // Установка seed для воспроизводимости
            if (parameters.Seed >= 0)
            {
                var random = new Random(parameters.Seed);
                logger.LogInformation($"Установлен seed: {parameters.Seed}");
            }

            // Построение модели
            logger.LogInformation("Построение модели MUSE...");
            var modelComponents = ModelBuilder.BuildModel(
                parameters,
                ruDictionary,
                ruEmb,
                enDictionary,
                enEmb,                
                withDiscriminator: true);

            // Создание тренера
            var trainer = new Trainer(
                sourceEmbeddings: modelComponents.SourceEmbeddings,
                targetEmbeddings: modelComponents.TargetEmbeddings,
                mappingMatrix: modelComponents.MappingMatrix,
                discriminator: modelComponents.Discriminator,
                sourceDictionary: modelComponents.SourceDictionary,
                targetDictionary: modelComponents.TargetDictionary,
                parameters: parameters
            );

            // Обучение
            logger.LogInformation("Начало обучения...");
            var startTime = DateTime.Now;
            trainer.Train();
            var trainingTime = DateTime.Now - startTime;

            logger.LogInformation($"Обучение завершено за {trainingTime.TotalMinutes:F2} минут");

            // Финальная оценка
            logger.LogInformation("Финальная оценка модели...");
            var evaluator = new WordTranslationEvaluator();
            var mappedSource = ApplyMapping(modelComponents.SourceEmbeddings, modelComponents.MappingMatrix);

            var finalAccuracy = evaluator.EvaluateAccuracy(
                mappedSource, modelComponents.TargetEmbeddings,
                modelComponents.SourceDictionary, modelComponents.TargetDictionary,
                parameters.MostFrequentValidation
            );

            logger.LogInformation($"Финальная точность: {finalAccuracy:F4}");

            // Подробная оценка P@K
            var precisionResults = evaluator.EvaluatePrecisionAtK(
                mappedSource, modelComponents.TargetEmbeddings,
                modelComponents.SourceDictionary, modelComponents.TargetDictionary,
                new[] { 1, 5, 10 },
                parameters.MostFrequentValidation
            );

            foreach (var (k, precision) in precisionResults)
            {
                logger.LogInformation($"P@{k}: {precision:F4}");
            }

            //// Экспорт результатов
            //if (parameters.ExportEmbeddings)
            //{
            //    await ExportResults(modelComponents, parameters, mappedSource, finalAccuracy);
            //}

            //// Сохранение статистики обучения
            //await SaveTrainingStatistics(trainer, parameters, finalAccuracy, trainingTime);

            logger.LogInformation("Программа успешно завершена");
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
    /// Применение отображающей матрицы к эмбеддингам.
    /// </summary>
    private static MatrixFloat_RowMajor ApplyMapping(MatrixFloat_RowMajor embeddings, MatrixFloat_RowMajor mapping)
    {
        var result = new MatrixFloat_RowMajor(embeddings.Dimensions);
        MathUtils.MatrixMultiply(embeddings, mapping, result);
        return result;
    }

    /// <summary>
    /// Экспорт результатов обучения.
    /// </summary>
    private static async Task ExportResults(ModelBuilder.ModelComponents modelComponents,
                                          Parameters parameters, MatrixFloat_RowMajor mappedSource,
                                          float accuracy)
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;
        logger.LogInformation("Экспорт результатов...");

        // Создание папки для результатов
        var outputDir = !string.IsNullOrEmpty(parameters.ExperimentPath)
            ? parameters.ExperimentPath
            : parameters.OutputPath;

        if (!Directory.Exists(outputDir))
        {
            Directory.CreateDirectory(outputDir);
        }

        // Экспорт отображенных исходных эмбеддингов
        var sourcePath = Path.Combine(outputDir, $"vectors-{parameters.SourceLanguage}.txt");
        EmbeddingLoader.ExportEmbeddings(mappedSource, modelComponents.SourceDictionary,
                                       sourcePath, parameters.ExportFormat);

        // Экспорт целевых эмбеддингов (для полноты)
        var targetPath = Path.Combine(outputDir, $"vectors-{parameters.TargetLanguage}.txt");
        EmbeddingLoader.ExportEmbeddings(modelComponents.TargetEmbeddings,
                                       modelComponents.TargetDictionary ?? modelComponents.SourceDictionary,
                                       targetPath, parameters.ExportFormat);

        // Экспорт отображающей матрицы
        var mappingPath = Path.Combine(outputDir, "mapping.txt");
        await File.WriteAllTextAsync(mappingPath, MatrixToString(modelComponents.MappingMatrix));

        logger.LogInformation($"Результаты экспортированы в {outputDir}");
    }

    /// <summary>
    /// Сохранение статистики обучения в JSON файл.
    /// </summary>
    private static async Task SaveTrainingStatistics(Trainer trainer, Parameters parameters,
                                                    float finalAccuracy, TimeSpan trainingTime)
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;
        logger.LogInformation("Сохранение статистики обучения...");

        var (mapLosses, discLosses, validationScores) = trainer.GetTrainingStats();

        var stats = new
        {
            parameters = new
            {
                source_language = parameters.SourceLanguage,
                target_language = parameters.TargetLanguage,
                embedding_dimension = parameters.EmbeddingDimension,
                max_vocabulary = parameters.MaxVocabulary,
                epochs = parameters.Epochs,
                map_learning_rate = parameters.MapLearningRate,
                discriminator_learning_rate = parameters.DiscriminatorLearningRate,
                seed = parameters.Seed
            },
            results = new
            {
                final_accuracy = finalAccuracy,
                training_time_minutes = trainingTime.TotalMinutes,
                map_losses = mapLosses,
                discriminator_losses = discLosses,
                validation_scores = validationScores
            },
            timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
        };

        var outputDir = !string.IsNullOrEmpty(parameters.ExperimentPath)
            ? parameters.ExperimentPath
            : parameters.OutputPath;

        if (!Directory.Exists(outputDir))
        {
            Directory.CreateDirectory(outputDir);
        }

        var statsPath = Path.Combine(outputDir, "training_stats.json");
        var json = JsonSerializer.Serialize(stats, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(statsPath, json);

        logger.LogInformation($"Статистика сохранена в {statsPath}");
    }

    /// <summary>
    /// Конвертация матрицы в строковое представление.
    /// </summary>
    private static string MatrixToString(MatrixFloat_RowMajor matrix)
    {
        var lines = new List<string>();
        int rows = matrix.Dimensions[0];
        int cols = matrix.Dimensions[1];

        for (int i = 0; i < rows; i++)
        {
            var values = new float[cols];
            for (int j = 0; j < cols; j++)
            {
                values[j] = matrix[i, j];
            }
            lines.Add(string.Join(" ", values.Select(v => v.ToString("G9"))));
        }

        return string.Join("\n", lines);
    }

    #endregion

    #region private fields

    private readonly ILoggersSet _loggersSet; 

    #endregion
}