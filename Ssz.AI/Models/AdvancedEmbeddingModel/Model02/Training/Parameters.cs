using Microsoft.Extensions.Logging;
using Ssz.Utils.Logging;
using System.Collections.Generic;
using System.Linq;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02.Training;

/// <summary>
/// Класс параметров конфигурации для обучения MUSE.
/// Содержит все настройки для загрузки данных, архитектуры модели и процесса обучения.
/// Соответствует argparse параметрам из оригинального Python кода.
/// </summary>
public class Parameters
{
    // Основные параметры эксперимента
    public int Seed { get; set; } = -1;
    public int Verbose { get; set; } = 2;
    public string ExperimentPath { get; set; } = "";
    public string ExperimentName { get; set; } = "debug";
    public string ExperimentId { get; set; } = "";
    public bool UseCuda { get; set; } = false; // В C# пока без CUDA
    public string ExportFormat { get; set; } = "txt";

    // Параметры данных
    public string SourceLanguage { get; set; } = "en";
    public string TargetLanguage { get; set; } = "es";
    public string SourceEmbeddingPath { get; set; } = "";
    public string TargetEmbeddingPath { get; set; } = "";
    public int EmbeddingDimension { get; set; } = 300;
    public int MaxVocabulary { get; set; } = 200000;

    // Параметры отображающей матрицы
    public bool MapIdentityInit { get; set; } = true;
    public float MapBeta { get; set; } = 0.001f;

    // Параметры дискриминатора
    public int DiscriminatorLayers { get; set; } = 2;
    public int DiscriminatorHiddenDim { get; set; } = 2048;
    public float DiscriminatorDropout { get; set; } = 0.0f;
    public float DiscriminatorInputDropout { get; set; } = 0.1f;
    public float DiscriminatorSmoothingFactor { get; set; } = 0.1f;
    public float DiscriminatorClipping { get; set; } = 0.0f;

    // Параметры обучения
    public int Epochs { get; set; } = 5;
    public int MapOptimizerSteps { get; set; } = 1000;
    public int DiscriminatorSteps { get; set; } = 5;
    public int MostFrequentValidation { get; set; } = 10000;

    // Параметры оптимизации отображения
    public string MapOptimizer { get; set; } = "sgd,lr=0.1";
    public float MapLearningRate { get; set; } = 0.1f;
    public float MapWeightDecay { get; set; } = 0.0f;

    // Параметры оптимизации дискриминатора
    public string DiscriminatorOptimizer { get; set; } = "sgd,lr=0.1";
    public float DiscriminatorLearningRate { get; set; } = 0.1f;
    public float DiscriminatorWeightDecay { get; set; } = 0.0f;

    // Параметры обучения словаря
    public string DictionaryMethod { get; set; } = "csls_knn_10";
    public float DictionaryBeta { get; set; } = 0.001f;
    public int DictionaryMaxRank { get; set; } = 15000;
    public int DictionaryMaxVocab { get; set; } = 200000;
    public int DictionaryTopK { get; set; } = 10;
    public float DictionaryThreshold { get; set; } = 0.0f;

    // Параметры валидации
    public int ValidationMetricStep { get; set; } = 1000;
    public string ValidationMetric { get; set; } = "mean_cosine-csls_knn_10-S2T-10000";

    // Параметры экспорта
    public bool ExportEmbeddings { get; set; } = true;
    public string OutputPath { get; set; } = "output/";

    /// <summary>
    /// Валидация параметров конфигурации.
    /// Проверяет корректность значений и их совместимость.
    /// </summary>
    /// <returns>True если все параметры корректны</returns>
    public bool Validate()
    {
        var errors = new List<string>();

        // Проверка обязательных путей
        if (string.IsNullOrEmpty(SourceEmbeddingPath))
            errors.Add("Не указан путь к исходным эмбеддингам");

        if (string.IsNullOrEmpty(TargetEmbeddingPath))
            errors.Add("Не указан путь к целевым эмбеддингам");

        // Проверка размерностей
        if (EmbeddingDimension <= 0)
            errors.Add("Размерность эмбеддингов должна быть положительной");

        if (MaxVocabulary <= 0)
            errors.Add("Максимальный размер словаря должен быть положительным");

        // Проверка параметров дискриминатора
        if (DiscriminatorLayers < 0)
            errors.Add("Количество слоев дискриминатора не может быть отрицательным");

        if (DiscriminatorHiddenDim <= 0)
            errors.Add("Размерность скрытых слоев дискриминатора должна быть положительной");

        // Проверка параметров обучения
        if (Epochs <= 0)
            errors.Add("Количество эпох должно быть положительным");

        if (MapLearningRate <= 0)
            errors.Add("Скорость обучения отображения должна быть положительной");

        if (DiscriminatorLearningRate <= 0)
            errors.Add("Скорость обучения дискриминатора должна быть положительной");

        if (errors.Any())
        {
            var logger = LoggersSet.Default.UserFriendlyLogger;
            foreach (var error in errors)
            {
                logger.LogError(error);
            }
            return false;
        }

        return true;
    }

    /// <summary>
    /// Создание параметров по умолчанию для быстрого тестирования.
    /// </summary>
    /// <returns>Объект с параметрами по умолчанию</returns>
    public static Parameters CreateDefault()
    {
        return new Parameters
        {
            ExperimentName = "muse_default",
            EmbeddingDimension = 300,
            MaxVocabulary = 50000, // Уменьшено для тестирования
            Epochs = 5,
            MapOptimizerSteps = 1000,
            DiscriminatorSteps = 5,
            ValidationMetricStep = 500
        };
    }
}