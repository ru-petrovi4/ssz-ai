using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Evaluation;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Models;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Training;
using static TorchSharp.torch;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;

/// <summary>
/// Параметры unsupervised обучения
/// </summary>
public sealed record UnsupervisedParameters : IDiscriminatorParameters, IMappingParameters, ITrainerParameters, IDictionaryBuilderParameters
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
    public bool UseCuda { get; init; } = true;
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
    public float MapBeta { get; init; } = 0.001f;

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
    public float DisDropout { get; init; } = 0.0f;
    /// <summary>
    /// Input dropout дискриминатора
    /// </summary>
    public float DisInputDropout { get; init; } = 0.1f;
    /// <summary>
    /// Количество шагов дискриминатора
    /// </summary>
    public int DisSteps { get; init; } = 5;
    /// <summary>
    /// Коэффициент потерь дискриминатора
    /// </summary>
    public float DisLambda { get; init; } = 1.0f;
    /// <summary>
    /// Количество наиболее частых слов для дискриминации
    /// </summary>
    public int DisMostFrequent { get; init; } = 75000;
    /// <summary>
    /// Сглаживание предсказаний дискриминатора
    /// </summary>
    public float DisSmooth { get; init; } = 0.1f;
    /// <summary>
    /// Обрезание весов дискриминатора"
    /// </summary>
    public float DisClipWeights { get; init; } = 0.0f;

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
    public string MapOptimizerConfig { get; init; } = "sgd&lr=0.1";
    /// <summary>
    /// Оптимизатор дискриминатора
    /// </summary>
    public string DisOptimizerConfig { get; init; } = "sgd&lr=0.1";
    /// <summary>
    /// Уменьшение learning rate (только SGD)
    /// </summary>
    public float LrDecay { get; init; } = 0.98f;
    /// <summary>
    /// Минимальный learning rate (только SGD)
    /// </summary>
    public float MinLr { get; init; } = 1e-6f;
    /// <summary>
    /// Сжатие learning rate при ухудшении метрики
    /// </summary>
    public float LrShrink { get; init; } = 0.5f;

    // Refinement parameters
    public int NRefinement { get; init; } = 1;

    // Dictionary parameters
    /// <summary>
    /// Путь к словарю для оценки
    /// </summary>
    public string DicoEval { get; init; } = "default";
    /// <summary>
    /// Метод построения словаря (nn, csls_knn_10, invsm_beta_30)
    /// </summary>
    public string DicoMethod { get; init; } = "csls_knn_10";
    /// <summary>
    /// Режим построения (SourceToTarget, TargetToSource, SourceToTarget|TargetToSource, SourceToTarget&TargetToSource)
    /// </summary>
    public string DicoBuild { get; init; } = "SourceToTarget";
    /// <summary>
    /// Порог уверенности для построения словаря
    /// </summary>
    public float DicoThreshold { get; init; } = 0.0f;
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
