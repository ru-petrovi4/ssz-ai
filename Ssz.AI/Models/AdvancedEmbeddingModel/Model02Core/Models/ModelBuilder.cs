using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Utils;
using Ssz.AI.Models;
using System;
using Ssz.Utils.Logging;
using Microsoft.Extensions.Logging;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Models;

/// <summary>
/// Строитель моделей для системы MUSE.
/// Загружает эмбеддинги, создает дискриминатор и инициализирует отображающую матрицу.
/// Оптимизирован для работы с большими словарями и высокой производительности.
/// </summary>
public static class ModelBuilder
{
    /// <summary>
    /// Результат построения модели, содержащий все необходимые компоненты.
    /// </summary>
    public class ModelComponents
    {
        public MatrixFloat SourceEmbeddings { get; set; } = null!;
        public MatrixFloat TargetEmbeddings { get; set; } = null!;
        public MatrixFloat MappingMatrix { get; set; } = null!;
        public Discriminator? Discriminator { get; set; }
        public Dictionary.Dictionary SourceDictionary { get; set; } = null!;
        public Dictionary.Dictionary? TargetDictionary { get; set; }
    }

    /// <summary>
    /// Построение всех компонентов модели MUSE.
    /// </summary>
    /// <param name="parameters">Параметры конфигурации</param>
    /// <param name="withDiscriminator">Флаг создания дискриминатора для adversarial обучения</param>
    /// <returns>Компоненты модели</returns>
    public static ModelComponents BuildModel(Training.Parameters parameters, bool withDiscriminator = true)
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;
        logger.LogInformation("Начало построения модели MUSE...");

        // Загрузка исходных эмбеддингов
        logger.LogInformation($"Загрузка исходных эмбеддингов: {parameters.SourceLanguage}");
        var (srcDict, srcEmb) = Utils.EmbeddingLoader.LoadEmbeddings(
            parameters.SourceEmbeddingPath,
            parameters.MaxVocabulary,
            parameters.EmbeddingDimension,
            parameters.SourceLanguage);

        // Загрузка целевых эмбеддингов
        logger.LogInformation($"Загрузка целевых эмбеддингов: {parameters.TargetLanguage}");
        var (tgtDict, tgtEmb) = Utils.EmbeddingLoader.LoadEmbeddings(
            parameters.TargetEmbeddingPath,
            parameters.MaxVocabulary,
            parameters.EmbeddingDimension,
            parameters.TargetLanguage);

        // Нормализация эмбеддингов для стабильности обучения
        logger.LogInformation("Нормализация эмбеддингов...");
        Utils.MathUtils.NormalizeEmbeddings(srcEmb);
        Utils.MathUtils.NormalizeEmbeddings(tgtEmb);

        // Инициализация отображающей матрицы
        logger.LogInformation("Инициализация отображающей матрицы...");
        var mappingMatrix = InitializeMappingMatrix(parameters.EmbeddingDimension, parameters.MapIdentityInit);

        // Создание дискриминатора при необходимости
        Discriminator? discriminator = null;
        if (withDiscriminator)
        {
            logger.LogInformation("Создание дискриминатора...");
            discriminator = new Discriminator(
                embeddingDim: parameters.EmbeddingDimension,
                hiddenDim: parameters.DiscriminatorHiddenDim,
                numLayers: parameters.DiscriminatorLayers,
                dropoutRate: parameters.DiscriminatorDropout,
                inputDropoutRate: parameters.DiscriminatorInputDropout);
        }

        logger.LogInformation("Модель успешно построена.");

        return new ModelComponents
        {
            SourceEmbeddings = srcEmb,
            TargetEmbeddings = tgtEmb,
            MappingMatrix = mappingMatrix,
            Discriminator = discriminator,
            SourceDictionary = srcDict,
            TargetDictionary = tgtDict
        };
    }

    /// <summary>
    /// Инициализация отображающей матрицы.
    /// Может быть единичной матрицей или случайной ортогональной матрицей.
    /// </summary>
    /// <param name="dimension">Размерность матрицы</param>
    /// <param name="identityInit">Флаг инициализации единичной матрицей</param>
    /// <returns>Инициализированная отображающая матрица</returns>
    private static MatrixFloat InitializeMappingMatrix(int dimension, bool identityInit)
    {
        var mapping = new MatrixFloat(new[] { dimension, dimension });

        if (identityInit)
        {
            // Инициализация единичной матрицей для стабильного старта
            for (int i = 0; i < dimension; i++)
            {
                for (int j = 0; j < dimension; j++)
                {
                    mapping[i, j] = i == j ? 1.0f : 0.0f;
                }
            }
        }
        else
        {
            // Инициализация случайной ортогональной матрицей
            var random = new Random();

            // Заполнение случайными значениями
            for (int i = 0; i < dimension; i++)
            {
                for (int j = 0; j < dimension; j++)
                {
                    mapping[i, j] = (float)(random.NextDouble() * 2.0 - 1.0) * 0.1f;
                }
            }

            // Ортогонализация методом QR-разложения через MathNet
            Utils.MathUtils.OrthogonalizeMatrix(mapping);
        }

        return mapping;
    }
}