using Microsoft.Extensions.Logging;
using System;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;

/// <summary>
/// Класс для вычисления метрик качества кластеризации на единичной сфере
/// Реализует Silhouette Score и Davies-Bouldin Index с использованием косинусной меры расстояния
/// Оптимизирован для высокой производительности с векторизованными операциями TorchSharp
/// </summary>
public static class SphericalClusteringMetrics
{
    /// <summary>
    /// Вычисляет Silhouette Score для кластеризации на единичной сфере
    /// 
    /// Силуэт измеряет, насколько хорошо каждая точка принадлежит своему кластеру
    /// по сравнению с другими кластерами. Значения от -1 до 1:
    /// - близко к 1: точка хорошо кластеризована
    /// - около 0: точка находится на границе между кластерами  
    /// - близко к -1: точка, возможно, назначена неправильному кластеру
    /// 
    /// Формула для точки i:
    /// s(i) = (b(i) - a(i)) / max(a(i), b(i))
    /// 
    /// где:
    /// a(i) = среднее расстояние от точки i до всех других точек в том же кластере
    /// b(i) = минимальное среднее расстояние от точки i до точек в других кластерах
    /// </summary>
    /// <param name="data">Нормированные данные [N x D] на единичной сфере</param>
    /// <param name="labels">Метки кластеров [N] (целые числа от 0 до K-1)</param>    
    /// <returns>Средний силуэт-коэффициент для всех точек</returns>
    public static float ComputeSilhouetteScore(Tensor data, Tensor labels, ILogger logger)
    {
        ValidateInputs(data, labels, logger);

        var numPoints = data.shape[0];
        var numClusters = torch.max(labels).item<long>() + 1;
        
        // Предвычисляем матрицу попарных расстояний между всеми точками
        // Используем векторизованные операции для максимальной производительности
        var distanceMatrix = ComputeDistanceMatrix(data);
        
        // Создаём маску принадлежности к кластерам [N x K]
        // cluster_masks[i, k] = 1, если точка i принадлежит кластеру k, иначе 0
        var clusterMasks = CreateClusterMasks(labels, numClusters);
        
        var silhouetteScores = torch.zeros(numPoints);
        
        // Вычисляем силуэт-коэффициент для каждой точки
        for (long i = 0; i < numPoints; i += 1)
        {
            var pointLabel = labels[i].item<long>();
            
            // a(i) = среднее расстояние до точек своего кластера (исключая саму точку)
            var sameClusterMask = clusterMasks[.., pointLabel].clone();
            sameClusterMask[i] = 0; // Исключаем саму точку i
            
            var sameClusterCount = torch.sum(sameClusterMask).item<float>();
            float aScore = 0.0f;
            
            if (sameClusterCount > 0.0f)
            {
                // Векторизованное вычисление среднего расстояния до точек своего кластера
                var sameClusterDistances = distanceMatrix[i] * sameClusterMask;
                aScore = torch.sum(sameClusterDistances).item<float>() / sameClusterCount;
            }
            
            // b(i) = минимальное среднее расстояние до точек других кластеров
            float bScore = float.MaxValue;
            
            for (long k = 0; k < numClusters; k += 1)
            {
                if (k == pointLabel) continue; // Пропускаем свой кластер
                
                var otherClusterMask = clusterMasks[.., k];
                var otherClusterCount = torch.sum(otherClusterMask).item<float>();
                
                if (otherClusterCount > 0)
                {
                    // Векторизованное вычисление среднего расстояния до точек другого кластера
                    var otherClusterDistances = distanceMatrix[i] * otherClusterMask;
                    var avgDistance = torch.sum(otherClusterDistances).item<float>() / otherClusterCount;
                    bScore = MathF.Min(bScore, avgDistance);
                }
            }
            
            // Вычисляем силуэт-коэффициент для точки i
            if (bScore == float.MaxValue || (aScore == 0.0f && bScore == 0.0f))
            {
                // Особый случай: только один кластер или все расстояния равны 0
                silhouetteScores[i] = 0.0f;
            }
            else
            {
                // s(i) = (b(i) - a(i)) / max(a(i), b(i))
                var maxScore = MathF.Max(aScore, bScore);
                silhouetteScores[i] = (float)((bScore - aScore) / maxScore);
            }
        }
        
        // Возвращаем средний силуэт-коэффициент по всем точкам
        var averageSilhouette = torch.mean(silhouetteScores).item<float>();
        
        logger.LogInformation($"Silhouette Score computed: {averageSilhouette:F4}");
        logger.LogInformation($"Interpretation: {InterpretSilhouetteScore(averageSilhouette)}");
        
        return averageSilhouette;
    }

    /// <summary>
    /// Вычисляет Davies-Bouldin Index для кластеризации на единичной сфере
    /// 
    /// DB Index измеряет среднее отношение внутрикластерного разброса 
    /// к межкластерному расстоянию. Меньшие значения указывают на лучшую кластеризацию.
    /// 
    /// Формула:
    /// DB = (1/K) * Σ(k=1 to K) max(j≠k) [(σ_k + σ_j) / d(μ_k, μ_j)]
    /// 
    /// где:
    /// σ_k = средний радиус кластера k (среднее расстояние точек от центра)
    /// d(μ_k, μ_j) = расстояние между центрами кластеров k и j
    /// K = количество кластеров
    /// </summary>
    /// <param name="data">Нормированные данные [N x D] на единичной сфере</param>
    /// <param name="labels">Метки кластеров [N] (целые числа от 0 до K-1)</param>    
    /// <returns>Davies-Bouldin Index (чем меньше, тем лучше)</returns>
    public static float ComputeDaviesBouldinIndex(Tensor data, Tensor labels, ILogger logger)
    {
        ValidateInputs(data, labels, logger);
        
        var numClusters = torch.max(labels).item<long>() + 1;
        
        // Вычисляем центры кластеров (нормированные направления средних μ_k)
        var clusterCenters = ComputeClusterCenters(data, labels, numClusters);
        
        // Вычисляем внутрикластерные разбросы (σ_k для каждого кластера)
        var clusterScatters = ComputeClusterScatters(data, labels, clusterCenters, numClusters);
        
        // Вычисляем межкластерные расстояния между всеми парами центров
        var centerDistances = ComputeDistanceMatrix(clusterCenters);
        
        float totalDB = 0.0f;
        
        // Для каждого кластера k находим максимальное отношение с другими кластерами
        for (long k = 0; k < numClusters; k++)
        {
            float maxRatio = 0.0f;
            
            for (long j = 0; j < numClusters; j++)
            {
                if (k == j) continue; // Пропускаем сравнение кластера с самим собой
                
                var scatterSum = clusterScatters[k].item<float>() + clusterScatters[j].item<float>();
                var centerDistance = centerDistances[k, j].item<float>();
                
                // Избегаем деления на ноль в случае совпадающих центров
                if (centerDistance > 1e-10f)
                {
                    var ratio = scatterSum / centerDistance;
                    maxRatio = Math.Max(maxRatio, ratio);
                }
            }
            
            totalDB += maxRatio;
        }
        
        // DB = среднее по всем кластерам
        var dbIndex = totalDB / numClusters;
        
        logger.LogInformation($"Davies-Bouldin Index computed: {dbIndex:F4}");
        logger.LogInformation($"Interpretation: {InterpretDBIndex(dbIndex)}");
        
        return dbIndex;
    }

    /// <summary>
    /// Вычисляет матрицу попарных расстояний между точками на единичной сфере
    /// Использует косинусную меру
    /// </summary>
    /// <param name="data">Данные [N x D] или [K x D]</param>    
    /// <returns>Матрица расстояний [N x N] или [K x K]</returns>
    private static Tensor ComputeDistanceMatrix(Tensor data)
    {
        // Вычисляем матрицу косинусных сходств через матричное произведение
        // cosine_similarity(x_i, x_j) = x_i^T * x_j (поскольку данные нормированы)
        var cosineSimilarities = torch.matmul(data, data.transpose(-2, -1));
        
        // Ограничиваем значения для численной стабильности
        cosineSimilarities = torch.clamp(cosineSimilarities, -1.0f + 1e-7f, 1.0f - 1e-7f);

        // Косинусное расстояние: d(x_i, x_j) = 1 - x_i^T * x_j
        return 1.0f - cosineSimilarities;

        //if (useAngularDistance)
        //{
        //    // Угловое расстояние: d(x_i, x_j) = arccos(x_i^T * x_j) / π
        //    // Нормализуем на π, чтобы расстояние было в диапазоне [0, 1]
        //    return torch.acos(cosineSimilarities) / Math.PI;
        //}
    }

    /// <summary>
    /// Создаёт бинарные маски принадлежности точек к кластерам
    /// </summary>
    /// <param name="labels">Метки кластеров [N]</param>
    /// <param name="numClusters">Количество кластеров K</param>
    /// <returns>Маски [N x K], где masks[i,k] = 1 если точка i в кластере k</returns>
    private static Tensor CreateClusterMasks(Tensor labels, long numClusters)
    {
        var numPoints = labels.shape[0];
        var masks = torch.zeros(new long[] { numPoints, numClusters });
        
        for (long i = 0; i < numPoints; i++)
        {
            var clusterLabel = labels[i].item<long>();
            masks[i, clusterLabel] = 1.0f;
        }
        
        return masks;
    }

    /// <summary>
    /// Вычисляет центры кластеров как нормированные средние направления
    /// </summary>
    /// <param name="data">Данные [N x D]</param>
    /// <param name="labels">Метки кластеров [N]</param>
    /// <param name="numClusters">Количество кластеров K</param>
    /// <returns>Центры кластеров [K x D] (нормированные)</returns>
    private static Tensor ComputeClusterCenters(Tensor data, Tensor labels, long numClusters)
    {
        var dimension = data.shape[1];
        var centers = torch.zeros(new long[] { numClusters, dimension });
        
        for (long k = 0; k < numClusters; k++)
        {
            // Находим все точки, принадлежащие кластеру k
            var clusterMask = torch.eq(labels, k);
            var clusterPoints = torch.masked_select(data, clusterMask.unsqueeze(1)).view(-1, dimension);
            
            if (clusterPoints.shape[0] > 0)
            {
                // Вычисляем среднее направление (векторная сумма)
                var meanDirection = torch.mean(clusterPoints, dimensions: [ 0 ]);
                
                // Нормализуем на единичную сферу
                centers[k] = meanDirection / torch.norm(meanDirection);
            }
        }
        
        return centers;
    }

    /// <summary>
    /// Вычисляет внутрикластерные разбросы (средние расстояния от центра)
    /// </summary>
    /// <param name="data">Данные [N x D]</param>
    /// <param name="labels">Метки кластеров [N]</param>
    /// <param name="centers">Центры кластеров [K x D]</param>
    /// <param name="numClusters">Количество кластеров K</param>
    /// <param name="useAngularDistance">Использовать угловое расстояние</param>
    /// <returns>Разбросы кластеров [K]</returns>
    private static Tensor ComputeClusterScatters(Tensor data, Tensor labels, Tensor centers, 
                                                long numClusters)
    {
        var scatters = torch.zeros(numClusters);
        
        for (long k = 0; k < numClusters; k += 1)
        {
            // Находим все точки кластера k
            var clusterMask = torch.eq(labels, k);
            var clusterIndices = torch.nonzero(clusterMask).squeeze(-1);
            
            if (clusterIndices.shape[0] > 0)
            {
                var clusterPoints = data.index_select(0, clusterIndices);
                var center = centers[k];

                // Косинусные расстояния: 1 - center^T * points
                var cosineSims = torch.matmul(clusterPoints, center);
                var distances = 1.0f - cosineSims;
                scatters[k] = torch.mean(distances);

                //// Вычисляем расстояния от каждой точки до центра кластера
                //if (useAngularDistance)
                //{
                //    // Угловые расстояния: arccos(center^T * points) / π
                //    var cosineSims = torch.matmul(clusterPoints, center);
                //    cosineSims = torch.clamp(cosineSims, -1.0 + 1e-7, 1.0 - 1e-7);
                //    var distances = torch.acos(cosineSims) / Math.PI;
                //    scatters[k] = torch.mean(distances);
                //}
            }
        }
        
        return scatters;
    }

    /// <summary>
    /// Проверяет корректность входных данных
    /// </summary>
    /// <param name="data">Данные для проверки</param>
    /// <param name="labels">Метки для проверки</param>
    private static void ValidateInputs(Tensor data, Tensor labels, ILogger logger)
    {
        if (data.shape[0] != labels.shape[0])
        {
            throw new ArgumentException($"Количество точек в данных ({data.shape[0]}) не совпадает с количеством меток ({labels.shape[0]})");
        }
        
        if (data.shape[0] < 2)
        {
            throw new ArgumentException("Необходимо минимум 2 точки данных для вычисления метрик");
        }
        
        var numClusters = torch.max(labels).item<long>() + 1;
        if (numClusters < 2)
        {
            throw new ArgumentException("Необходимо минимум 2 кластера для вычисления метрик");
        }
        
        // Проверяем нормированность данных
        var norms = torch.norm(data, dimension: 1);
        var minNorm = torch.min(norms).item<float>();
        var maxNorm = torch.max(norms).item<float>();
        
        if (MathF.Abs(minNorm - 1.0f) > 1e-5f || MathF.Abs(maxNorm - 1.0f) > 1e-5f)
        {
            logger.LogInformation($"Предупреждение: данные могут быть не полностью нормированы. " +
                            $"Диапазон норм: [{minNorm:F6}, {maxNorm:F6}]");
        }
    }

    /// <summary>
    /// Интерпретирует значение Silhouette Score
    /// </summary>
    /// <param name="score">Значение силуэта</param>
    /// <returns>Текстовая интерпретация</returns>
    private static string InterpretSilhouetteScore(float score)
    {
        if (score >= 0.7f) return "Отличная структура кластеров";
        if (score >= 0.5f) return "Хорошая структура кластеров";
        if (score >= 0.3f) return "Умеренная структура кластеров";
        if (score >= 0.1f) return "Слабая структура кластеров";
        return "Очень слабая структура кластеров или неподходящее количество кластеров";
    }

    /// <summary>
    /// Интерпретирует значение Davies-Bouldin Index
    /// </summary>
    /// <param name="dbIndex">Значение DB индекса</param>
    /// <returns>Текстовая интерпретация</returns>
    private static string InterpretDBIndex(float dbIndex)
    {
        if (dbIndex <= 0.5f) return "Отличное разделение кластеров";
        if (dbIndex <= 1.0f) return "Хорошее разделение кластеров";
        if (dbIndex <= 1.5f) return "Удовлетворительное разделение кластеров";
        if (dbIndex <= 2.0f) return "Слабое разделение кластеров";
        return "Очень слабое разделение кластеров";
    }

    /// <summary>
    /// Вычисляет детальный отчёт по качеству кластеризации
    /// Включает обе метрики и дополнительную статистику
    /// </summary>
    /// <param name="data">Нормированные данные [N x D]</param>
    /// <param name="labels">Метки кластеров [N]</param>    
    public static void ComputeDetailedEvaluationReport(Tensor data, Tensor labels, ILogger logger)
    {
        logger.LogInformation("=== Детальный отчёт по качеству кластеризации ===");
        logger.LogInformation($"Количество точек: {data.shape[0]}");
        logger.LogInformation($"Размерность: {data.shape[1]}");
        
        var numClusters = torch.max(labels).item<long>() + 1;
        logger.LogInformation($"Количество кластеров: {numClusters}");
        
        // Статистика по размерам кластеров
        logger.LogInformation("\nСтатистика кластеров:");
        for (long k = 0; k < numClusters; k += 1)
        {
            var clusterSize = torch.sum(torch.eq(labels, k)).item<long>();
            var percentage = (float)clusterSize / data.shape[0] * 100;
            logger.LogInformation($"  Кластер {k}: {clusterSize} точек ({percentage:F1}%)");
        }
        
        var distanceType = "косинусное";
        logger.LogInformation($"\nТип расстояния: {distanceType}");
        
        // Вычисляем основные метрики
        logger.LogInformation("\n--- Основные метрики ---");
        var silhouetteScore = ComputeSilhouetteScore(data[..5000, ..], labels[..5000], logger);
        var dbIndex = ComputeDaviesBouldinIndex(data[..5000, ..], labels[..5000], logger);
        
        // Дополнительная статистика
        logger.LogInformation("\n--- Дополнительная статистика ---");
        ComputeInertiaStatistics(data, labels, logger);
        
        logger.LogInformation("\n=== Итоговая оценка ===");
        var overallQuality = EvaluateOverallQuality(silhouetteScore, dbIndex);
        logger.LogInformation($"Общее качество кластеризации: {overallQuality}");
    }

    /// <summary>
    /// Вычисляет статистику внутрикластерной инерции
    /// </summary>
    private static void ComputeInertiaStatistics(Tensor data, Tensor labels, ILogger logger)
    {
        var numClusters = torch.max(labels).item<long>() + 1;
        var centers = ComputeClusterCenters(data, labels, numClusters);
        
        float totalInertia = 0.0f;
        
        for (long k = 0; k < numClusters; k += 1)
        {
            var clusterMask = torch.eq(labels, k);
            var clusterIndices = torch.nonzero(clusterMask).squeeze(-1);
            
            if (clusterIndices.shape[0] > 0)
            {
                var clusterPoints = data.index_select(0, clusterIndices);
                var center = centers[k];
                
                // Вычисляем внутрикластерную инерцию
                float clusterInertia;

                var distances = 1.0f - torch.matmul(clusterPoints, center);
                clusterInertia = torch.sum(torch.pow(distances, 2)).item<float>();

                //if (useAngularDistance)
                //{
                //    var cosineSims = torch.matmul(clusterPoints, center);
                //    cosineSims = torch.clamp(cosineSims, -1.0 + 1e-7, 1.0 - 1e-7);
                //    var distances = torch.acos(cosineSims);
                //    clusterInertia = torch.sum(torch.pow(distances, 2)).item<float>();
                //}
                
                totalInertia += clusterInertia;
                logger.LogInformation($"  Инерция кластера {k}: {clusterInertia:F4}");
            }
        }
        
        logger.LogInformation($"Общая внутрикластерная инерция: {totalInertia:F4}");
    }

    /// <summary>
    /// Даёт общую оценку качества кластеризации на основе метрик
    /// </summary>
    private static string EvaluateOverallQuality(float silhouetteScore, float dbIndex)
    {
        var silhouetteGood = silhouetteScore >= 0.5f;
        var dbGood = dbIndex <= 1.0f;
        
        if (silhouetteGood && dbGood)
            return "Высокое качество - кластеры хорошо разделены и компактны";
        if (silhouetteGood || dbGood)
            return "Удовлетворительное качество - есть резерв для улучшения";
        return "Низкое качество - рекомендуется пересмотреть параметры кластеризации";
    }
}

///// <summary>
///// Демонстрационная программа использования метрик качества кластеризации
///// </summary>
//public class ClusterEvaluationDemo
//{
//    public static void Main()
//    {
//        logger.LogInformation("=== Демонстрация метрик качества vMF кластеризации ===");
        
//        // Устанавливаем случайное семя для воспроизводимости
//        torch.manual_seed(42);
        
//        // Генерируем тестовые данные
//        var (testData, trueLabels) = GenerateTestDataWithLabels();
        
//        // Симулируем результат кластеризации (в реальности получаем из vMF алгоритма)
//        var predictedLabels = SimulateClusteringResults(testData, trueLabels);
        
//        logger.LogInformation($"Данные: {testData.shape[0]} точек, размерность {testData.shape[1]}");
//        logger.LogInformation($"Истинных кластеров: {torch.max(trueLabels).item<long>() + 1}");
//        logger.LogInformation($"Найдено кластеров: {torch.max(predictedLabels).item<long>() + 1}");
        
//        // Вычисляем детальный отчёт
//        logger.LogInformation("\n=== Оценка с угловым расстоянием ===");
//        SphericalClusteringMetrics.ComputeDetailedEvaluationReport(testData, predictedLabels);
        
//        logger.LogInformation("\n=== Оценка с косинусным расстоянием ===");
//        SphericalClusteringMetrics.ComputeDetailedEvaluationReport(testData, predictedLabels);
        
//        // Сравнение с идеальной кластеризацией
//        logger.LogInformation("\n=== Сравнение с идеальными метками ===");
//        var idealSilhouette = SphericalClusteringMetrics.ComputeSilhouetteScore(testData, trueLabels);
//        var idealDB = SphericalClusteringMetrics.ComputeDaviesBouldinIndex(testData, trueLabels);
        
//        logger.LogInformation($"Идеальный Silhouette Score: {idealSilhouette:F4}");
//        logger.LogInformation($"Идеальный Davies-Bouldin Index: {idealDB:F4}");
//    }    
//}