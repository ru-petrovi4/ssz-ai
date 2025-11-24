using MathNet.Numerics;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;
using Ssz.Utils.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;

/// <summary>
/// Реализация кластеризации von Mises-Fisher (vMF) на единичной сфере
/// Использует EM-алгоритм для оценки параметров смеси vMF распределений
/// </summary>
public class VonMisesFisherClusterer
{
    private readonly Device _device;
    private readonly ILogger _logger;

    // Поля класса для хранения параметров модели
    /// <summary>
    /// Количество кластеров K
    /// </summary>
    private readonly int _numClusters;

    /// <summary>
    /// Максимальное количество итераций EM
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Порог сходимости по логарифмической вероятности
    /// </summary>
    private readonly float _tolerance;

    private Tensor _oldVectorsTensor_device = null!;

    private Tensor _oldVectorsTensor = null!;

    private long _numSamples;

    private long _dimension;

    // Параметры модели - изучаются в процессе обучения
    /// <summary>
    /// μ_k - направления средних для каждого кластера [K x D]
    /// </summary>
    /// <remarks>device: CPU</remarks>
    public Tensor MeanDirections { get; private set; } = null!;    

    /// <summary>
    /// κ_k - параметры концентрации для каждого кластера [K]
    /// </summary>
    /// <remarks>device: CPU</remarks>
    public Tensor Concentrations { get; private set; } = null!;

    /// <summary>
    /// α_k - коэффициенты смешивания [K]
    /// </summary>
    /// <remarks>device: CPU</remarks>
    public Tensor MixingCoefficients { get; private set; } = null!;

    // Логи процесса обучения
    public List<float> LogLikelihoodHistory { get; private set; }

    /// <summary>
    /// Конструктор кластеризатора vMF
    /// </summary>
    /// <param name="device"></param>
    /// <param name="logger"></param>
    /// <param name="numClusters">Количество кластеров K</param>
    /// <param name="maxIterations">Максимальное количество итераций EM алгоритма</param>
    /// <param name="tolerance">Порог сходимости для логарифмической вероятности</param>    
    public VonMisesFisherClusterer(
        Device device,
        ILogger logger,
        int numClusters, 
        int maxIterations, 
        float tolerance)
    {
        _device = device;
        _logger = logger;
        _numClusters = numClusters;
        _maxIterations = maxIterations;
        _tolerance = tolerance;        
        LogLikelihoodHistory = new List<float>();
    }

    /// <summary>
    /// Основной метод обучения кластеризатора на нормированных данных
    /// </summary>
    /// <param name="oldVectorsTensor">Нормированные данные [N x D], где N - количество точек, D - размерность. Device: CPU</param>
    public void Fit(Tensor oldVectorsTensor)
    {
        // Проверяем, что данные корректно нормированы
        ValidateNormalizedData(oldVectorsTensor);
            
        // Инициализируем параметры модели
        InitializeParameters(oldVectorsTensor);
            
        float prevLogLikelihood = float.NegativeInfinity;

        // Основной цикл EM алгоритма        
        for (int iteration = 0; iteration < _maxIterations; iteration += 1)
        {
            // E-шаг: вычисляем posterior probabilities p(k|x_i)
            using var posteriors_device = ComputePosteriors_device();

            // M-шаг: обновляем параметры модели
            UpdateParameters(posteriors_device);

            // Вычисляем логарифмическую вероятность для проверки сходимости
            var logLikelihood = ComputeLogLikelihood(_oldVectorsTensor[..1000, ..]);
            LogLikelihoodHistory.Add(logLikelihood);

            // Проверяем сходимость
            if (Math.Abs(logLikelihood - prevLogLikelihood) < _tolerance)
            {
                _logger.LogInformation($"Сходимость достигнута на итерации {iteration + 1}");
                break;
            }

            prevLogLikelihood = logLikelihood;

            // Выводим прогресс каждые 10 итераций
            //if ((iteration + 1) % 10 == 0)
            {
                _logger.LogInformation($"Итерация {iteration + 1}: Log-Likelihood = {logLikelihood:F6}");
            }
        }

        _logger.LogInformation($"Обучение завершено. Финальная Log-Likelihood: {prevLogLikelihood:F6}");
    }

    /// <summary>
    /// 
    /// </summary>    
    /// <param name="clusterization_AlgorithmData"></param>
    public void GetResult(
        List<Word> words,
        Clusterization_AlgorithmData clusterization_AlgorithmData
        )
    { 
        // Вычисляем логарифмические вероятности для численной стабильности
        using var logProbabilities = torch.zeros(size: new long[] { _numSamples, _numClusters });

        for (int k = 0; k < _numClusters; k++)
        {
            // Вычисляем cosine similarity между данными и k-м центром
            var cosineSimilarities = torch.matmul(_oldVectorsTensor, MeanDirections[k, ..].t());

            // Вычисляем логарифм нормализующей константы c_d(κ)
            var logNormalizingConstant = ComputeLogNormalizingConstant(
                Concentrations[k].item<float>(),
                _dimension
            );

            // Логарифм vMF плотности: log c_d(κ) + κ * μ^T * x
            logProbabilities[.., k] = logNormalizingConstant +
                Concentrations[k].item<float>() * cosineSimilarities +
                torch.log(MixingCoefficients[k]);

            var clusterInfo = clusterization_AlgorithmData.ClusterInfos[k];
            MeanDirections.select(dim: 0, index: k).data<float>().CopyTo(clusterInfo.CentroidOldVectorNormalized);
            clusterInfo.MixingCoefficient = MixingCoefficients[k].item<float>();
            clusterInfo.Concentration = Concentrations[k].item<float>();
        }

        // Жёсткое назначение: назначаем каждую точку кластеру с максимальной вероятностью
        var assignments = torch.argmax(logProbabilities, dim: 1);
        for (int i = 0; i < _numSamples; i++)
        {
            var assignedCluster = assignments[i].item<long>();            
            words[i].ClusterIndex = (int)assignedCluster;
        }

        clusterization_AlgorithmData.MeanDirections = MeanDirections;
        clusterization_AlgorithmData.Concentrations = Concentrations;
        clusterization_AlgorithmData.MixingCoefficients = MixingCoefficients;
    }

    /// <summary>
    /// Вычисляет логарифм нормализующей константы c_d(κ) для vMF распределения
    /// Использует аппроксимацию для высоких размерностей чтобы избежать переполнения
    /// </summary>
    /// <param name="concentration">Параметр концентрации κ</param>
    /// <param name="dimension">Размерность d</param>
    /// <returns>Логарифм нормализующей константы</returns>
    public static float ComputeLogNormalizingConstant(float concentration, long dimension)
    {
        double d = dimension;
        double nu = d / 2.0 - 1.0;  // Порядок Бесселя ν = d/2 - 1
        double halfD = d / 2.0;     // d/2

        if (concentration < 1e-8)  // Порог для малого κ, где I_ν(κ) ≈ 0 из-за underflow
        {
            // Правильный лимит для κ → 0: log c_d(0) = log Γ(d/2) - log 2 - (d/2) log π
            // Площадь единичной сферы S^{d-1}: 2 π^{d/2} / Γ(d/2), так что c(0) = Γ(d/2) / (2 π^{d/2})
            return (float)(MathNet.Numerics.SpecialFunctions.GammaLn(halfD) - Math.Log(2.0) - halfD * Math.Log(Math.PI));
        }

        // Общие члены: - (d/2) log(2π) = - (ν + 1) log(2π)
        double log2piHalfD = halfD * Math.Log(2.0 * Math.PI);
        // (d/2 - 1) log κ = ν log κ
        double nuLogKappa = nu * Math.Log(concentration);
        double logI;

        // Вычисляем модифицированную функцию Бесселя I_ν(κ)
        double besselI = MathNet.Numerics.SpecialFunctions.BesselI(nu, concentration);

        if (double.IsPositiveInfinity(besselI) || besselI == 0.0)
        {
            // Асимптотическая аппроксимация для большого κ: log I_ν(κ) ≈ κ - 0.5 log(2 π κ) - (4ν² - 1)/(8 κ)
            // Это log[ exp(κ) / sqrt(2 π κ) * (1 - (4ν² - 1)/(8 κ) + ...) ] ≈ κ - 0.5 log(2 π κ) - (4ν² - 1)/(8 κ)
            double logTerm = 0.5 * Math.Log(2.0 * Math.PI * concentration);
            double correction = (4.0 * nu * nu - 1.0) / (8.0 * concentration);
            logI = concentration - logTerm - correction;
        }
        else
        {
            logI = Math.Log(besselI);
        }

        // Полная формула: log c = ν log κ - (ν + 1) log(2π) - log I_ν(κ)
        double logC = nuLogKappa - log2piHalfD - logI;
        return (float)logC;
    }

    /// <summary>
    /// Проверяет, что входные данные правильно нормированы (лежат на единичной сфере)
    /// </summary>
    /// <param name="data">Данные для проверки</param>
    private void ValidateNormalizedData(Tensor data)
    {
        // Вычисляем нормы всех векторов
        var norms = torch.norm(data, dimension: 1);
        var minNorm = torch.min(norms).item<float>();
        var maxNorm = torch.max(norms).item<float>();
            
        // Проверяем, что все нормы приблизительно равны 1.0
        if (Math.Abs(minNorm - 1.0) > 1e-6 || Math.Abs(maxNorm - 1.0) > 1e-6)
        {
            throw new ArgumentException($"Данные должны быть нормированы на единичную сферу. " +
                $"Найдены нормы в диапазоне [{minNorm:F8}, {maxNorm:F8}]");
        }
    }

    /// <summary>
    /// Инициализирует параметры модели перед началом EM алгоритма
    /// Использует kmeans++ подобную инициализацию для выбора начальных центров
    /// </summary>
    /// <param name="oldVectorsTensor">Входные данные</param>    
    private void InitializeParameters(Tensor oldVectorsTensor)
    {
        _numSamples = oldVectorsTensor.shape[0];
        _dimension = oldVectorsTensor.shape[1];

        _oldVectorsTensor = oldVectorsTensor;
        _oldVectorsTensor_device = oldVectorsTensor.to(_device);

        var (numSamples, dimension) = (oldVectorsTensor.shape[0], oldVectorsTensor.shape[1]);

        using (var disposeScope = torch.NewDisposeScope())
        {
            // Инициализация направлений средних с помощью kmeans++ подобного метода
            var meanDirections_device = torch.zeros(new long[] { _numClusters, dimension }, dtype: torch.float32, device: _device);

            // Выбираем первый центр случайно
            var randomIndex = torch.randint(low: 0, high: numSamples, size: new long[] { 1 }).item<long>();
            meanDirections_device[0] = _oldVectorsTensor_device[randomIndex];

            // Выбираем остальные центры с вероятностью пропорциональной расстоянию до ближайшего центра
            for (int k = 1; k < _numClusters; k += 1)
            {
                // Для каждой точки данных находим расстояние до ближайшего уже выбранного центра

                var similarity = torch.mm(_oldVectorsTensor_device, meanDirections_device[..k, ..].t());
                var (probabilities, indices) = torch.topk(similarity, k: 1, dim: 1, largest: true);
                probabilities.neg_().add_(1.0f).pow_(2);

                // UNOPTIMIZED
                //var distances = torch.zeros(numSamples);
                //for (long i = 0; i < numSamples; i += 1)
                //{
                //    var minDistance = float.MaxValue;

                //    // Вычисляем минимальное расстояние до уже выбранных центров
                //    for (int j = 0; j < k; j += 1)
                //    {
                //        // Используем угловое расстояние: 1 - cosine_similarity
                //        var cosineSimilarity = torch.dot(oldVectorsTensor[i], MeanDirections[j]).item<float>();
                //        var angularDistance = 1.0 - cosineSimilarity;
                //        minDistance = Math.Min(minDistance, angularDistance);
                //    }

                //    distances[i] = minDistance;
                //}

                //// Выбираем следующий центр с вероятностью пропорциональной квадрату расстояния
                //var probabilities = torch.pow(distances, 2);
                
                probabilities = probabilities / torch.sum(probabilities);

                // Используем multinomial sampling для выбора индекса
                var selectedIndex = torch.multinomial(input: probabilities.t(), num_samples: 1).item<long>();
                meanDirections_device[k] = _oldVectorsTensor_device[selectedIndex];

                if ((k + 1) % 10 == 0)
                {
                    _logger.LogInformation($"Инициализирован кластер {k}/{_numClusters}.");
                }
            }

            MeanDirections = meanDirections_device.to(CPU).DetachFromDisposeScope();
        }            

        // Инициализация параметров концентрации
        // Начинаем с умеренных значений концентрации
        Concentrations = torch.ones(size: _numClusters, dtype: torch.float32) * 1.0f;

        // Инициализация коэффициентов смешивания (равномерное распределение)
        MixingCoefficients = torch.ones(_numClusters, dtype: torch.float32) / _numClusters;

        _logger.LogInformation("Параметры инициализированы:");
        _logger.LogInformation($"Количество кластеров: {_numClusters}");
        _logger.LogInformation($"Размерность данных: {dimension}");
        _logger.LogInformation($"Количество образцов: {numSamples}");
    }

    /// <summary>
    /// E-шаг EM алгоритма: вычисляет posterior probabilities p(k|x_i)
    /// </summary>
    /// <param name="oldVectorsTensor">Входные данные [N x D]</param>
    /// <returns>Матрица posterior probabilities [N x K]</returns>
    private Tensor ComputePosteriors_device()
    {        
        Tensor posteriors_device; // = torch.zeros(new long[] { numSamples, _numClusters });

        using (var disposeScope = torch.NewDisposeScope())
        {
            // Вычисляем логарифмические вероятности для численной стабильности
            using var logProbabilities_device = torch.zeros(size: new long[] { _numSamples, _numClusters }, device: _device);

            // [N x K]
            var meanDirections_device = MeanDirections.to(_device);           
            var concentrations_device = Concentrations.to(_device);
            var mixingCoefficients_device = MixingCoefficients.to(_device);

            var cosineSimilarities_device = torch.mm(_oldVectorsTensor_device, meanDirections_device.transpose(dim0: 0, dim1: 1));

            for (int k = 0; k < _numClusters; k += 1)
            {
                // Вычисляем логарифм нормализующей константы c_d(κ)
                var logNormalizingConstant = ComputeLogNormalizingConstant(
                    Concentrations[k].item<float>(),
                    _dimension
                );

                // Логарифм vMF плотности: log c_d(κ) + κ * μ^T * x
                logProbabilities_device[.., k] = logNormalizingConstant +
                    concentrations_device[k] * cosineSimilarities_device[.., k] +
                    torch.log(mixingCoefficients_device[k]);
            }

            //if (_useHardAssignment)
            //{
            //    // Жёсткое назначение: назначаем каждую точку кластеру с максимальной вероятностью
            //    var assignments = torch.argmax(logProbabilities, dim: 1);
            //    for (long i = 0; i < numSamples; i += 1)
            //    {
            //        var assignedCluster = assignments[i].item<long>();
            //        posteriors[i, assignedCluster] = 1.0f;
            //    }
            //}
            //else
            //{            
            //}

            // Мягкое назначение: используем softmax для получения вероятностей
            posteriors_device = torch.softmax(logProbabilities_device, dim: 1).DetachFromDisposeScope();
        }

        return posteriors_device;
    }

    /// <summary>
    /// M-шаг EM алгоритма: обновляет параметры модели на основе posterior probabilities
    /// </summary>    
    /// <param name="posteriors_device">Posterior probabilities [N x K]. Device: _device</param>
    private void UpdateParameters(Tensor posteriors_device)
    {
        List<int> emptyClusters = new(_numClusters);

        using (var disposeScope = torch.NewDisposeScope())
        {   
            // Векторное вычисление размеров эффективных выборок [K]
            var effectiveSampleSizes_device = torch.sum(posteriors_device, dim: 0);

            // Обновляем коэффициенты смешивания векторно [K]
            MixingCoefficients = (effectiveSampleSizes_device / _numSamples).to(CPU).DetachFromDisposeScope();

            for (int k = 0; k < _numClusters; k += 1)
            {
                // Обновляем коэффициент смешивания α_k
                var effectiveSampleSize = torch.sum(posteriors_device[.., k]).item<float>();

                if (effectiveSampleSize < 1e-6f) // Кластер пуст или почти пуст
                {
                    // Добавить в список для переинициализации
                    emptyClusters.Add(k);
                }
            }

            // Векторное вычисление всех взвешенных сумм [K x D]
            // posteriors.transpose(0,1): K x N, mm с N x D -> K x D
            var weightedSums_device = torch.mm(posteriors_device.transpose(0, 1), _oldVectorsTensor_device);

            for (int k = 0; k < _numClusters; k += 1)
            {
                // Взвешенная сумма для k-го кластера [D]
                var weightedSum_device = weightedSums_device[k];  
                
                // Обновляем направление среднего μ_k
                float resultantLength = torch.norm(weightedSum_device).item<float>();
                // Пропускаем назначение, если нет элементов.
                if (resultantLength > 1e-8f)
                {
                    MeanDirections[k] = (weightedSum_device / resultantLength).to(CPU);
                }

                // Вычисляем средний resultant length для оценки κ_k
                float effectiveSampleSize = effectiveSampleSizes_device[k].item<float>();
                float meanResultantLength = resultantLength / effectiveSampleSize;

                // Обновляем параметр концентрации κ_k используя аппроксимацию из статьи
                var newConcentration = ApproximateConcentration(meanResultantLength, _dimension);
                if (!float.IsNaN(newConcentration))
                    Concentrations[k] = newConcentration;
            }
        }         

        //ReinitializeEmptyClusters(emptyClusters, oldVectorsTensor, posteriors);
        SplitLargestCluster(emptyClusters);
        
        float epsilon = 1e-8f;
        MixingCoefficients = torch.clamp(MixingCoefficients, min: epsilon);  // Floor        
        // Нормализуем коэффициенты смешивания для обеспечения того, что их сумма равна 1        
        MixingCoefficients = MixingCoefficients / torch.sum(MixingCoefficients);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="emptyClusters"></param>
    /// <param name="oldVectorsTensor">Входные данные [N x D]. Device: CPU</param>
    /// <param name="posteriors">Posterior probabilities [N x K]. Device: CPU</param>
    private void ReinitializeEmptyClusters(List<int> emptyClusters, Tensor oldVectorsTensor, Tensor posteriors)
    {
        foreach (int emptyCluster in emptyClusters)
        {
            // Найти точку, наиболее удалённую от всех текущих центров
            var distances = torch.mm(oldVectorsTensor, MeanDirections.t()); // [N × K] cosine similarities
            var maxDistances = torch.max(distances, dim: 1).values; // [N] max similarity for each point
            var farthestPointIdx = torch.argmin(maxDistances); // Точка с минимальной max similarity

            // Переинициализировать центр кластера
            MeanDirections[emptyCluster] = oldVectorsTensor[farthestPointIdx].clone();

            // Установить умеренную концентрацию
            Concentrations[emptyCluster] = 1.0f;

            // Установить малый коэффициент смешивания
            MixingCoefficients[emptyCluster] = 1.0f / _numClusters;
        }
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="emptyClusters"></param>
    /// <param name="oldVectorsTensor">Входные данные [N x D]. Device: CPU</param>
    /// <param name="posteriors_device">Posterior probabilities [N x K]. Device: _device</param>
    private void SplitLargestCluster(List<int> emptyClusters)
    {   
        foreach (int emptyCluster in emptyClusters)
        {
            _logger.LogInformation($"Разбиение кластера {emptyCluster}.");

            Tensor posteriors;
            using (var posteriors_device = ComputePosteriors_device())
            {
                posteriors = posteriors_device.to(CPU);
            }
                // Найти кластер с максимальным числом точек
            var clusterSizes = torch.sum(posteriors, dim: 0); // [K]
            var largestClusterIndex = torch.argmax(clusterSizes).item<long>();

            // Найти точки, принадлежащие крупнейшему кластеру
            var belongsToLargest = posteriors[.., largestClusterIndex] > 0.5f;
            var clusterPoints = _oldVectorsTensor[belongsToLargest];

            if (clusterPoints.shape[0] > 1)
            {
                // Выбрать точку, наиболее удалённую от центра крупнейшего кластера
                var similarities = torch.mv(clusterPoints, MeanDirections[largestClusterIndex]);
                var farthestInCluster = torch.argmin(similarities);

                // Переинициализировать пустой кластер этой точкой
                MeanDirections[emptyCluster] = clusterPoints[farthestInCluster].clone();
                Concentrations[emptyCluster] = Concentrations[largestClusterIndex] * 0.5f;
                MixingCoefficients[emptyCluster] = MixingCoefficients[largestClusterIndex] * 0.5f;
                MixingCoefficients[largestClusterIndex] *= 0.5f;
            }
        }
    }

    /// <summary>
    /// Вычисляет приближённое значение параметра концентрации κ
    /// Использует аппроксимацию из статьи: κ ≈ r̄d - r̄³ / (1 - r̄²)
    /// где r̄ - средний resultant length
    /// </summary>
    /// <param name="meanResultantLength">Средний resultant length r̄</param>
    /// <param name="dimension">Размерность данных d</param>
    /// <returns>Приближённое значение κ</returns>
    private float ApproximateConcentration(float meanResultantLength, long dimension)
    {
        // Ограничиваем r̄ чтобы избежать деления на ноль
        var r = MathF.Max(0.0f, MathF.Min(0.999999f, meanResultantLength));
        var d = (float)dimension;
            
        // Применяем аппроксимацию из статьи Banerjee et al.
        // κ ≈ r̄d - r̄³ / (1 - r̄²)
        var numerator = r * d - MathF.Pow(r, 3);
        var denominator = 1.0f - MathF.Pow(r, 2);
        if (denominator < 1e-8f)
            denominator = 1e-8f;

        var kappa = numerator / denominator;

        // Ограничиваем κ положительными разумными значениями
        // ORIG: return MathF.Max(0.01f, MathF.Min(1000.0f, kappa));
        return MathF.Max(0.01f, MathF.Min(1000.0f, kappa)); 
    }        

    /// <summary>
    /// Вычисляет полную логарифмическую вероятность данных для текущих параметров модели
    /// </summary>
    /// <param name="oldVectorsTensor">Входные данные</param>
    /// <returns>Логарифмическая вероятность</returns>
    private float ComputeLogLikelihood(Tensor oldVectorsTensor)
    {
        var numSamples = oldVectorsTensor.shape[0];
        var totalLogLikelihood = 0.0f;

        //using (var disposeScope = torch.NewDisposeScope())
        {
            // Вычисляем cosine similarity
            var cosineSimilarity = torch.mm(oldVectorsTensor, MeanDirections.t());

            for (long i = 0; i < numSamples; i += 1)
            {
                var sampleLogLikelihood = float.NegativeInfinity;

                for (int k = 0; k < _numClusters; k += 1)
                {
                    // Вычисляем логарифм компоненты смеси
                    var logNormalizingConstant = ComputeLogNormalizingConstant(
                        Concentrations[k].item<float>(),
                        oldVectorsTensor.shape[1]
                    );

                    var logComponent = MathF.Log(MixingCoefficients[k].item<float>()) +
                                        logNormalizingConstant +
                                        Concentrations[k].item<float>() * cosineSimilarity[i, k].item<float>();

                    // Используем log-sum-exp trick для численной стабильности
                    if (sampleLogLikelihood == float.NegativeInfinity)
                    {
                        sampleLogLikelihood = logComponent;
                    }
                    else
                    {
                        var maxLog = MathF.Max(sampleLogLikelihood, logComponent);
                        sampleLogLikelihood = maxLog + MathF.Log(
                            MathF.Exp(sampleLogLikelihood - maxLog) + MathF.Exp(logComponent - maxLog)
                        );                        
                    }
                }

                if (!float.IsNaN(sampleLogLikelihood))
                    totalLogLikelihood += sampleLogLikelihood;
            }
        }            
            
        return totalLogLikelihood;
    }

    /// <summary>
    /// Предсказывает кластерные назначения для новых данных
    /// </summary>    
    /// <returns>Назначения кластеров [N] (индексы от 0 до K-1)</returns>
    public Tensor Predict()
    {
        using var posteriors_device = ComputePosteriors_device();
        return torch.argmax(posteriors_device.to(CPU), dim: 1);
    }

    /// <summary>
    /// Возвращает мягкие назначения (вероятности принадлежности) для данных
    /// </summary>    
    /// <returns>Матрица вероятностей [N x K]</returns>
    public Tensor PredictProba()
    {
        using var posteriors_device = ComputePosteriors_device();
        return posteriors_device.to(CPU);
    }

    /// <summary>
    /// Выводит информацию о изученной модели
    /// </summary>
    public void PrintModelSummary()
    {
        _logger.LogInformation("=== Результаты vMF Кластеризации ===");
        _logger.LogInformation($"Количество кластеров: {_numClusters}");
        _logger.LogInformation($"Алгоритм назначения: {(false ? "Hard (жёсткое)" : "Soft (мягкое)")}");
            
        _logger.LogInformation("\nПараметры кластеров:");
        for (int k = 0; k < _numClusters; k++)
        {
            _logger.LogInformation($"\nКластер {k}:");
            _logger.LogInformation($"  Коэффициент смешивания α_{k}: {MixingCoefficients[k].item<float>():F4}");
            _logger.LogInformation($"  Концентрация κ_{k}: {Concentrations[k].item<float>():F4}");
            _logger.LogInformation($"  Направление μ_{k}: [{string.Join(", ", MeanDirections[k].data<float>().Take(5).Select(x => x.ToString("F3")))}...]");
        }
            
        if (LogLikelihoodHistory.Any())
        {
            _logger.LogInformation($"\nФинальная log-likelihood: {LogLikelihoodHistory.Last():F6}");
            _logger.LogInformation($"Количество итераций: {LogLikelihoodHistory.Count}");
        }
    }
}
