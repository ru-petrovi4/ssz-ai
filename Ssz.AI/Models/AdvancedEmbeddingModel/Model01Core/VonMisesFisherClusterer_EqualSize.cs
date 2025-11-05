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
/// Реализация кластеризации von Mises-Fisher (vMF) на единичной сфере с ОДИНАКОВЫМ количеством элементов в кластерах
/// Использует модифицированный EM-алгоритм со сбалансированным назначением
/// </summary>
public class VonMisesFisherClusterer_EqualSize
{
    private readonly Device _device;
    private readonly IUserFriendlyLogger _userFriendlyLogger;

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
    private int _numSamples;
    private long _dimension;

    /// <summary>
    /// Целевой размер кластера (одинаковый для всех)
    /// </summary>
    private long _targetClusterSize;

    // Параметры модели - изучаются в процессе обучения
    /// <summary>
    /// μ_k - направления средних для каждого кластера [K x D]
    /// </summary>
    /// device: CPU
    public Tensor MeanDirections { get; private set; } = null!;

    /// <summary>
    /// κ_k - параметры концентрации для каждого кластера [K]
    /// </summary>
    /// device: CPU
    public Tensor Concentrations { get; private set; } = null!;

    /// <summary>
    /// α_k - коэффициенты смешивания [K] (при сбалансированном назначении все равны 1/K)
    /// </summary>
    /// device: CPU
    public Tensor MixingCoefficients { get; private set; } = null!;

    // Логи процесса обучения
    public List<float> LogLikelihoodHistory { get; private set; }

    /// <summary>
    /// Конструктор кластеризатора vMF со сбалансированным назначением
    /// </summary>
    /// <param name="device">Устройство для вычислений (CPU/CUDA)</param>
    /// <param name="userFriendlyLogger">Логгер для вывода информации</param>
    /// <param name="numClusters">Количество кластеров K</param>
    /// <param name="maxIterations">Максимальное количество итераций EM алгоритма</param>
    /// <param name="tolerance">Порог сходимости для логарифмической вероятности</param>
    public VonMisesFisherClusterer_EqualSize(
        Device device,
        IUserFriendlyLogger userFriendlyLogger,
        int numClusters,
        int maxIterations,
        float tolerance)
    {
        _device = device;
        _userFriendlyLogger = userFriendlyLogger;
        _numClusters = numClusters;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        LogLikelihoodHistory = new List<float>();
    }

    /// <summary>
    /// Основной метод обучения кластеризатора на нормированных данных со СБАЛАНСИРОВАННЫМ назначением
    /// </summary>
    /// <param name="oldVectorsTensor">Нормированные данные [N x D], где N - количество точек, D - размерность. Device: CPU</param>
    public void Fit(Tensor oldVectorsTensor)
    {
        // Проверяем, что данные корректно нормированы
        ValidateNormalizedData(oldVectorsTensor);

        // Инициализируем параметры модели
        InitializeParameters(oldVectorsTensor);

        // Вычисляем целевой размер кластера (одинаковый для всех кластеров)
        // targetClusterSize = floor(N / K)
        _targetClusterSize = _numSamples / _numClusters;

        _userFriendlyLogger.LogInformation($"Целевой размер каждого кластера: {_targetClusterSize}");
        _userFriendlyLogger.LogInformation($"Количество точек, которые не войдут в кластеры: {_numSamples - _targetClusterSize * _numClusters}");

        float prevLogLikelihood = float.NegativeInfinity;

        // Основной цикл EM алгоритма со сбалансированным назначением
        for (int iteration = 0; iteration < _maxIterations; iteration += 1)
        {
            // E-шаг (модифицированный): вычисляем СБАЛАНСИРОВАННОЕ жёсткое назначение
            // Вместо мягких вероятностей p(k|x_i) используем жёсткое назначение с ограничением размера
            using var assignments_device = ComputeBalancedAssignments_device();

            // M-шаг: обновляем параметры модели на основе сбалансированного назначения
            UpdateParametersBalanced(assignments_device);

            // Вычисляем логарифмическую вероятность для проверки сходимости
            var logLikelihood = ComputeLogLikelihood(_oldVectorsTensor[..Math.Min(1000, _numSamples), ..]);
            LogLikelihoodHistory.Add(logLikelihood);

            // Проверяем сходимость
            if (Math.Abs(logLikelihood - prevLogLikelihood) < _tolerance)
            {
                _userFriendlyLogger.LogInformation($"Сходимость достигнута на итерации {iteration + 1}");
                break;
            }

            prevLogLikelihood = logLikelihood;

            _userFriendlyLogger.LogInformation($"Итерация {iteration + 1}: Log-Likelihood = {logLikelihood:F6}");
        }

        _userFriendlyLogger.LogInformation($"Обучение завершено. Финальная Log-Likelihood: {prevLogLikelihood:F6}");
    }

    /// <summary>
    /// Модифицированный E-шаг: вычисляет СБАЛАНСИРОВАННОЕ жёсткое назначение точек кластерам
    /// Решает задачу оптимального назначения с ограничением: каждый кластер содержит ровно targetClusterSize точек
    /// </summary>
    /// <returns>Тензор назначений [N] со значениями 0..K-1, где каждое значение k встречается ровно targetClusterSize раз. Device: _device</returns>
    private Tensor ComputeBalancedAssignments_device()
    {
        using (var disposeScope = torch.NewDisposeScope())
        {
            // Вычисляем матрицу "стоимости" (отрицательная log-вероятность) [N x K]
            // Чем выше вероятность принадлежности точки i к кластеру k, тем НИЖЕ стоимость
            var costMatrix_device = torch.zeros(size: new long[] { _numSamples, _numClusters }, device: _device);

            var meanDirections_device = MeanDirections.to(_device);
            var concentrations_device = Concentrations.to(_device);

            // Вычисляем cosine similarities [N x K]
            var cosineSimilarities_device = torch.mm(_oldVectorsTensor_device, meanDirections_device.transpose(dim0: 0, dim1: 1));

            for (int k = 0; k < _numClusters; k += 1)
            {
                // Логарифм нормализующей константы log c_d(κ_k)
                var logNormalizingConstant = ComputeLogNormalizingConstant(
                    Concentrations[k].item<float>(),
                    _dimension
                );

                // Логарифм vMF плотности для кластера k: log c_d(κ) + κ * μ^T * x
                // Стоимость = ОТРИЦАТЕЛЬНАЯ log-вероятность
                costMatrix_device[.., k] = -(logNormalizingConstant +
                    concentrations_device[k] * cosineSimilarities_device[.., k]);
            }

            // Применяем жадный алгоритм сбалансированного назначения
            // Возвращает тензор [N] с назначениями 0..K-1
            var assignments_device = GreedyBalancedAssignment(costMatrix_device, _targetClusterSize);

            return assignments_device.DetachFromDisposeScope();
        }
    }

    /// <summary>
    /// Жадный сбалансированный алгоритм назначения точек кластерам
    /// Оптимизированная версия с использованием приоритетной очереди
    /// </summary>
    /// <param name="similarities">Тензор схожести [N x K] - косинусные сходства между точками и центроидами</param>
    /// <returns>Тензор назначений [N] - индекс кластера для каждой точки</returns>
    private Tensor GreedyBalancedAssignment(Tensor similarities, long targetSize)
    {
        using (var disposeScope = torch.NewDisposeScope())
        {
            var numSamples = similarities.shape[0];
            var numClusters = similarities.shape[1];           

            // Тензор для хранения назначений: для каждой точки указывается индекс её кластера
            // Инициализируется значением -1, что означает "ещё не назначено"
            var assignments = torch.full(new long[] { numSamples }, -1, dtype: ScalarType.Int64, device: torch.CPU);

            // Тензор для отслеживания текущего размера каждого кластера
            // Инициализируется нулями: в начале все кластеры пустые
            // Тензор [K], где clusterSizes[k] - текущее количество точек в кластере k
            using var clusterSizes = torch.zeros(new long[] { numClusters }, dtype: ScalarType.Int64, device: torch.CPU);

            // Создаём матрицу стоимостей: cost = -similarity
            // Отрицательное сходство означает, что минимизация стоимости = максимизация сходства
            // Тензор [N x K], где costMatrix[i, k] - стоимость назначения точки i в кластер k
            using var costMatrix = similarities.neg();

            // ВЕКТОРИЗАЦИЯ: Получаем плоский вид матрицы стоимостей для эффективной сортировки
            // flatten() преобразует матрицу [N x K] в вектор [N * K]
            // Каждый элемент вектора соответствует одному варианту назначения
            using var costFlat = costMatrix.flatten();

            // Получаем индексы отсортированных элементов в порядке возрастания стоимости
            // argsort() возвращает индексы, которые упорядочивают массив
            // sortedIndices - тензор [N * K] с индексами от 0 до N*K-1
            // Индексы упорядочены так, что costFlat[sortedIndices[0]] <= costFlat[sortedIndices[1]] <= ...
            using var sortedIndices = torch.argsort(costFlat, dim: 0, descending: false);

            // Счётчик назначенных точек для отслеживания прогресса
            long assignedCount = 0;

            // Общее количество возможных назначений (N * K)
            long totalOptions = numSamples * numClusters;

            // Проходим по отсортированным индексам (от лучших назначений к худшим)
            for (long idx = 0; idx < totalOptions; idx += 1)
            {
                // Получаем плоский индекс текущего варианта назначения
                // flatIdx - индекс в диапазоне [0, N*K-1]
                long flatIdx = sortedIndices[idx].item<long>();

                // Преобразуем плоский индекс обратно в двумерные координаты (точка, кластер)
                // Формула декодирования: flatIdx = pointIdx * K + clusterIdx
                // pointIdx = flatIdx / K (целочисленное деление)
                // clusterIdx = flatIdx % K (остаток от деления)
                long pointIdx = flatIdx / numClusters;
                long clusterIdx = flatIdx % numClusters;

                // Получаем текущее состояние точки: назначена ли она уже какому-то кластеру
                // -1 означает "не назначена"
                long currentAssignment = assignments[pointIdx].item<long>();

                // Получаем текущий размер кластера
                long currentClusterSize = clusterSizes[clusterIdx].item<long>();

                // Проверяем два условия для допустимого назначения:
                // 1. Точка ещё не назначена (currentAssignment == -1)
                // 2. Кластер не заполнен (currentClusterSize < targetSize)
                if (currentAssignment == -1 && currentClusterSize < targetSize)
                {
                    // Назначаем точку кластеру
                    // Обновляем тензор назначений: для точки pointIdx записываем индекс кластера
                    assignments[pointIdx] = clusterIdx;

                    // Увеличиваем размер кластера на 1
                    // clusterSizes[clusterIdx] = clusterSizes[clusterIdx] + 1
                    clusterSizes[clusterIdx] = clusterSizes[clusterIdx].add(1);

                    // Увеличиваем счётчик назначенных точек на 1
                    assignedCount += 1;

                    // Если все точки назначены (assignedCount == numSamples), прерываем цикл
                    // Это оптимизация: нет смысла продолжать, если работа завершена
                    if (assignedCount == numSamples)
                    {
                        break;
                    }
                }
            }

            // ОБРАБОТКА НЕНАЗНАЧЕННЫХ ТОЧЕК (если N не делится нацело на K)
            // Если количество точек не кратно количеству кластеров, останутся неназначенные точки
            // Например, при N=100 и K=3: 100 / 3 = 33, останется 1 неназначенная точка
            if (assignedCount < numSamples)
            {
                var remainingCount = numSamples - assignedCount;
                _userFriendlyLogger.LogWarning($"Осталось {remainingCount} неназначенных точек (N mod K != 0). Распределяем по кластерам с минимальной стоимостью.");

                // ВЕКТОРИЗАЦИЯ: Находим маску неназначенных точек
                // assignments.eq(-1) создаёт булев тензор [N], где True = точка не назначена
                using var unassignedMask = assignments.eq(-1);

                // Получаем индексы неназначенных точек
                // nonzero() возвращает тензор [M, 1], где M - количество неназначенных точек
                // squeeze() убирает размерность 1, получаем тензор [M]
                using var unassignedIndices = unassignedMask.nonzero().squeeze(1);

                // Количество неназначенных точек
                long unassignedCount = unassignedIndices.shape[0];

                // Для каждой неназначенной точки находим кластер с минимальной стоимостью
                for (long i = 0; i < unassignedCount; i += 1)
                {
                    // Получаем индекс текущей неназначенной точки
                    long pointIdx = unassignedIndices[i].item<long>();

                    // ВЕКТОРИЗАЦИЯ: Получаем вектор стоимостей для данной точки
                    // costMatrix[pointIdx] - тензор [K] со стоимостями назначения в каждый кластер
                    using var pointCosts = costMatrix[pointIdx];

                    // Находим индекс кластера с минимальной стоимостью
                    // argmin() возвращает индекс минимального элемента в тензоре
                    // dim=0 означает поиск минимума вдоль первой (и единственной) размерности
                    using var bestClusterTensor = torch.argmin(pointCosts, dim: 0);
                    long bestCluster = bestClusterTensor.item<long>();

                    // Назначаем точку кластеру с минимальной стоимостью
                    assignments[pointIdx] = bestCluster;

                    // Увеличиваем размер выбранного кластера на 1
                    // Этот кластер теперь будет иметь на 1 точку больше целевого размера
                    // clusterSizes[bestCluster] = clusterSizes[bestCluster] + 1
                    clusterSizes[bestCluster] = clusterSizes[bestCluster].add(1);
                }

                // Логируем финальное распределение размеров кластеров
                // Преобразуем тензор clusterSizes в массив для вывода
                var clusterSizesArray = new long[numClusters];
                for (int k = 0; k < numClusters; k += 1)
                {
                    clusterSizesArray[k] = clusterSizes[k].item<long>();
                }
                _userFriendlyLogger.LogInformation($"Финальные размеры кластеров: [{string.Join(", ", clusterSizesArray)}]");
            }

            // Возвращаем результат - тензор назначений, перемещённый в основную память
            // MoveToOuterDisposeScope() гарантирует, что тензор не будет удалён при выходе из using
            return assignments.MoveToOuterDisposeScope();
        }
    }

    /// <summary>
    /// Модифицированный M-шаг: обновляет параметры модели на основе ЖЁСТКОГО сбалансированного назначения
    /// </summary>
    /// <param name="assignments_device">Жёсткие назначения [N] со значениями 0..K-1. Device: _device</param>
    private void UpdateParametersBalanced(Tensor assignments_device)
    {
        var assignments = assignments_device.to(CPU);

        List<int> emptyClusters = new(_numClusters);

        using (var disposeScope = torch.NewDisposeScope())
        {
            // Обновляем направления средних μ_k и концентрации κ_k
            for (int k = 0; k < _numClusters; k += 1)
            {
                // Булева маска точек, принадлежащих кластеру k
                var mask = assignments.eq(k); // [N]
                var clusterPoints = _oldVectorsTensor[mask]; // [clusterSize x D]

                long clusterSize = clusterPoints.shape[0];

                if (clusterSize < 1)
                {
                    continue; // Пустой кластер, обработаем позже
                }

                // Взвешенная сумма векторов кластера (в данном случае просто сумма, так как веса одинаковые)
                var sumVectors = torch.sum(clusterPoints, dim: 0); // [D]

                // Длина результирующего вектора (resultant length)
                float resultantLength = torch.norm(sumVectors).item<float>();

                // Обновляем направление среднего: μ_k = sum / ||sum||
                if (resultantLength > 1e-8f)
                {
                    MeanDirections[k] = sumVectors / resultantLength;
                }

                // Средняя длина результирующего вектора: r̄ = ||sum|| / clusterSize
                float meanResultantLength = resultantLength / clusterSize;

                // Обновляем параметр концентрации κ_k
                var newConcentration = ApproximateConcentration(meanResultantLength, _dimension);
                if (!float.IsNaN(newConcentration))
                {
                    Concentrations[k] = newConcentration;
                }
            }

            // Обрабатываем пустые кластеры (разбиением крупнейшего)
            if (emptyClusters.Count > 0)
            {
                SplitLargestCluster(emptyClusters, assignments);
            }
        }
    }

    /// <summary>
    /// Разбиение крупнейшего кластера для заполнения пустых кластеров
    /// </summary>
    /// <param name="emptyClusters">Список индексов пустых кластеров</param>
    /// <param name="assignments">Текущие назначения [N]. Device: CPU</param>
    private void SplitLargestCluster(List<int> emptyClusters, Tensor assignments)
    {
        foreach (int emptyCluster in emptyClusters)
        {
            _userFriendlyLogger.LogInformation($"Переинициализация пустого кластера {emptyCluster}.");

            // Найти кластер с максимальным числом точек
            var clusterSizes = torch.zeros(_numClusters);
            for (int k = 0; k < _numClusters; k += 1)
            {
                clusterSizes[k] = torch.sum(assignments.eq(k)).item<long>();
            }

            var largestClusterIndex = (int)torch.argmax(clusterSizes).item<long>();

            // Найти точки, принадлежащие крупнейшему кластеру
            var belongsToLargest = assignments.eq(largestClusterIndex);
            var clusterPoints = _oldVectorsTensor[belongsToLargest];

            if (clusterPoints.shape[0] > 1)
            {
                // Выбрать точку, наиболее удалённую от центра крупнейшего кластера
                var similarities = torch.mv(clusterPoints, MeanDirections[largestClusterIndex]);
                var farthestInCluster = torch.argmin(similarities).item<long>();

                // Переинициализировать пустой кластер этой точкой
                MeanDirections[emptyCluster] = clusterPoints[farthestInCluster].clone();
                Concentrations[emptyCluster] = Concentrations[largestClusterIndex] * 0.5f;                
            }
        }
    }

    /// <summary>
    /// Получение результатов кластеризации со СБАЛАНСИРОВАННЫМ назначением
    /// </summary>
    /// <param name="words">Список слов для назначения кластеров</param>
    /// <param name="clusterization_AlgorithmData">Данные алгоритма кластеризации</param>
    public void GetResult(
        List<Word> words,
        Clusterization_AlgorithmData clusterization_AlgorithmData)
    {
        // Вычисляем сбалансированное назначение для всех точек
        using var assignments_device = ComputeBalancedAssignments_device();
        var assignments = assignments_device.to(CPU);

        // Сохраняем информацию о кластерах
        for (int k = 0; k < _numClusters; k += 1)
        {
            var clusterInfo = clusterization_AlgorithmData.ClusterInfos[k];
            MeanDirections.select(dim: 0, index: k).data<float>().CopyTo(clusterInfo.CentroidOldVectorNormalized);
            clusterInfo.MixingCoefficient = MixingCoefficients[k].item<float>();
            clusterInfo.Concentration = Concentrations[k].item<float>();
        }

        // Назначаем каждое слово его кластеру
        for (int i = 0; i < _numSamples; i += 1)
        {
            var assignedCluster = assignments[i].item<long>();
            words[i].ClusterIndex = (int)assignedCluster;
        }

        clusterization_AlgorithmData.MeanDirections = MeanDirections;
        clusterization_AlgorithmData.Concentrations = Concentrations;
        clusterization_AlgorithmData.MixingCoefficients = MixingCoefficients;

        // Логируем распределение размеров кластеров
        var clusterSizes = new long[_numClusters];
        for (int i = 0; i < _numSamples; i += 1)
        {
            var cluster = assignments[i].item<long>();
            clusterSizes[cluster] += 1;
        }

        _userFriendlyLogger.LogInformation("Распределение размеров кластеров:");
        for (int k = 0; k < _numClusters; k += 1)
        {
            _userFriendlyLogger.LogInformation($"  Кластер {k}: {clusterSizes[k]} элементов");
        }
    }    

    /// <summary>
    /// Вычисляет логарифм нормализующей константы c_d(κ) для vMF распределения
    /// Использует аппроксимацию для высоких размерностей чтобы избежать переполнения
    /// 
    /// Формула vMF плотности: f(x|μ,κ) = c_d(κ) * exp(κ * μ^T * x)
    /// где c_d(κ) = κ^(d/2-1) / ((2π)^(d/2) * I_(d/2-1)(κ))
    /// I_ν(κ) - модифицированная функция Бесселя первого рода порядка ν
    /// </summary>
    /// <param name="concentration">Параметр концентрации κ</param>
    /// <param name="dimension">Размерность d</param>
    /// <returns>Логарифм нормализующей константы</returns>
    public static float ComputeLogNormalizingConstant(float concentration, long dimension)
    {
        double d = dimension;
        double nu = d / 2.0 - 1.0; // Порядок Бесселя ν = d/2 - 1
        double halfD = d / 2.0; // d/2

        if (concentration < 1e-8) // Порог для малого κ, где I_ν(κ) ≈ 0 из-за underflow
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
        _numSamples = (int)oldVectorsTensor.shape[0];
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

                // Преобразуем similarity в distance: distance = 1 - similarity
                // Возводим в квадрат для усиления различий
                probabilities.neg_().add_(1.0f).pow_(2);

                probabilities = probabilities / torch.sum(probabilities);

                // Используем multinomial sampling для выбора индекса
                var selectedIndex = torch.multinomial(input: probabilities.t(), num_samples: 1).item<long>();
                meanDirections_device[k] = _oldVectorsTensor_device[selectedIndex];

                if ((k + 1) % 10 == 0)
                {
                    _userFriendlyLogger.LogInformation($"Инициализирован кластер {k}/{_numClusters}.");
                }
            }

            MeanDirections = meanDirections_device.to(CPU).DetachFromDisposeScope();
        }

        // Инициализация параметров концентрации
        // Начинаем с умеренных значений концентрации
        Concentrations = torch.ones(size: _numClusters, dtype: torch.float32) * 1.0f;

        // Инициализация коэффициентов смешивания (равномерное распределение)
        MixingCoefficients = torch.ones(_numClusters, dtype: torch.float32) / _numClusters;

        _userFriendlyLogger.LogInformation("Параметры инициализированы:");
        _userFriendlyLogger.LogInformation($"Количество кластеров: {_numClusters}");
        _userFriendlyLogger.LogInformation($"Размерность данных: {dimension}");
        _userFriendlyLogger.LogInformation($"Количество образцов: {numSamples}");
    }

    /// <summary>
    /// Вычисляет приближённое значение параметра концентрации κ
    /// Использует аппроксимацию из статьи Banerjee et al.: κ ≈ r̄d - r̄³ / (1 - r̄²)
    /// где r̄ - средний resultant length, d - размерность
    /// 
    /// Физический смысл:
    /// - r̄ близко к 0: точки распределены равномерно по сфере → κ ≈ 0 (низкая концентрация)
    /// - r̄ близко к 1: точки сконцентрированы вокруг μ → κ велико (высокая концентрация)
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
        return MathF.Max(0.01f, MathF.Min(1000.0f, kappa));
    }

    /// <summary>
    /// Вычисляет полную логарифмическую вероятность данных для текущих параметров модели
    /// 
    /// Формула: log L = Σᵢ log[ Σₖ αₖ * f_vMF(xᵢ | μₖ, κₖ) ]
    /// где αₖ - коэффициент смешивания, f_vMF - плотность vMF распределения
    /// </summary>
    /// <param name="oldVectorsTensor">Входные данные</param>
    /// <returns>Логарифмическая вероятность</returns>
    private float ComputeLogLikelihood(Tensor oldVectorsTensor)
    {
        var numSamples = oldVectorsTensor.shape[0];
        var totalLogLikelihood = 0.0f;

        // Вычисляем cosine similarity [N x K]
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

        return totalLogLikelihood;
    }

    /// <summary>
    /// Предсказывает кластерные назначения для данных со СБАЛАНСИРОВАННЫМ назначением
    /// </summary>
    /// <returns>Назначения кластеров [N] (индексы от 0 до K-1)</returns>
    public Tensor Predict()
    {
        using var assignments_device = ComputeBalancedAssignments_device();
        return assignments_device.to(CPU);
    }

    /// <summary>
    /// Выводит информацию о изученной модели
    /// </summary>
    public void PrintModelSummary()
    {
        _userFriendlyLogger.LogInformation("=== Результаты vMF Кластеризации (СБАЛАНСИРОВАННОЕ назначение) ===");
        _userFriendlyLogger.LogInformation($"Количество кластеров: {_numClusters}");
        _userFriendlyLogger.LogInformation($"Целевой размер кластера: {_targetClusterSize}");
        _userFriendlyLogger.LogInformation("\nПараметры кластеров:");

        for (int k = 0; k < _numClusters; k += 1)
        {
            _userFriendlyLogger.LogInformation($"\nКластер {k}:");
            _userFriendlyLogger.LogInformation($"  Коэффициент смешивания α_{k}: {MixingCoefficients[k].item<float>():F4}");
            _userFriendlyLogger.LogInformation($"  Концентрация κ_{k}: {Concentrations[k].item<float>():F4}");
            _userFriendlyLogger.LogInformation($"  Направление μ_{k}: [{string.Join(", ", MeanDirections[k].data<float>().Take(5).Select(x => x.ToString("F3")))}...]");
        }

        if (LogLikelihoodHistory.Any())
        {
            _userFriendlyLogger.LogInformation($"\nФинальная log-likelihood: {LogLikelihoodHistory.Last():F6}");
            _userFriendlyLogger.LogInformation($"Количество итераций: {LogLikelihoodHistory.Count}");
        }
    }
}
