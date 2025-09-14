using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Models;
using System;
using System.Collections.Generic;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Utils;

/// <summary>
/// Простая реализация SGD оптимизатора для обучения нейронных сетей.
/// Поддерживает momentum, weight decay и градиентное обновление весов.
/// Оптимизирован для работы с MatrixFloat и высокой производительности.
/// </summary>
public class SGDOptimizer
{
    private readonly float _learningRate;
    private readonly float _weightDecay;
    private readonly float _momentum;

    // Хранение momentum для каждого параметра
    private readonly Dictionary<object, MatrixFloat> _momentumMatrices = new();
    private readonly Dictionary<object, float[]> _momentumVectors = new();

    /// <summary>
    /// Инициализация SGD оптимизатора.
    /// </summary>
    /// <param name="learningRate">Скорость обучения</param>
    /// <param name="weightDecay">Коэффициент L2 регуляризации</param>
    /// <param name="momentum">Коэффициент momentum (по умолчанию 0.9)</param>
    public SGDOptimizer(float learningRate, float weightDecay = 0.0f, float momentum = 0.9f)
    {
        _learningRate = learningRate;
        _weightDecay = weightDecay;
        _momentum = momentum;
    }

    /// <summary>
    /// Обновление отображающей матрицы по градиентам.
    /// Применяет SGD с momentum и weight decay.
    /// </summary>
    /// <param name="mappingMatrix">Отображающая матрица для обновления</param>
    /// <param name="gradients">Градиенты матрицы</param>
    public void UpdateMappingMatrix(MatrixFloat mappingMatrix, MatrixFloat gradients)
    {
        // Инициализация momentum матрицы при первом обращении
        if (!_momentumMatrices.ContainsKey(mappingMatrix))
        {
            _momentumMatrices[mappingMatrix] = new MatrixFloat(mappingMatrix.Dimensions);
        }

        var momentum = _momentumMatrices[mappingMatrix];

        // Применение weight decay (L2 регуляризация)
        if (_weightDecay > 0)
        {
            ApplyWeightDecay(mappingMatrix, gradients);
        }

        // Обновление momentum: m = β * m + (1-β) * g
        UpdateMomentum(momentum, gradients);

        // Обновление параметров: θ = θ - α * m
        ApplyGradientUpdate(mappingMatrix, momentum);
    }

    /// <summary>
    /// Обновление весов дискриминатора.
    /// Обновляет все слои дискриминатора с учетом momentum.
    /// </summary>
    /// <param name="discriminator">Дискриминатор для обновления</param>
    /// <param name="gradients">Список градиентов для каждого слоя</param>
    public void UpdateWeights(Models.Discriminator discriminator, List<(MatrixFloat weightGradients, float[] biasGradients)> gradients)
    {
        for (int layer = 0; layer < discriminator.LayerCount; layer++)
        {
            var weights = discriminator.GetWeights(layer);
            var biases = discriminator.GetBiases(layer);
            var (weightGrads, biasGrads) = gradients[layer];

            // Обновление весов
            UpdateMappingMatrix(weights, weightGrads);

            // Обновление смещений
            UpdateBiases(biases, biasGrads, layer);
        }
    }

    /// <summary>
    /// Обновление смещений (bias) с momentum.
    /// </summary>
    /// <param name="biases">Массив смещений</param>
    /// <param name="gradients">Градиенты смещений</param>
    /// <param name="layerId">Идентификатор слоя для momentum</param>
    private void UpdateBiases(float[] biases, float[] gradients, int layerId)
    {
        var key = $"bias_{layerId}";

        // Инициализация momentum вектора при первом обращении
        if (!_momentumVectors.ContainsKey(key))
        {
            _momentumVectors[key] = new float[biases.Length];
        }

        var momentum = _momentumVectors[key];

        // Применение weight decay
        if (_weightDecay > 0)
        {
            for (int i = 0; i < gradients.Length; i++)
            {
                gradients[i] += _weightDecay * biases[i];
            }
        }

        // Обновление momentum и параметров
        for (int i = 0; i < biases.Length; i++)
        {
            momentum[i] = _momentum * momentum[i] + (1 - _momentum) * gradients[i];
            biases[i] -= _learningRate * momentum[i];
        }
    }

    /// <summary>
    /// Применение L2 регуляризации (weight decay) к градиентам.
    /// </summary>
    /// <param name="weights">Веса для регуляризации</param>
    /// <param name="gradients">Градиенты для модификации</param>
    private void ApplyWeightDecay(MatrixFloat weights, MatrixFloat gradients)
    {
        var weightData = weights.Data.AsSpan();
        var gradData = gradients.Data.AsSpan();

        // Векторизованное добавление: gradients += weightDecay * weights
        for (int i = 0; i < gradData.Length; i++)
        {
            gradData[i] += _weightDecay * weightData[i];
        }
    }

    /// <summary>
    /// Обновление momentum матрицы.
    /// </summary>
    /// <param name="momentum">Momentum матрица</param>
    /// <param name="gradients">Текущие градиенты</param>
    private void UpdateMomentum(MatrixFloat momentum, MatrixFloat gradients)
    {
        var momentumData = momentum.Data.AsSpan();
        var gradData = gradients.Data.AsSpan();

        // Векторизованное обновление: momentum = β * momentum + (1-β) * gradients
        for (int i = 0; i < momentumData.Length; i++)
        {
            momentumData[i] = _momentum * momentumData[i] + (1 - _momentum) * gradData[i];
        }
    }

    /// <summary>
    /// Применение градиентного обновления к параметрам.
    /// </summary>
    /// <param name="parameters">Параметры для обновления</param>
    /// <param name="momentum">Momentum значения</param>
    private void ApplyGradientUpdate(MatrixFloat parameters, MatrixFloat momentum)
    {
        var paramData = parameters.Data.AsSpan();
        var momentumData = momentum.Data.AsSpan();

        // Векторизованное обновление: parameters = parameters - learningRate * momentum
        for (int i = 0; i < paramData.Length; i++)
        {
            paramData[i] -= _learningRate * momentumData[i];
        }
    }

    /// <summary>
    /// Сброс состояния momentum (используется при изменении learning rate).
    /// </summary>
    public void ResetMomentum()
    {
        foreach (var momentum in _momentumMatrices.Values)
        {
            momentum.Clear();
        }

        foreach (var momentum in _momentumVectors.Values)
        {
            Array.Clear(momentum, 0, momentum.Length);
        }
    }
}
