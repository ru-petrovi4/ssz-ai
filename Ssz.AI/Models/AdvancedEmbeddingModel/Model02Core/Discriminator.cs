using System;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Models;

/// <summary>
/// Параметры для создания дискриминатора
/// </summary>
public interface IDiscriminatorParameters
{
    /// <summary>
    /// Размерность эмбеддингов
    /// </summary>
    int EmbDim { get; }

    /// <summary>
    /// Количество скрытых слоев дискриминатора
    /// </summary>
    int DisHidLayers { get; }

    /// <summary>
    /// Размерность скрытых слоев
    /// </summary>
    int DisHidDim { get; }

    /// <summary>
    /// Dropout для скрытых слоев
    /// </summary>
    double DisDropout { get; }

    /// <summary>
    /// Input dropout
    /// </summary>
    double DisInputDropout { get; }
}

/// <summary>
/// Дискриминатор для состязательного обучения эмбеддингов
/// Аналог Discriminator класса из models.py с оптимизациями для .NET 9
/// </summary>
public sealed class Discriminator : Module<Tensor, Tensor>
{
    #region Private Fields

    /// <summary>
    /// Последовательность слоев дискриминатора
    /// </summary>
    private readonly Sequential _layers;

    /// <summary>
    /// Параметры дискриминатора
    /// </summary>
    private readonly IDiscriminatorParameters _discriminatorParameters;

    #endregion

    #region Constructor

    /// <summary>
    /// Инициализирует новый экземпляр дискриминатора
    /// </summary>
    /// <param name="discriminatorParameters">Параметры дискриминатора</param>
    /// <param name="name">Имя модуля</param>
    /// <exception cref="ArgumentNullException">Если параметры равны null</exception>
    public Discriminator(IDiscriminatorParameters discriminatorParameters, string name = "discriminator") : base(name)
    {
        _discriminatorParameters = discriminatorParameters ?? throw new ArgumentNullException(nameof(discriminatorParameters));

        // Строим архитектуру дискриминатора
        var layers = new List<Module<Tensor, Tensor>>();

        // Input dropout слой
        if (_discriminatorParameters.DisInputDropout > 0)
        {
            layers.Add(Dropout(_discriminatorParameters.DisInputDropout));
        }

        // Строим скрытые слои
        for (int i = 0; i <= _discriminatorParameters.DisHidLayers; i++)
        {
            int inputDim = i == 0 ? _discriminatorParameters.EmbDim : _discriminatorParameters.DisHidDim;
            int outputDim = i == _discriminatorParameters.DisHidLayers ? 1 : _discriminatorParameters.DisHidDim;

            // Линейный слой
            layers.Add(Linear(inputDim, outputDim));

            // Активация и dropout для всех слоев кроме последнего
            if (i < _discriminatorParameters.DisHidLayers)
            {
                layers.Add(LeakyReLU(0.2)); // Используем LeakyReLU как в оригинале

                if (_discriminatorParameters.DisDropout > 0)
                {
                    layers.Add(Dropout(_discriminatorParameters.DisDropout));
                }
            }
        }

        // Финальная сигмоидная активация
        layers.Add(Sigmoid());

        // Создаем последовательную модель
        _layers = Sequential(layers.ToArray());

        // Регистрируем подмодуль
        RegisterComponents();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Прямой проход через дискриминатор
    /// </summary>
    /// <param name="input">Входной тензор эмбеддингов [batch_size, embedding_dim]</param>
    /// <returns>Вероятности принадлежности к целевому языку [batch_size]</returns>
    /// <exception cref="ArgumentException">Если размерности входа некорректны</exception>
    public override Tensor forward(Tensor input)
    {
        // Проверяем размерности входного тензора
        if (input.dim() != 2)
            throw new ArgumentException($"Ожидается 2D тензор, получен {input.dim()}D", nameof(input));

        if (input.size(1) != _discriminatorParameters.EmbDim)
            throw new ArgumentException(
                $"Ожидается размерность эмбеддингов {_discriminatorParameters.EmbDim}, получена {input.size(1)}",
                nameof(input));

        // Пропускаем через последовательность слоев
        var output = _layers.forward(input);

        // Возвращаем одномерный тензор вероятностей
        return output.view(-1);
    }

    /// <summary>
    /// Инициализирует веса дискриминатора
    /// </summary>
    /// <param name="seed">Seed для генератора случайных чисел</param>
    public void InitializeWeights(int? seed = null)
    {
        if (seed.HasValue)
        {
            manual_seed(seed.Value);
        }

        // Применяем Xavier/Glorot инициализацию к линейным слоям
        foreach (var module in _layers.modules())
        {
            if (module is Linear linear)
            {
                init.xavier_uniform_(linear.weight!);
                if (linear.bias is not null)
                {
                    init.zeros_(linear.bias);
                }
            }
        }
    }

    /// <summary>
    /// Обрезает градиенты для стабильности обучения
    /// </summary>
    /// <param name="maxNorm">Максимальная норма градиентов</param>
    public void ClipGradients(double maxNorm)
    {
        if (maxNorm > 0)
        {
            nn.utils.clip_grad_norm_(parameters(), maxNorm);
        }
    }

    /// <summary>
    /// Получает информацию об архитектуре дискриминатора
    /// </summary>
    /// <returns>Строковое описание архитектуры</returns>
    public string GetArchitectureInfo()
    {
        var totalParams = 0L;
        var trainableParams = 0L;

        foreach (var param in parameters())
        {
            var paramCount = param.numel();
            totalParams += paramCount;
            if (param.requires_grad)
            {
                trainableParams += paramCount;
            }
        }

        return $"Discriminator Architecture:\n" +
               $"  Embedding Dimension: {_discriminatorParameters.EmbDim}\n" +
               $"  Hidden Layers: {_discriminatorParameters.DisHidLayers}\n" +
               $"  Hidden Dimension: {_discriminatorParameters.DisHidDim}\n" +
               $"  Dropout: {_discriminatorParameters.DisDropout}\n" +
               $"  Input Dropout: {_discriminatorParameters.DisInputDropout}\n" +
               $"  Total Parameters: {totalParams:N0}\n" +
               $"  Trainable Parameters: {trainableParams:N0}";
    }

    #endregion

    #region Protected Methods

    /// <summary>
    /// Освобождает ресурсы
    /// </summary>
    /// <param name="disposing">Признак освобождения управляемых ресурсов</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _layers?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}