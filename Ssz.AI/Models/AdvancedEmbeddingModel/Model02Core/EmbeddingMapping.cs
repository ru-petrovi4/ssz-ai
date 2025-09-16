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
/// Параметры для создания модели выравнивания
/// </summary>
public interface IMappingParameters
{
    /// <summary>
    /// Размерность эмбеддингов
    /// </summary>
    int EmbDim { get; }

    /// <summary>
    /// Инициализировать как единичную матрицу
    /// </summary>
    bool MapIdInit { get; }

    /// <summary>
    /// Параметр для ортогонализации (beta)
    /// </summary>
    double MapBeta { get; }

    /// <summary>
    /// Использовать ли bias в линейном преобразовании
    /// </summary>
    bool UseBias { get; }
}

/// <summary>
/// Модель линейного преобразования для выравнивания эмбеддингов
/// Аналог mapping из models.py с поддержкой ортогонализации
/// </summary>
public sealed class EmbeddingMapping : Module<Tensor, Tensor>
{
    #region Private Fields

    /// <summary>
    /// Линейное преобразование
    /// </summary>
    private readonly Linear _mappingLinear;

    /// <summary>
    /// Параметры модели
    /// </summary>
    private readonly IMappingParameters _mappingParameters;

    #endregion

    #region Constructor

    /// <summary>
    /// Инициализирует новую модель выравнивания эмбеддингов
    /// </summary>
    /// <param name="mappingParameters">Параметры модели</param>
    /// <param name="name">Имя модуля</param>
    /// <exception cref="ArgumentNullException">Если параметры равны null</exception>
    public EmbeddingMapping(IMappingParameters mappingParameters, string name = "mapping") : base(name)
    {
        _mappingParameters = mappingParameters ?? throw new ArgumentNullException(nameof(mappingParameters));

        // Создаем линейное преобразование
        _mappingLinear = Linear(_mappingParameters.EmbDim, _mappingParameters.EmbDim, _mappingParameters.UseBias);

        // Инициализируем веса
        InitializeWeights();

        // Регистрируем компоненты
        RegisterComponents();
    }

    #endregion

    #region Public Properties        

    /// <summary>
    /// Веса преобразования
    /// </summary>
    public Tensor Weight => _mappingLinear.weight!;

    #endregion

    #region Public Methods

    /// <summary>
    /// Прямой проход через модель выравнивания
    /// </summary>
    /// <param name="input">Входные эмбеддинги [batch_size, embedding_dim] или [embedding_dim]</param>
    /// <returns>Преобразованные эмбеддинги</returns>
    /// <exception cref="ArgumentException">Если размерности некорректны</exception>
    public override Tensor forward(Tensor input)
    {
        // Проверяем размерности
        var inputDim = input.dim();
        var lastDimSize = input.size(-1);

        if (lastDimSize != _mappingParameters.EmbDim)
            throw new ArgumentException(
                $"Ожидается размерность эмбеддингов {_mappingParameters.EmbDim}, получена {lastDimSize}",
                nameof(input));

        return _mappingLinear.forward(input);
    }

    /// <summary>
    /// Применяет решение задачи ортогонального Прокруста
    /// Находит оптимальную ортогональную матрицу для выравнивания эмбеддингов
    /// </summary>
    /// <param name="sourceEmbeddings">Исходные эмбеддинги [n_pairs, embedding_dim]</param>
    /// <param name="targetEmbeddings">Целевые эмбеддинги [n_pairs, embedding_dim]</param>
    /// <exception cref="ArgumentException">Если размерности не совпадают</exception>
    public void ApplyProcrustesAlignment(Tensor sourceEmbeddings, Tensor targetEmbeddings)
    {
        if (sourceEmbeddings.shape[0] != targetEmbeddings.shape[0])
            throw new ArgumentException("Количество исходных и целевых эмбеддингов должно совпадать");

        if (sourceEmbeddings.shape[1] != _mappingParameters.EmbDim ||
            targetEmbeddings.shape[1] != _mappingParameters.EmbDim)
            throw new ArgumentException("Размерности эмбеддингов должны соответствовать параметрам модели");

        using var _ = no_grad();

        // Вычисляем матрицу M = B^T * A (где A - source, B - target)
        var M = targetEmbeddings.transpose(0, 1).mm(sourceEmbeddings);

        // Применяем SVD разложение: M = U * S * V^T
        var (U, S, Vt) = linalg.svd(M, fullMatrices: true);

        // Оптимальная ортогональная матрица: W = U * V^T
        var optimalW = U.mm(Vt);

        // Обновляем веса модели
        _mappingLinear.weight!.copy_(optimalW);
    }

    /// <summary>
    /// Применяет ортогонализацию к весам модели
    /// Использует итеративную процедуру для поддержания ортогональности
    /// </summary>
    public void OrthogonalizeWeights()
    {
        if (_mappingParameters.MapBeta <= 0)
            return;

        using var _ = no_grad();

        var W = _mappingLinear.weight!;
        var beta = _mappingParameters.MapBeta;

        // Применяем формулу: W = (1 + β) * W - β * W * W^T * W
        var WtW = W.transpose(0, 1).mm(W);
        var WWtW = W.mm(WtW);

        var newW = W.mul(1 + beta).sub(WWtW.mul(beta));
        W.copy_(newW);
    }

    /// <summary>
    /// Инициализирует веса модели
    /// </summary>
    /// <param name="seed">Seed для генератора случайных чисел</param>
    public void InitializeWeights(int? seed = null)
    {
        if (seed.HasValue)
        {
            manual_seed(seed.Value);
        }

        using var _ = no_grad();

        if (_mappingParameters.MapIdInit)
        {
            // Инициализируем как единичную матрицу
            init.eye_(_mappingLinear.weight!);
        }
        else
        {
            // Используем ортогональную инициализацию
            init.orthogonal_(_mappingLinear.weight!);
        }

        // Инициализируем bias нулями (если используется)
        if (_mappingLinear.bias is not null)
        {
            init.zeros_(_mappingLinear.bias);
        }
    }

    /// <summary>
    /// Проверяет ортогональность матрицы преобразования
    /// </summary>
    /// <param name="tolerance">Допустимая погрешность</param>
    /// <returns>Степень ортогональности (0 = идеально ортогональная)</returns>
    public double CheckOrthogonality(double tolerance = 1e-6)
    {
        using var _ = no_grad();

        var W = _mappingLinear.weight!;
        var WtW = W.transpose(0, 1).mm(W);
        var identity = eye(W.size(0), device: W.device, dtype: W.dtype);

        var orthogonalityError = (WtW - identity).norm().item<float>(); // VALFIX
        return orthogonalityError;
    }

    /// <summary>
    /// Получает информацию о модели
    /// </summary>
    /// <returns>Строковое описание модели</returns>
    public string GetModelInfo()
    {
        var paramCount = _mappingLinear.weight!.numel();
        var orthogonalityError = CheckOrthogonality();

        return $"Embedding Mapping Model:\n" +
               $"  Dimension: {_mappingParameters.EmbDim}x{_mappingParameters.EmbDim}\n" +
               $"  Parameters: {paramCount:N0}\n" +
               $"  Use Bias: {_mappingParameters.UseBias}\n" +
               $"  Orthogonalization Beta: {_mappingParameters.MapBeta}\n" +
               $"  Current Orthogonality Error: {orthogonalityError:E6}";
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
            _mappingLinear?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
