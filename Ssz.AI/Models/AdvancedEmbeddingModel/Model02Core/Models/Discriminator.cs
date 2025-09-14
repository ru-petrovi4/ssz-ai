using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Models;

/// <summary>
/// Дискриминатор для adversarial обучения в MUSE.
/// Реализует многослойную нейросеть для различения исходных и целевых эмбеддингов.
/// Оптимизирован для максимальной производительности через SIMD и TensorPrimitives.
/// </summary>
public class Discriminator
{
    // Веса и смещения для каждого слоя
    private readonly List<MatrixFloat> _weights;
    private readonly List<float[]> _biases;
    private readonly float _dropoutRate;
    private readonly float _inputDropoutRate;
    private readonly int _embeddingDim;
    private readonly int _hiddenDim;
    private readonly int _numLayers;
    private readonly Random _random;

    /// <summary>
    /// Инициализация дискриминатора с заданными параметрами.
    /// </summary>
    /// <param name="embeddingDim">Размерность входных эмбеддингов</param>
    /// <param name="hiddenDim">Размерность скрытых слоев</param>
    /// <param name="numLayers">Количество скрытых слоев</param>
    /// <param name="dropoutRate">Коэффициент dropout для скрытых слоев</param>
    /// <param name="inputDropoutRate">Коэффициент dropout для входного слоя</param>
    public Discriminator(int embeddingDim, int hiddenDim, int numLayers,
                        float dropoutRate = 0.0f, float inputDropoutRate = 0.0f)
    {
        _embeddingDim = embeddingDim;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _dropoutRate = dropoutRate;
        _inputDropoutRate = inputDropoutRate;
        _random = new Random();

        _weights = new List<MatrixFloat>();
        _biases = new List<float[]>();

        // Инициализация весов и смещений для каждого слоя
        InitializeWeights();
    }

    /// <summary>
    /// Инициализация весов методом Xavier/Glorot для стабильного обучения.
    /// Использует нормальное распределение с дисперсией, зависящей от размера слоя.
    /// </summary>
    private void InitializeWeights()
    {
        for (int layer = 0; layer <= _numLayers; layer++)
        {
            int inputSize = layer == 0 ? _embeddingDim : _hiddenDim;
            int outputSize = layer == _numLayers ? 1 : _hiddenDim;

            // Xavier инициализация: стандартное отклонение = sqrt(2 / (input_size + output_size))
            float stddev = MathF.Sqrt(2.0f / (inputSize + outputSize));

            var weight = new MatrixFloat(new[] { inputSize, outputSize });
            var bias = new float[outputSize];

            // Заполнение весов случайными значениями из нормального распределения
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    weight[i, j] = SampleNormal(0, stddev);
                }
            }

            // Инициализация смещений нулями
            Array.Clear(bias, 0, bias.Length);

            _weights.Add(weight);
            _biases.Add(bias);
        }
    }

    /// <summary>
    /// Генерация случайного числа из нормального распределения методом Бокса-Мюллера.
    /// </summary>
    /// <param name="mean">Среднее значение</param>
    /// <param name="stddev">Стандартное отклонение</param>
    /// <returns>Случайное число из нормального распределения</returns>
    private float SampleNormal(float mean, float stddev)
    {
        // Метод Бокса-Мюллера для генерации нормального распределения
        float u1 = 1.0f - (float)_random.NextDouble(); // uniform(0,1] random doubles
        float u2 = 1.0f - (float)_random.NextDouble();
        float randStdNormal = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Sin(2.0f * MathF.PI * u2);
        return mean + stddev * randStdNormal;
    }

    /// <summary>
    /// Прямое прохождение через дискриминатор с dropout для регуляризации.
    /// Использует LeakyReLU активацию и сигмоид на выходе.
    /// </summary>
    /// <param name="input">Входные эмбеддинги размерностью [batch_size, embedding_dim]</param>
    /// <param name="isTraining">Флаг обучения для применения dropout</param>
    /// <returns>Вероятности принадлежности к целевому языку [batch_size]</returns>
    public float[] Forward(MatrixFloat input, bool isTraining = false)
    {
        if (input.Dimensions[1] != _embeddingDim)
            throw new ArgumentException($"Неверная размерность входа: ожидается {_embeddingDim}, получено {input.Dimensions[1]}");

        int batchSize = input.Dimensions[0];
        var current = input.Clone();

        // Применение входного dropout
        if (isTraining && _inputDropoutRate > 0)
        {
            ApplyDropout(current, _inputDropoutRate);
        }

        // Прохождение через все слои
        for (int layer = 0; layer <= _numLayers; layer++)
        {
            var weights = _weights[layer];
            var biases = _biases[layer];

            int outputSize = weights.Dimensions[1];
            var next = new MatrixFloat(new[] { batchSize, outputSize });

            // Матричное умножение: next = current * weights + bias
            MatrixMultiply(current, weights, next);
            AddBias(next, biases);

            // Применение активации
            if (layer < _numLayers)
            {
                // LeakyReLU для скрытых слоев
                ApplyLeakyReLU(next, 0.2f);

                // Dropout для скрытых слоев
                if (isTraining && _dropoutRate > 0)
                {
                    ApplyDropout(next, _dropoutRate);
                }
            }
            else
            {
                // Sigmoid для выходного слоя
                ApplySigmoid(next);
            }

            current = next;
        }

        // Возврат результата как одномерного массива
        var result = new float[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            result[i] = current[i, 0];
        }

        return result;
    }

    /// <summary>
    /// Высокопроизводительное матричное умножение с использованием TensorPrimitives.
    /// Реализует операцию result = input * weights.
    /// </summary>
    /// <param name="input">Входная матрица [batch_size, input_dim]</param>
    /// <param name="weights">Матрица весов [input_dim, output_dim]</param>
    /// <param name="result">Результирующая матрица [batch_size, output_dim]</param>
    private static void MatrixMultiply(MatrixFloat input, MatrixFloat weights, MatrixFloat result)
    {
        int batchSize = input.Dimensions[0];
        int inputDim = input.Dimensions[1];
        int outputDim = weights.Dimensions[1];

        // Параллельная обработка каждой строки batch'а
        Parallel.For(0, batchSize, i =>
        {
            var inputRow = input.Data.AsSpan(i * inputDim, inputDim);
            var resultRow = result.Data.AsSpan(i * outputDim, outputDim);

            // Вычисление каждого элемента выходной строки
            for (int j = 0; j < outputDim; j++)
            {
                var weightCol = weights.GetColumn(j);
                // Использование SIMD-оптимизированного скалярного произведения
                resultRow[j] = TensorPrimitives.Dot(inputRow, weightCol);
            }
        });
    }

    /// <summary>
    /// Добавление смещения к каждой строке матрицы.
    /// Использует векторизованные операции для максимальной производительности.
    /// </summary>
    /// <param name="matrix">Матрица для добавления смещения</param>
    /// <param name="bias">Вектор смещений</param>
    private static void AddBias(MatrixFloat matrix, float[] bias)
    {
        int batchSize = matrix.Dimensions[0];
        int outputSize = matrix.Dimensions[1];

        Parallel.For(0, batchSize, i =>
        {
            var row = matrix.Data.AsSpan(i * outputSize, outputSize);
            TensorPrimitives.Add(row, bias, row);
        });
    }

    /// <summary>
    /// Применение LeakyReLU активации: f(x) = max(alpha * x, x).
    /// Использует векторизованные операции для максимальной скорости.
    /// </summary>
    /// <param name="matrix">Матрица для применения активации</param>
    /// <param name="alpha">Коэффициент для отрицательных значений</param>
    private static void ApplyLeakyReLU(MatrixFloat matrix, float alpha)
    {
        var data = matrix.Data.AsSpan();

        // Векторизованное применение LeakyReLU
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] < 0)
                data[i] *= alpha;
        }
    }

    /// <summary>
    /// Применение сигмоидной активации: f(x) = 1 / (1 + exp(-x)).
    /// Использует быстрое приближение для exp через TensorPrimitives.
    /// </summary>
    /// <param name="matrix">Матрица для применения активации</param>
    private static void ApplySigmoid(MatrixFloat matrix)
    {
        var data = matrix.Data.AsSpan();

        // Применение отрицания для последующего exp
        TensorPrimitives.Negate(data, data);

        // Быстрое вычисление экспоненты
        TensorPrimitives.Exp(data, data);

        // Добавление 1 и взятие обратного значения: 1 / (1 + exp(-x))
        TensorPrimitives.Add(data, 1.0f, data);

        // Вычисление обратного значения для получения sigmoid
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = 1.0f / data[i];
        }
    }

    /// <summary>
    /// Применение dropout для регуляризации во время обучения.
    /// Случайно обнуляет элементы с заданной вероятностью и масштабирует остальные.
    /// </summary>
    /// <param name="matrix">Матрица для применения dropout</param>
    /// <param name="dropoutRate">Вероятность обнуления элемента (0.0 - 1.0)</param>
    private void ApplyDropout(MatrixFloat matrix, float dropoutRate)
    {
        if (dropoutRate <= 0.0f) return;

        float scale = 1.0f / (1.0f - dropoutRate); // Масштабирование для сохранения ожидаемого значения
        var data = matrix.Data.AsSpan();

        for (int i = 0; i < data.Length; i++)
        {
            if (_random.NextSingle() < dropoutRate)
            {
                data[i] = 0.0f; // Обнуление элемента
            }
            else
            {
                data[i] *= scale; // Масштабирование для компенсации
            }
        }
    }

    /// <summary>
    /// Получение весов определенного слоя для оптимизации.
    /// </summary>
    /// <param name="layer">Номер слоя</param>
    /// <returns>Матрица весов слоя</returns>
    public MatrixFloat GetWeights(int layer) => _weights[layer];

    /// <summary>
    /// Получение смещений определенного слоя для оптимизации.
    /// </summary>
    /// <param name="layer">Номер слоя</param>
    /// <returns>Массив смещений слоя</returns>
    public float[] GetBiases(int layer) => _biases[layer];

    /// <summary>
    /// Получение общего количества слоев.
    /// </summary>
    public int LayerCount => _weights.Count;
}