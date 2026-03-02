// LISSOM Model Implementation in C# (.NET 10)
// Based on: "A Global Orientation Map in the Primary Visual Cortex (V1)"
// Authors: Ryan T Philips & V Srinivasa Chakravarthy (2017)

using System;
using System.Linq;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace Ssz.AI.Models.ImageProcessingModel.LISSOMModel;

/// <summary>
/// Параметры модели LISSOM
/// </summary>
public class LISSOMParameters
{
    // Коэффициенты масштабирования
    public double P { get; set; } = 1.05;    // Афферентный вклад
    public double Q { get; set; } = 2.3;     // Возбуждающий вклад
    public double R { get; set; } = 2.45;    // Тормозящий вклад

    // Скорости обучения
    public double EtaA { get; set; } = 0.5;   // Афферентные связи
    public double EtaE { get; set; } = 0.3;   // Возбуждающие связи
    public double EtaI { get; set; } = 0.11;  // Тормозящие связи

    // Радиусы связей
    public double RadA { get; set; } = 1.0;   // Афферентный радиус
    public double RadE { get; set; } = 0.03;  // Возбуждающий радиус
    public double RadI { get; set; } = 0.55;  // Тормозящий радиус

    // Параметры функции активации (кусочно-линейная сигмоида)
    public double AlphaL { get; set; } = 0.1;   // Нижний порог
    public double AlphaU { get; set; } = 0.65;  // Верхний порог

    // Размеры слоёв
    public int RetinaWidth { get; set; } = 64;
    public int RetinaHeight { get; set; } = 64;
    public int V1Width { get; set; } = 48;
    public int V1Height { get; set; } = 48;

    // Параметры обучения
    public int SettlingIterations { get; set; } = 20;  // Итераций для установления активности
    public int TrainingEpochs { get; set; } = 20000;   // Эпох обучения
}

/// <summary>
/// Главный класс модели LISSOM
/// </summary>
public class LISSOMNetwork
{
    private readonly LISSOMParameters _params;
    private readonly Random _random;

    // Весовые матрицы
    private Matrix<double>[,] _afferentWeights;    // A[i,j][a,b]
    private Matrix<double>[,] _excitatoryWeights;  // E[i,j][k,l]
    private Matrix<double>[,] _inhibitoryWeights;  // I[i,j][k,l]

    // Слои активности
    private Matrix<double> _retinaActivity;
    private Matrix<double> _v1Activity;
    private Matrix<double> _v1ActivityPrev;

    public LISSOMNetwork(LISSOMParameters parameters)
    {
        _params = parameters;
        _random = new Random(42); // Фиксированный seed для воспроизводимости
        
        InitializeWeights();
        InitializeActivityLayers();
    }

    /// <summary>
    /// Инициализация весовых матриц
    /// </summary>
    private void InitializeWeights()
    {
        _afferentWeights = new Matrix<double>[_params.V1Height, _params.V1Width];
        _excitatoryWeights = new Matrix<double>[_params.V1Height, _params.V1Width];
        _inhibitoryWeights = new Matrix<double>[_params.V1Height, _params.V1Width];

        // Инициализация для каждого нейрона V1
        for (int i = 0; i < _params.V1Height; i++)
        {
            for (int j = 0; j < _params.V1Width; j++)
            {
                // Афферентные веса (от сетчатки к V1)
                _afferentWeights[i, j] = CreateGaussianWeightMatrix(
                    _params.RetinaHeight, 
                    _params.RetinaWidth,
                    MapV1ToRetina(i, j, _params.V1Height, _params.V1Width, 
                                 _params.RetinaHeight, _params.RetinaWidth),
                    _params.RadA
                );

                // Возбуждающие латеральные веса
                _excitatoryWeights[i, j] = CreateGaussianWeightMatrix(
                    _params.V1Height,
                    _params.V1Width,
                    (i, j),
                    _params.RadE
                );

                // Тормозящие латеральные веса
                _inhibitoryWeights[i, j] = CreateGaussianWeightMatrix(
                    _params.V1Height,
                    _params.V1Width,
                    (i, j),
                    _params.RadI
                );
            }
        }
    }

    /// <summary>
    /// Создание весовой матрицы с гауссовым распределением
    /// </summary>
    private Matrix<double> CreateGaussianWeightMatrix(int height, int width, 
        (int row, int col) center, double radius)
    {
        var weights = Matrix<double>.Build.Dense(height, width);
        
        for (int a = 0; a < height; a++)
        {
            for (int b = 0; b < width; b++)
            {
                double distance = Math.Sqrt(
                    Math.Pow(a - center.row, 2) + 
                    Math.Pow(b - center.col, 2)
                );
                
                // Гауссово распределение
                double weight = Math.Exp(-Math.Pow(distance / radius, 2) / 2.0);
                
                // Небольшая случайная инициализация
                weight += (_random.NextDouble() - 0.5) * 0.1;
                weight = Math.Max(0, weight);
                
                weights[a, b] = weight;
            }
        }

        // Нормализация
        double sum = weights.RowSums().Sum();
        if (sum > 0)
            weights = weights / sum;

        return weights;
    }

    /// <summary>
    /// Отображение координат V1 на координаты сетчатки
    /// </summary>
    private (int, int) MapV1ToRetina(int v1_i, int v1_j, 
        int v1Height, int v1Width, int retinaHeight, int retinaWidth)
    {
        int retina_i = (int)(v1_i * (double)retinaHeight / v1Height);
        int retina_j = (int)(v1_j * (double)retinaWidth / v1Width);
        return (retina_i, retina_j);
    }

    /// <summary>
    /// Инициализация слоёв активности
    /// </summary>
    private void InitializeActivityLayers()
    {
        _retinaActivity = Matrix<double>.Build.Dense(_params.RetinaHeight, _params.RetinaWidth);
        _v1Activity = Matrix<double>.Build.Dense(_params.V1Height, _params.V1Width);
        _v1ActivityPrev = Matrix<double>.Build.Dense(_params.V1Height, _params.V1Width);
    }

    /// <summary>
    /// Кусочно-линейная функция активации (аппроксимация сигмоиды)
    /// </summary>
    private double ActivationFunction(double s)
    {
        if (s <= _params.AlphaL)
            return 0.0;
        else if (s >= _params.AlphaU)
            return 1.0;
        else
            return (s - _params.AlphaL) / (_params.AlphaU - _params.AlphaL);
    }

    /// <summary>
    /// Применение функции активации к матрице
    /// </summary>
    private Matrix<double> ApplyActivation(Matrix<double> input)
    {
        return input.Map(ActivationFunction);
    }

    /// <summary>
    /// Вычисление начальной активности V1 (только афферентные входы)
    /// </summary>
    private void ComputeInitialV1Activity()
    {
        for (int i = 0; i < _params.V1Height; i++)
        {
            for (int j = 0; j < _params.V1Width; j++)
            {
                // y_ij = g(∑_{a,b} A_{ij,ab} × x_{ab})
                double afferentInput = (_afferentWeights[i, j].PointwiseMultiply(_retinaActivity))
                    .RowSums().Sum();
                
                _v1Activity[i, j] = ActivationFunction(afferentInput);
            }
        }
    }

    /// <summary>
    /// Вычисление активности V1 с латеральными связями
    /// </summary>
    private void ComputeV1ActivityWithLateral()
    {
        var newActivity = Matrix<double>.Build.Dense(_params.V1Height, _params.V1Width);

        for (int i = 0; i < _params.V1Height; i++)
        {
            for (int j = 0; j < _params.V1Width; j++)
            {
                // Афферентный вход
                double afferentInput = (_afferentWeights[i, j].PointwiseMultiply(_retinaActivity))
                    .RowSums().Sum();

                // Возбуждающий латеральный вход
                double excitatoryInput = (_excitatoryWeights[i, j].PointwiseMultiply(_v1ActivityPrev))
                    .RowSums().Sum();

                // Тормозящий латеральный вход
                double inhibitoryInput = (_inhibitoryWeights[i, j].PointwiseMultiply(_v1ActivityPrev))
                    .RowSums().Sum();

                // y_ij(t) = g(p*afferent + q*excitatory - r*inhibitory)
                double totalInput = _params.P * afferentInput + 
                                   _params.Q * excitatoryInput - 
                                   _params.R * inhibitoryInput;

                newActivity[i, j] = ActivationFunction(totalInput);
            }
        }

        _v1Activity = newActivity;
    }

    /// <summary>
    /// Установление активности (settling) - итеративное вычисление до сходимости
    /// </summary>
    private void SettleActivity()
    {
        // Начальная активность
        ComputeInitialV1Activity();

        // Итерации с латеральными связями
        for (int iter = 0; iter < _params.SettlingIterations; iter++)
        {
            _v1ActivityPrev = _v1Activity.Clone();
            ComputeV1ActivityWithLateral();

            // Проверка сходимости (опционально)
            double diff = (_v1Activity - _v1ActivityPrev).L2Norm();
            if (diff < 1e-6)
                break;
        }
    }

    /// <summary>
    /// Обновление весов по нормализованному хеббианскому правилу
    /// </summary>
    private void UpdateWeights()
    {
        // Обновление афферентных весов
        for (int i = 0; i < _params.V1Height; i++)
        {
            for (int j = 0; j < _params.V1Width; j++)
            {
                double postsynaptic = _v1Activity[i, j];
                
                // w_{ij,mn}(t+1) = (w + η*y*P) / sum(w + η*y*P)
                var deltaW = _retinaActivity * (_params.EtaA * postsynaptic);
                var newWeights = _afferentWeights[i, j] + deltaW;
                
                // Нормализация
                double sum = newWeights.RowSums().Sum();
                if (sum > 0)
                    _afferentWeights[i, j] = newWeights / sum;
            }
        }

        // Обновление возбуждающих весов
        for (int i = 0; i < _params.V1Height; i++)
        {
            for (int j = 0; j < _params.V1Width; j++)
            {
                double postsynaptic = _v1Activity[i, j];
                
                var deltaW = _v1Activity * (_params.EtaE * postsynaptic);
                var newWeights = _excitatoryWeights[i, j] + deltaW;
                
                double sum = newWeights.RowSums().Sum();
                if (sum > 0)
                    _excitatoryWeights[i, j] = newWeights / sum;
            }
        }

        // Обновление тормозящих весов
        for (int i = 0; i < _params.V1Height; i++)
        {
            for (int j = 0; j < _params.V1Width; j++)
            {
                double postsynaptic = _v1Activity[i, j];
                
                var deltaW = _v1Activity * (_params.EtaI * postsynaptic);
                var newWeights = _inhibitoryWeights[i, j] + deltaW;
                
                double sum = newWeights.RowSums().Sum();
                if (sum > 0)
                    _inhibitoryWeights[i, j] = newWeights / sum;
            }
        }
    }

    /// <summary>
    /// Предъявление стимула сетчатке
    /// </summary>
    public void PresentStimulus(Matrix<double> stimulus)
    {
        if (stimulus.RowCount != _params.RetinaHeight || 
            stimulus.ColumnCount != _params.RetinaWidth)
        {
            throw new ArgumentException(
                $"Stimulus size must be {_params.RetinaHeight}x{_params.RetinaWidth}"
            );
        }

        _retinaActivity = stimulus.Clone();
    }

    /// <summary>
    /// Один шаг обучения
    /// </summary>
    public void TrainStep(Matrix<double> stimulus)
    {
        PresentStimulus(stimulus);
        SettleActivity();
        UpdateWeights();
    }

    /// <summary>
    /// Полное обучение модели
    /// </summary>
    public void Train(Func<Matrix<double>> stimulusGenerator, 
        Action<int, double> progressCallback = null)
    {
        for (int epoch = 0; epoch < _params.TrainingEpochs; epoch++)
        {
            var stimulus = stimulusGenerator();
            TrainStep(stimulus);

            // Вызов callback для отслеживания прогресса
            if (progressCallback != null && epoch % 100 == 0)
            {
                double avgActivity = _v1Activity.RowSums().Sum() / 
                                    (_params.V1Height * _params.V1Width);
                progressCallback(epoch, avgActivity);
            }
        }
    }

    /// <summary>
    /// Тестирование модели (без обучения)
    /// </summary>
    public Matrix<double> Test(Matrix<double> stimulus)
    {
        PresentStimulus(stimulus);
        SettleActivity();
        return _v1Activity.Clone();
    }

    /// <summary>
    /// Получение текущей активности V1
    /// </summary>
    public Matrix<double> GetV1Activity() => _v1Activity.Clone();

    /// <summary>
    /// Получение карты предпочтений для конкретного признака
    /// </summary>
    public Matrix<double> GetPreferenceMap(Func<int, int, Matrix<double>> stimulusGenerator,
        int numOrientations)
    {
        var preferenceMap = Matrix<double>.Build.Dense(_params.V1Height, _params.V1Width);
        var maxResponseMap = Matrix<double>.Build.Dense(_params.V1Height, _params.V1Width);

        // Тестируем каждую ориентацию
        for (int orientation = 0; orientation < numOrientations; orientation++)
        {
            double angle = 2.0 * Math.PI * orientation / numOrientations;
            
            for (int i = 0; i < _params.V1Height; i++)
            {
                for (int j = 0; j < _params.V1Width; j++)
                {
                    var stimulus = stimulusGenerator(i, j);
                    var response = Test(stimulus);
                    double currentResponse = response[i, j];

                    if (currentResponse > maxResponseMap[i, j])
                    {
                        maxResponseMap[i, j] = currentResponse;
                        preferenceMap[i, j] = angle;
                    }
                }
            }
        }

        return preferenceMap;
    }
}

/// <summary>
/// Генератор стимулов для обучения
/// </summary>
public class StimulusGenerator
{
    private readonly Random _random;
    private readonly int _width;
    private readonly int _height;

    public StimulusGenerator(int width, int height, int seed = 42)
    {
        _random = new Random(seed);
        _width = width;
        _height = height;
    }

    /// <summary>
    /// Генерация прямоугольной полосы (как в статье)
    /// Aspect ratio = 0.025 (высоко вытянутые, близкие к радиальным)
    /// </summary>
    public Matrix<double> GenerateRectangularBar(double? angle = null, 
        double? size = null, double? centerX = null, double? centerY = null)
    {
        var stimulus = Matrix<double>.Build.Dense(_height, _width);

        // Случайные параметры, если не указаны
        double theta = angle ?? _random.NextDouble() * 2 * Math.PI;
        double length = size ?? (_random.NextDouble() * 3.67 + 0.33); // от 0.33° до 4°
        double cx = centerX ?? _width / 2.0;
        double cy = centerY ?? _height / 2.0;

        double aspectRatio = 0.025; // Из статьи
        double width = length * aspectRatio;

        // Создание полосы
        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                // Поворот координат
                double dx = x - cx;
                double dy = y - cy;
                double rotX = dx * Math.Cos(-theta) - dy * Math.Sin(-theta);
                double rotY = dx * Math.Sin(-theta) + dy * Math.Cos(-theta);

                // Проверка попадания в прямоугольник
                if (Math.Abs(rotX) <= length / 2.0 && Math.Abs(rotY) <= width / 2.0)
                {
                    stimulus[y, x] = 1.0;
                }
            }
        }

        return stimulus;
    }

    /// <summary>
    /// Генерация радиальной полосы (направленной от центра)
    /// </summary>
    public Matrix<double> GenerateRadialBar(int v1_i, int v1_j, 
        int v1Height, int v1Width)
    {
        // Вычисление меридионального угла
        double cx = _width / 2.0;
        double cy = _height / 2.0;
        
        double px = (v1_j / (double)v1Width) * _width;
        double py = (v1_i / (double)v1Height) * _height;
        
        double angle = Math.Atan2(py - cy, px - cx);

        return GenerateRectangularBar(angle: angle, centerX: px, centerY: py);
    }

    /// <summary>
    /// Генерация синусоидальной решётки
    /// </summary>
    public Matrix<double> GenerateSinusoidalGrating(double orientation, 
        double frequency, double phase = 0.0)
    {
        var stimulus = Matrix<double>.Build.Dense(_height, _width);

        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                double distance = x * Math.Cos(orientation) + y * Math.Sin(orientation);
                double value = (Math.Sin(2 * Math.PI * frequency * distance + phase) + 1) / 2.0;
                stimulus[y, x] = value;
            }
        }

        return stimulus;
    }

    /// <summary>
    /// Генерация кольцевого стимула
    /// </summary>
    public Matrix<double> GenerateAnnularStimulus(double innerRadius, 
        double outerRadius, double frequency, double orientation)
    {
        var stimulus = Matrix<double>.Build.Dense(_height, _width);
        double cx = _width / 2.0;
        double cy = _height / 2.0;

        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                double dx = x - cx;
                double dy = y - cy;
                double r = Math.Sqrt(dx * dx + dy * dy);

                if (r >= innerRadius && r <= outerRadius)
                {
                    // Синусоидальная решётка внутри кольца
                    double distance = x * Math.Cos(orientation) + y * Math.Sin(orientation);
                    double value = (Math.Sin(2 * Math.PI * frequency * distance) + 1) / 2.0;
                    stimulus[y, x] = value;
                }
            }
        }

        return stimulus;
    }
}

/// <summary>
/// Анализ результатов модели
/// </summary>
public class LISSOMAnalyzer
{
    /// <summary>
    /// Вычисление круговой кросс-корреляции между двумя картами
    /// </summary>
    public static (double correlation, double shift) CircularCrossCorrelation(
        Matrix<double> orientationMap, Matrix<double> meridionalMap)
    {
        int height = orientationMap.RowCount;
        int width = orientationMap.ColumnCount;

        // Вычисление круговых средних
        double oMean = CircularMean(orientationMap);
        double mMean = CircularMean(meridionalMap);

        // R(o-m) и R(o+m)
        double rDiff = 0.0;
        double rSum = 0.0;
        double sumSin2O = 0.0;
        double sumSin2M = 0.0;

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                double o = orientationMap[i, j];
                double m = meridionalMap[i, j];

                rDiff += Math.Sin(o - m);
                rSum += Math.Sin(o + m);
                sumSin2O += Math.Pow(Math.Sin(o - oMean), 2);
                sumSin2M += Math.Pow(Math.Sin(m - mMean), 2);
            }
        }

        // r_c = [R(o-m) - R(o+m)] / [2√(Σsin²(o-ō) × Σsin²(m-m̄))]
        double denominator = 2.0 * Math.Sqrt(sumSin2O * sumSin2M);
        double rc = denominator > 0 ? (rDiff - rSum) / denominator : 0.0;

        // Вычисление сдвига
        double shift = Math.Atan2(rSum, rDiff);

        return (rc, shift);
    }

    /// <summary>
    /// Вычисление кругового среднего
    /// </summary>
    private static double CircularMean(Matrix<double> angles)
    {
        double sumSin = 0.0;
        double sumCos = 0.0;

        for (int i = 0; i < angles.RowCount; i++)
        {
            for (int j = 0; j < angles.ColumnCount; j++)
            {
                sumSin += Math.Sin(angles[i, j]);
                sumCos += Math.Cos(angles[i, j]);
            }
        }

        return Math.Atan2(sumSin, sumCos);
    }

    /// <summary>
    /// Сохранение карты в текстовый файл
    /// </summary>
    public static void SaveMapToFile(Matrix<double> map, string filename)
    {
        using (var writer = new System.IO.StreamWriter(filename))
        {
            for (int i = 0; i < map.RowCount; i++)
            {
                for (int j = 0; j < map.ColumnCount; j++)
                {
                    writer.Write($"{map[i, j]:F6}\t");
                }
                writer.WriteLine();
            }
        }
    }
}

/// <summary>
/// Пример использования
/// </summary>
public class TestProgram
{
    public static void MainTest(string[] args)
    {
        Console.WriteLine("LISSOM Model - Radial Bias Simulation");
        Console.WriteLine("=====================================\n");

        // Создание параметров модели
        var parameters = new LISSOMParameters
        {
            RetinaWidth = 64,
            RetinaHeight = 64,
            V1Width = 48,
            V1Height = 48,
            TrainingEpochs = 10000,
            SettlingIterations = 20
        };

        // Создание модели
        var model = new LISSOMNetwork(parameters);
        var stimGen = new StimulusGenerator(parameters.RetinaWidth, parameters.RetinaHeight);

        Console.WriteLine("Начало обучения модели...");
        Console.WriteLine($"Эпохи обучения: {parameters.TrainingEpochs}");
        Console.WriteLine($"Размер сетчатки: {parameters.RetinaHeight}x{parameters.RetinaWidth}");
        Console.WriteLine($"Размер V1: {parameters.V1Height}x{parameters.V1Width}\n");

        // Обучение модели на радиальных полосах
        model.Train(
            stimulusGenerator: () => stimGen.GenerateRectangularBar(),
            progressCallback: (epoch, avgActivity) =>
            {
                Console.WriteLine($"Эпоха {epoch}: Средняя активность V1 = {avgActivity:F4}");
            }
        );

        Console.WriteLine("\nОбучение завершено!");
        Console.WriteLine("\nТестирование модели...\n");

        // Тестирование на различных ориентациях
        int numOrientations = 12;
        Console.WriteLine($"Тестирование на {numOrientations} ориентациях");

        // Получение карты предпочтений ориентации
        var orientationMap = model.GetPreferenceMap(
            (i, j) => stimGen.GenerateSinusoidalGrating(
                orientation: 0.0, 
                frequency: 0.5
            ),
            numOrientations
        );

        // Получение карты меридиональных углов
        var meridionalMap = model.GetPreferenceMap(
            (i, j) => stimGen.GenerateRadialBar(i, j, parameters.V1Height, parameters.V1Width),
            numOrientations
        );

        // Анализ корреляции
        var (rc, shift) = LISSOMAnalyzer.CircularCrossCorrelation(
            orientationMap, 
            meridionalMap
        );

        Console.WriteLine($"\nРезультаты анализа:");
        Console.WriteLine($"Круговая кросс-корреляция r_c = {rc:F4}");
        Console.WriteLine($"Сдвиг Δ_shift = {shift * 180 / Math.PI:F2}°");
        Console.WriteLine($"\nОжидаемые значения (Freeman et al., 2011):");
        Console.WriteLine($"r_c ≈ 0.52, Δ_shift ≈ 1°");

        // Сохранение результатов
        LISSOMAnalyzer.SaveMapToFile(orientationMap, "orientation_map.txt");
        LISSOMAnalyzer.SaveMapToFile(meridionalMap, "meridional_map.txt");

        Console.WriteLine("\nКарты сохранены в файлы:");
        Console.WriteLine("- orientation_map.txt");
        Console.WriteLine("- meridional_map.txt");
    }
}
