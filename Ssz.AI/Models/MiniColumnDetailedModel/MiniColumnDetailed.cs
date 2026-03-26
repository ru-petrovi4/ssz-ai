using Avalonia.Media;
using Ssz.Utils;
using Ssz.Utils.Avalonia.Model3D;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Collections;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
//  МИНИКОЛОНКА КОРЫ МОЗГА
//
//  Параметры, соответствующие анатомии человека:
//    - Диаметр: ~40 мкм (диапазон 30–50 мкм)
//    - Высота (слои II–VI): ~2000 мкм
//    - Нейронов: 80–120 реально, здесь 200 (задание)
//
//  Координатная система (мкм):
//    X, Y — горизонтальные оси в плоскости коры
//    Z     — вертикальная ось (глубина слоёв, 0 = поверхность)
//
//  Источник: Buxhoeveden & Casanova 2002 (Brain),
//            Wikipedia: Cortical minicolumn.
// ============================================================
public sealed class MiniColumnDetailed : IDisposable
{
    // ----------------------------------------------------------
    //  ФИЗИЧЕСКИЕ РАЗМЕРЫ МИНИКОЛОНКИ (мкм)
    // ----------------------------------------------------------

    /// <summary>Радиус миниколонки в горизонтальной плоскости (мкм).</summary>
    public const float ColumnRadiusUm = 20.0f;   // диаметр ~40 мкм

    /// <summary>Высота миниколонки (мкм), покрывает слои II–VI.</summary>
    public const float ColumnHeightUm = 2000.0f;

    // ----------------------------------------------------------
    //  ПАРАМЕТРЫ АКСОНОВ
    // ----------------------------------------------------------

    /// <summary>Число аксонов в миниколонке.</summary>
    public const int AxonCount = 200;

    /// <summary>Число исходящих синапсов на каждый аксон.</summary>
    public const int SynapsesPerAxon = 10_000; // Orig^10_000;

    public FastList<ActiveZone>? Temp_ActiveZones;

    /// <summary>
    /// Среднее число точек ветвления аксона.
    /// По данным Bhatt et al. 2009: среднее 4.5, диапазон 0–17.
    /// </summary>
    private const float MeanBranchPoints = 4.5f;

    /// <summary>
    /// Средняя длина сегмента аксона между ветвлениями (мкм).
    /// Из данных: суммарная длина ~5–20 мм, ветвлений ~4–8 =>
    /// сегмент ~500–1000 мкм. Используем 600 мкм.
    /// </summary>
    private const float MeanSegmentLengthUm = 600.0f;

    /// <summary>
    /// Число промежуточных точек на каждом сегменте аксона.
    /// Определяет детализацию траектории аксона.
    /// </summary>
    private const int PointsPerSegment = 8;

    // ----------------------------------------------------------
    //  ДАННЫЕ МИНИКОЛОНКИ
    // ----------------------------------------------------------

    /// <summary>Все 200 аксонов миниколонки.</summary>
    public readonly Axon[] Axons;    

    // ----------------------------------------------------------
    //  ГЕНЕРАТОР СЛУЧАЙНЫХ ЧИСЕЛ
    //  Инициализируется с фиксированным seed для воспроизводимости.
    // ----------------------------------------------------------
    private readonly Random _random;

    private readonly Device _device;

    // ============================================================
    //  КОНСТРУКТОР: ГЕНЕРАЦИЯ ВСЕЙ МИНИКОЛОНКИ
    // ============================================================
    /// <summary>
    /// Создаёт миниколонку: генерирует 200 аксонов с биологически
    /// правдоподобной морфологией и 10 000 синапсов каждый.
    /// </summary>
    /// <param name="random">Seed для генератора случайных чисел.</param>
    public MiniColumnDetailed(Random random)
    {
        _random = random;
        _device = cuda.is_available() ? CUDA : CPU;

        Axons = new Axon[AxonCount];        

        // ----------------------------------------------------------
        //  ШАГ 1: Разместить 200 сом в цилиндре миниколонки.
        //  Сомы распределены по всем слоям (Z = 0..2000 мкм)
        //  и в радиусе ColumnRadiusUm от центральной оси.
        // ----------------------------------------------------------
        var somaPositions = GenerateSomaPositions();

        // ----------------------------------------------------------
        //  ШАГ 2: Для каждого нейрона вырастить аксон.
        // ----------------------------------------------------------
        for (int i = 0; i < AxonCount; i += 1)
        {
            AxonPoint root = GrowAxon(somaPositions[i], i);
            Synapse[] synapses = PlaceSynapses(root);
            Axons[i] = new Axon(i, root, synapses);
        }        
    }

    // ============================================================
    //  ГЕНЕРАЦИЯ ПОЗИЦИЙ СОМ
    // ============================================================
    /// <summary>
    /// Располагает 200 нейронных сом равномерно в цилиндре
    /// миниколонки. Диаметр ~40 мкм, высота ~2000 мкм.
    /// Сомы размещены в 5 слоях (II, III, IV, V, VI),
    /// что соответствует реальной корковой анатомии.
    /// </summary>
    private Vector3[] GenerateSomaPositions()
    {
        // Границы слоёв (Z в мкм): слой II-III верхний, VI нижний
        // Примерные относительные толщины слоёв коры:
        // L II: 0–200, L III: 200–600, L IV: 600–900,
        // L V: 900–1400, L VI: 1400–2000
        ReadOnlySpan<(float zMin, float zMax, int neuronCount)> layers =
        [
            (0f,    200f,   20),   // слой II
            (200f,  600f,   50),   // слой III
            (600f,  900f,   30),   // слой IV
            (900f,  1400f,  60),   // слой V
            (1400f, 2000f,  40),   // слой VI
        ];
        // Итого: 200 нейронов распределены по 5 слоям.

        var positions = new Vector3[AxonCount];
        int idx = 0;

        foreach (var (zMin, zMax, count) in layers)
        {
            for (int n = 0; n < count; n += 1)
            {
                // Равномерное распределение в круге (метод отбора)
                float x, y;
                do
                {
                    x = (_random.NextSingle() * 2.0f - 1.0f) * ColumnRadiusUm;
                    y = (_random.NextSingle() * 2.0f - 1.0f) * ColumnRadiusUm;
                }
                while (x * x + y * y > ColumnRadiusUm * ColumnRadiusUm);

                float z = zMin + _random.NextSingle() * (zMax - zMin);
                positions[idx] = new Vector3(x, y, z);
                idx += 1;
            }
        }

        return positions;
    }

    // ============================================================
    //  РОСТ АКСОНА
    // ============================================================
    /// <summary>
    /// Строит дерево аксона, начиная с позиции сомы.
    ///
    /// Биологически правдоподобная морфология:
    ///   1. AIS (axon initial segment): ~60–80 мкм вертикально вниз
    ///   2. Горизонтальное распространение по слою с ветвлениями
    ///   3. Бинарные ветвления, среднее число ~4.5 на аксон
    ///   4. Угловые отклонения сегментов: плавная кривая (Perlin-like)
    ///
    /// Координаты: X,Y — горизонталь, Z — вертикаль (глубина).
    /// </summary>
    private AxonPoint GrowAxon(Vector3 somaPos, int axonIdx)
    {
        // Случайное число точек ветвления: Poisson-подобное ~4.5
        int totalBranchPoints = SamplePoissonBranchCount();

        // Создаём корневой узел в точке сомы
        var root = new AxonPoint(somaPos);

        // ----------------------------------------------------------
        //  AIS: аксон выходит из основания сомы и идёт вертикально
        //  вниз ~60–100 мкм. Это биологически правильно — AIS
        //  всегда направлен перпендикулярно поверхности коры.
        // ----------------------------------------------------------
        float aisLength = 60.0f + _random.NextSingle() * 40.0f; // 60–100 мкм
        int aisPoints = 4;
        float aisStep = aisLength / aisPoints;

        AxonPoint current = root;
        for (int p = 1; p <= aisPoints; p += 1)
        {
            // Небольшое горизонтальное отклонение (реалистичная кривизна)
            float jitter = 1.5f;
            var pos = new Vector3(
                somaPos.X + (_random.NextSingle() - 0.5f) * jitter,
                somaPos.Y + (_random.NextSingle() - 0.5f) * jitter,
                somaPos.Z + p * aisStep  // идёт вниз (увеличение Z)
            );
            var next = new AxonPoint(pos);
            current.Next.Add(next);
            current = next;
        }

        // ----------------------------------------------------------
        //  РЕКУРСИВНЫЙ РОСТ ВЕТВЕЙ
        //  Используем стек вместо рекурсии для производительности.
        // ----------------------------------------------------------
        // Стек: (текущий узел, оставшиеся ветвления, направление)
        var growthStack = new Stack<(AxonPoint node, int branchesLeft, Vector3 direction)>(32);

        // Начальное направление: горизонтально с небольшим наклоном
        float angle = _random.NextSingle() * MathF.PI * 2.0f; // случайный азимут
        var initDir = new Vector3(
            MathF.Cos(angle) * 0.9f,
            MathF.Sin(angle) * 0.9f,
            0.2f + _random.NextSingle() * 0.2f  // небольшой вертикальный компонент
        );
        initDir = Vector3.Normalize(initDir);

        growthStack.Push((current, totalBranchPoints, initDir));

        while (growthStack.Count > 0)
        {
            var (node, branchesLeft, direction) = growthStack.Pop();

            // Длина текущего сегмента (варьируется биологически)
            float segLen = MeanSegmentLengthUm * (0.6f + _random.NextSingle() * 0.8f);
            float stepLen = segLen / PointsPerSegment;

            // Рост сегмента от node до его конца
            AxonPoint segEnd = GrowSegment(node, direction, stepLen, PointsPerSegment);

            // Ветвление: если ещё есть ветвления — делим на 2 ветки
            if (branchesLeft > 0)
            {
                int branch1 = branchesLeft / 2;
                int branch2 = branchesLeft - branch1 - 1;

                // Угол расходящихся ветвей: ~30–60 градусов
                float spreadAngle = 0.4f + _random.NextSingle() * 0.5f; // ~23–52°
                Vector3 dir1 = RotateVector(direction, spreadAngle, _random);
                Vector3 dir2 = RotateVector(direction, -spreadAngle, _random);

                growthStack.Push((segEnd, branch1, Vector3.Normalize(dir1)));
                growthStack.Push((segEnd, branch2, Vector3.Normalize(dir2)));
            }
            // Иначе — терминальный сегмент (конец ветки = синаптический бутон)
        }

        return root;
    }

    // ============================================================
    //  РОСТ ОДНОГО СЕГМЕНТА АКСОНА
    // ============================================================
    /// <summary>
    /// Строит линейный сегмент аксона из нескольких точек.
    /// Добавляет небольшой случайный изгиб (биологически:
    /// аксоны не прямые, а плавно искривлённые).
    /// Возвращает последнюю точку сегмента.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private AxonPoint GrowSegment(AxonPoint start, Vector3 direction,
                                  float stepLen, int numPoints)
    {
        AxonPoint current = start;
        Vector3 dir = direction;

        for (int p = 0; p < numPoints; p += 1)
        {
            // Случайное малое отклонение направления (изгиб аксона)
            // Максимальное отклонение ~10° на шаг
            float bendAngle = (float)(_random.NextDouble() - 0.5) * 0.18f;
            dir = Vector3.Normalize(RotateVector(dir, bendAngle, _random));

            var pos = current.Position + dir * stepLen;

            // Ограничение Z: аксон не выходит за пределы колонки
            pos.Z = Math.Clamp(pos.Z, 0f, ColumnHeightUm);

            var next = new AxonPoint(pos);
            current.Next.Add(next);
            current = next;
        }

        return current;
    }

    // ============================================================
    //  ВРАЩЕНИЕ ВЕКТОРА
    // ============================================================
    /// <summary>
    /// Поворачивает вектор v на угол angle вокруг случайной оси,
    /// перпендикулярной v. Используется для создания ветвлений
    /// и изгибов аксона.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector3 RotateVector(Vector3 v, float angle, Random random)
    {
        // Находим произвольную ось, перпендикулярную v
        // Используем метод Хьюза–Мёллера для устойчивости
        Vector3 perp;
        if (MathF.Abs(v.X) <= MathF.Abs(v.Y) && MathF.Abs(v.X) <= MathF.Abs(v.Z))
            perp = new Vector3(0, -v.Z, v.Y);
        else if (MathF.Abs(v.Y) <= MathF.Abs(v.Z))
            perp = new Vector3(-v.Z, 0, v.X);
        else
            perp = new Vector3(-v.Y, v.X, 0);

        perp = Vector3.Normalize(perp);

        // Поворачиваем ось в горизонтальной плоскости случайно
        float phi = (float)(random.NextDouble() * Math.PI * 2.0);
        Vector3 crossV = Vector3.Cross(v, perp);
        Vector3 rotAxis = perp * MathF.Cos(phi) + crossV * MathF.Sin(phi);
        rotAxis = Vector3.Normalize(rotAxis);

        // Формула Родрига: v_rot = v*cos(a) + (axis×v)*sin(a) + axis*(axis·v)*(1-cos(a))
        float cosA = MathF.Cos(angle);
        float sinA = MathF.Sin(angle);
        float dot = Vector3.Dot(rotAxis, v);
        return v * cosA + Vector3.Cross(rotAxis, v) * sinA + rotAxis * dot * (1f - cosA);
    }

    // ============================================================
    //  ВЫБОРКА ЧИСЛА ВЕТВЛЕНИЙ (ПУАССОН-ПОДОБНОЕ)
    // ============================================================
    /// <summary>
    /// Сэмплирует число точек ветвления для одного аксона.
    /// Среднее λ = 4.5 (по Bhatt et al. 2009).
    /// Использует алгоритм Кнута для генерации Пуассона.
    /// </summary>
    private int SamplePoissonBranchCount()
    {
        const double lambda = 4.5;
        double L = Math.Exp(-lambda);
        double p = 1.0;
        int k = 0;
        do
        {
            k += 1;
            p *= _random.NextDouble();
        }
        while (p > L);
        // Ограничиваем диапазон [1, 12] для адекватной топологии дерева
        return Math.Clamp(k - 1, 1, 12);
    }

    // ============================================================
    //  РАЗМЕЩЕНИЕ СИНАПСОВ НА ДЕРЕВЕ АКСОНА
    // ============================================================
    /// <summary>
    /// Размещает синапсы равномерно по всей длине аксона.
    /// Алгоритм собирает все сегменты (отрезки между узлами аксона),
    /// вычисляет их суммарную длину и распределяет синапсы
    /// строго с одинаковым интервалом вдоль ветвей (с небольшим шумом).
    /// </summary>
    private Synapse[] PlaceSynapses(AxonPoint root)
    {
        // 1. Сначала собираем все отрезки (сегменты) дерева аксона.
        // Используем легковесные кортежи (ValueTuple) для хранения структуры:
        // Начальная координата, Конечная координата и Длина отрезка в мкм.
        var segments = new List<(Vector3 Start, Vector3 End, float Length)>(capacity: 512);

        // Стек для обхода дерева без рекурсии, что гарантирует максимальную производительность
        var traversalStack = new Stack<AxonPoint>(128);
        traversalStack.Push(root);

        // Переменная для хранения общей длины всех ветвей аксона
        float totalLength = 0f;

        // Обходим всё дерево аксона в глубину
        while (traversalStack.Count > 0)
        {
            var pt = traversalStack.Pop();

            // Если у точки есть дочерние узлы (продолжения аксона)
            if (pt.Next != null)
            {
                // Используем += 1 вместо запрещенного ++
                for (int i = 0; i < pt.Next.Count; i += 1)
                {
                    var child = pt.Next[i];

                    // Вычисляем длину отрезка с помощью SIMD-совместимого метода System.Numerics
                    float length = Vector3.Distance(pt.Position, child.Position);

                    // Записываем только валидные отрезки, имеющие ненулевую длину
                    if (length > 0f)
                    {
                        segments.Add((pt.Position, child.Position, length));
                        totalLength += length; // Накапливаем общую длину
                    }

                    traversalStack.Push(child);
                }
            }
        }

        FastList<Synapse> synapses = new(SynapsesPerAxon);

        // 2. Краевой случай: Если аксон вырожден (длина нулевая), то размещаем все синапсы в корне
        if (totalLength <= 0f || segments.Count == 0)
        {
            for (int s = 0; s < SynapsesPerAxon; s += 1)
            {
                synapses.Add(new Synapse(root.Position));
            }
            return synapses.ToArray();
        }

        // 3. Вычисляем шаг, через который будут расставлены синапсы для идеальной равномерности
        float step = totalLength / SynapsesPerAxon;

        // Переменная для отслеживания индекса текущего сегмента при линейном проходе
        int currentSegmentIdx = 0;

        // Расстояние от начала текущего сегмента, на котором размещается очередной синапс
        // Начинаем с половины шага, чтобы края аксона не были излишне "плотными"
        float currentDistInSegment = step * 0.5f;

        // 4. Проходим по необходимому количеству синапсов и размещаем их
        for (int s = 0; s < SynapsesPerAxon; s += 1)
        {
            // Если мы вышли за пределы длины текущего отрезка, переходим к следующему
            // (цикл нужен для случаев, когда отрезки слишком короткие по сравнению с шагом)
            while (currentDistInSegment > segments[currentSegmentIdx].Length && currentSegmentIdx < segments.Count - 1)
            {
                currentDistInSegment -= segments[currentSegmentIdx].Length;
                currentSegmentIdx += 1; // Увеличиваем индекс без использования ++
            }

            // Получаем текущий сегмент аксона
            var seg = segments[currentSegmentIdx];

            // Вычисляем долю (от 0.0 до 1.0) пройденного расстояния по отрезку
            float t = currentDistInSegment / seg.Length;

            // Выполняем линейную интерполяцию для нахождения точной позиции синапса на отрезке
            var basePos = Vector3.Lerp(seg.Start, seg.End, t);

            // 5. Добавляем небольшое случайное смещение (~1.5 мкм) для реализма,
            // имитируя бутоны, расположенные на малом расстоянии от центральной оси аксона
            float jitter = 1.5f;
            var synPos = new Vector3(
                basePos.X + (_random.NextSingle() - 0.5f) * jitter * 2f,
                basePos.Y + (_random.NextSingle() - 0.5f) * jitter * 2f,
                basePos.Z + (_random.NextSingle() - 0.5f) * jitter * 2f
            );

            // Создаем новый синапс только, если он примерно внутри миниколонки. Остальные не интересуют
            float r = ColumnRadiusUm * 2.0f;
            if (synPos.X > -r && synPos.X < r &&
                    synPos.Y > -r && synPos.Y < r)
                synapses.Add(new Synapse(synPos));

            // Продвигаемся дальше по отрезку на заданный равномерный шаг
            currentDistInSegment += step;
        }

        // Возвращаем массив готовых синапсов, равномерно расставленных по всему аксону
        return synapses.ToArray();
    }        

    // ============================================================
    //  ПОИСК АКТИВНЫХ ЗОН (С ИСПОЛЬЗОВАНИЕМ TORCHSHARP)
    // ============================================================
    /// <summary>
    /// Ищет пространственные области (зоны) заданного радиуса,
    /// внутри которых присутствуют синапсы как минимум от N различных активных аксонов.
    /// Алгоритм использует вокселизацию пространства и 3D-свертку на базе TorchSharp
    /// для точного математического поиска максимумов плотности в любой точке пространства
    /// (а не только с центрами в синапсах), после чего проводит дедупликацию.
    /// </summary>
    /// <param name="activityBits">Вектор, где 1.0f означает активность аксона.</param>
    /// <param name="radiusUm">Радиус поиска (мкм).</param>
    /// <param name="minActiveAxons">Минимальное количество уникальных активных аксонов в радиусе.</param>
    /// <returns>Список найденных уникальных зон активности.</returns>
    public void FindActiveZones(
        float[] activityBits,
        float radiusUm,
        int minActiveAxons)
    {
        using var disposeScope = torch.NewDisposeScope();

        // ----------------------------------------------------------
        //  ШАГ 1: Извлечение индексов активных аксонов
        // ----------------------------------------------------------
        _activeAxons.Clear();
        for (int i = 0; i < AxonCount; i += 1) 
        {            
            if (Axons[i].Temp_IsActive = (activityBits[i] > 0.5f))
            {                
                _activeAxons.Add(i);
            }
        }

        int activeCount = _activeAxons.Count;

        // Если активных аксонов меньше чем порог, зон быть не может
        if (activeCount < minActiveAxons)
        {
            Temp_ActiveZones = null;
            return;
        }

        // ----------------------------------------------------------
        //  ШАГ 2: Определение границ пространства и параметров сетки
        // ----------------------------------------------------------
        float voxelSizeUm = 2.0f; // Orig: Шаг сетки (2 мкм) — баланс между точностью и памятью GPU/CPU

        SceneBounds sceneBounds = new();

        // Находим реальные границы (bounding box) для активных синапсов
        for (int a = 0; a < activeCount; a += 1)
        {
            var synapses = Axons[_activeAxons[a]].Synapses;
            for (int s_Index = 0; s_Index < synapses.Length; s_Index += 1)
            {
                sceneBounds.Update(synapses[s_Index].Position);                
            }
        }

        // Расширяем границы на величину радиуса, чтобы захватить краевые зоны
        sceneBounds.XMin -= radiusUm; sceneBounds.YMin -= radiusUm; sceneBounds.ZMin -= radiusUm;
        sceneBounds.XMax += radiusUm; sceneBounds.YMax += radiusUm; sceneBounds.ZMax += radiusUm;

        // Вычисляем размерности тензора: [Глубина(Z), Высота(Y), Ширина(X)]
        int width = (int)MathF.Ceiling((sceneBounds.XMax - sceneBounds.XMin) / voxelSizeUm);
        int height = (int)MathF.Ceiling((sceneBounds.YMax - sceneBounds.YMin) / voxelSizeUm);
        int depth = (int)MathF.Ceiling((sceneBounds.ZMax - sceneBounds.ZMin) / voxelSizeUm);

        // ----------------------------------------------------------
        // ШАГ 3: Создание входного тензора (Grid)
        // ----------------------------------------------------------
        long totalVoxels = (long)activeCount * depth * height * width;

        // ==========================================================
        // ПЕРЕИСПОЛЬЗОВАНИЕ ТЕНЗОРОВ (ABSOLUTE ZERO-ALLOCATION)
        // ==========================================================
        if (_gridTensorBuffer is null)
            _gridTensorBuffer = new TensorBuffer(_device, totalVoxels);
        else
            _gridTensorBuffer.EnsureCapacity(totalVoxels);

        // 1. Создаем срезы ровно под размер текущих данных
        using var gridTensor_Cpu_Flat = _gridTensorBuffer.Tensor_Cpu_Buffer!.slice(0, 0, totalVoxels, 1);
        using var gridTensor_device_Flat = _gridTensorBuffer.Tensor_device_Buffer.slice(0, 0, totalVoxels, 1);

        // 2. Быстро очищаем память (зануляем) на уровне C++
        gridTensor_Cpu_Flat.zero_();

        long spatialSize = (long)depth * height * width;
        long areaSize = (long)height * width;

        // 3. Получаем прямое окно (Span) в неуправляемую память CPU-тензора.
        // Это работает мгновенно и не делает никаких копий.
        var gridTensorData = gridTensor_Cpu_Flat.data<float>();

        // 4. Проецируем синапсы напрямую в память TorchSharp
        for (int a = 0; a < activeCount; a += 1)
        {
            var synapses = Axons[_activeAxons[a]].Synapses;
            long channelOffset = a * spatialSize;

            for (int s = 0; s < synapses.Length; s += 1)
            {
                var p = synapses[s].Position;
                int ix = (int)((p.X - sceneBounds.XMin) / voxelSizeUm);
                int iy = (int)((p.Y - sceneBounds.YMin) / voxelSizeUm);
                int iz = (int)((p.Z - sceneBounds.ZMin) / voxelSizeUm);

                if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 && iz < depth)
                {
                    long index = channelOffset + (iz * areaSize) + (iy * width) + ix;

                    // Прямая запись в нативную память без P/Invoke!
                    // Скорость записи идентична float* в C++
                    gridTensorData[index] = 1.0f;
                }
            }
        }

        // 5. Копируем данные из оперативной памяти (CPU) в видеопамять (Device)
        gridTensor_device_Flat.copy_(gridTensor_Cpu_Flat);

        // 6. Формируем итоговый 5D-view для свертки
        Tensor gridTensor_device = gridTensor_device_Flat.view(1, activeCount, depth, height, width);

        // ----------------------------------------------------------
        //  ШАГ 4: Создание сферического ядра для 3D-свертки
        // ----------------------------------------------------------
        int kRad = (int)MathF.Ceiling(radiusUm / voxelSizeUm);
        int kSize = kRad * 2 + 1; // Нечетный размер ядра для наличия четкого центра
        float[] kernelData = new float[activeCount * kSize * kSize * kSize];

        long kSpatial = (long)kSize * kSize * kSize;
        long kArea = (long)kSize * kSize;

        // Формируем сферу. Если дистанция от центра <= радиус, ставим 1.0
        for (int kz = -kRad; kz <= kRad; kz += 1)
            for (int ky = -kRad; ky <= kRad; ky += 1)
                for (int kx = -kRad; kx <= kRad; kx += 1)
                {
                    float dist = MathF.Sqrt(kx * kx + ky * ky + kz * kz) * voxelSizeUm;
                    if (dist <= radiusUm)
                    {
                        int ikx = kx + kRad;
                        int iky = ky + kRad;
                        int ikz = kz + kRad;
                        long offset = (ikz * kArea) + (iky * kSize) + ikx;

                        // Дублируем веса ядра для каждого канала (Depthwise Convolution)
                        for (int c = 0; c < activeCount; c += 1)
                        {
                            kernelData[c * kSpatial + offset] = 1.0f;
                        }
                    }
                }

        // Тензор весов свертки: [OutChannels=activeCount, InChannels=1 (из-за groups), kD, kH, kW]
        using var weightTensor_device = tensor(kernelData, new long[] { activeCount, 1, kSize, kSize, kSize }, dtype: ScalarType.Float32, device: _device);

        // ----------------------------------------------------------
        //  ШАГ 5: Выполнение вычислений через TorchSharp
        // ----------------------------------------------------------
        // Depthwise 3D свертка (каждый аксон обрабатывается независимо)
        using var convResult_device = nn.functional.conv3d(
            gridTensor_device,
            weightTensor_device,
            padding: new long[] { kRad, kRad, kRad },
            groups: activeCount);

        // Бинаризуем результат: если > 0, значит синапсы этого аксона есть в радиусе R
        using var presentMask_device = convResult_device.gt(0.0f).to_type(ScalarType.Float32);

        // Суммируем измерения вдоль каналов (dim=1), чтобы получить число уникальных аксонов
        using var activeAxonsCount_device = presentMask_device.sum(new long[] { 1 }, keepdim: false);

        // Оставляем только те воксели, где число аксонов >= N
        using var validZonesMask_device = activeAxonsCount_device.ge(minActiveAxons);

        // Получаем индексы (координаты) всех успешных вокселей. 
        // Формат: [Кол-во вокселей, 4 (Batch, Z, Y, X)]
        using var nonZeroIndices_device = validZonesMask_device.nonzero();

        long numValidVoxels = nonZeroIndices_device.shape[0];
        long dims = nonZeroIndices_device.shape[1]; // Должно быть равно 4
        long[] flatIndices = null!;
        if (numValidVoxels > 0)
            flatIndices = nonZeroIndices_device.data<long>().ToArray();

        // ----------------------------------------------------------
        //  ШАГ 6: Преобразование обратно в 3D пространство (Vector3)
        // ----------------------------------------------------------
        var rawCenters = new List<Vector3>((int)numValidVoxels);
        for (long i = 0; i < numValidVoxels; i += 1)
        {
            // Извлекаем индексы: Batch(0), Depth(1), Height(2), Width(3)
            long zIdx = flatIndices[i * dims + 1];
            long yIdx = flatIndices[i * dims + 2];
            long xIdx = flatIndices[i * dims + 3];

            float xPos = sceneBounds.XMin + (xIdx * voxelSizeUm) + (voxelSizeUm * 0.5f);
            float yPos = sceneBounds.YMin + (yIdx * voxelSizeUm) + (voxelSizeUm * 0.5f);
            float zPos = sceneBounds.ZMin + (zIdx * voxelSizeUm) + (voxelSizeUm * 0.5f);

            rawCenters.Add(new Vector3(xPos, yPos, zPos));
        }

        // ----------------------------------------------------------
        //  ШАГ 7: Дедупликация (Слияние близлежащих вокселей)
        // ----------------------------------------------------------
        // Воксели часто образуют сплошные "облака" плотности. Мы объединяем
        // все воксели, находящиеся в радиусе R друг от друга, в единые зоны.
        var finalZones = new FastList<ActiveZone>(1024);
        float mergeRadiusSq = radiusUm * radiusUm;
        bool[] merged = new bool[rawCenters.Count];

        for (int i = 0; i < rawCenters.Count; i += 1)
        {
            if (merged[i]) continue;

            Vector3 sumPos = rawCenters[i];
            int count = 1;
            merged[i] = true;

            for (int j = i + 1; j < rawCenters.Count; j += 1)
            {
                if (merged[j]) continue;

                if (Vector3.DistanceSquared(rawCenters[i], rawCenters[j]) <= mergeRadiusSq)
                {
                    sumPos += rawCenters[j];
                    count += 1;
                    merged[j] = true;
                }
            }

            // Вычисляем геометрический центр объединения и формируем зону
            finalZones.Add(new ActiveZone
            {
                Center = sumPos / count
            });
        }

        Temp_ActiveZones = finalZones;
    }

    public void Dispose()
    {
        _gridTensorBuffer?.Dispose();
    }

    private readonly FastList<int> _activeAxons = new FastList<int>(capacity: AxonCount);

    // Кэшированный тензор для переиспользования памяти
    private TensorBuffer? _gridTensorBuffer;    
}

public class TensorBuffer : IDisposable
{
    public TensorBuffer(Device device, long capacity)
    {
        Device = device;
        Tensor_Buffer_Capacity = capacity; //Math.Max(totalVoxels, Tensor_Buffer_Capacity == 0 ? totalVoxels : Tensor_Buffer_Capacity * 2);

        Tensor_device_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: device).DetachFromDisposeScope();
        Tensor_Cpu_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: CPU).DetachFromDisposeScope();
    }

    public void EnsureCapacity(long capacity)
    {
        if (capacity <= Tensor_Buffer_Capacity)
            return;

        Tensor_device_Buffer.Dispose();
        Tensor_Cpu_Buffer.Dispose();

        Tensor_Buffer_Capacity = capacity; //Math.Max(totalVoxels, Tensor_Buffer_Capacity == 0 ? totalVoxels : Tensor_Buffer_Capacity * 2);

        Tensor_device_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: Device).DetachFromDisposeScope();
        Tensor_Cpu_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: CPU).DetachFromDisposeScope();
    }

    public readonly Device Device;

    public Tensor Tensor_device_Buffer;
    public Tensor Tensor_Cpu_Buffer;
    // Текущая вместимость (в элементах), чтобы понимать, когда нужно перевыделять память
    public long Tensor_Buffer_Capacity = 0;

    public void Dispose()
    {
        Tensor_device_Buffer?.Dispose();
        Tensor_Cpu_Buffer?.Dispose();
    }
}


//// ============================================================
//    //  ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: ЯЧЕЙКА ПРОСТРАНСТВЕННОГО ИНДЕКСА
//    // ============================================================
//    [MethodImpl(MethodImplOptions.AggressiveInlining)]
//    private (int, int, int) GetCell(Vector3 pos)
//    {
//        int ix = (int)MathF.Floor(pos.X / _cellSizeUm);
//        int iy = (int)MathF.Floor(pos.Y / _cellSizeUm);
//        int iz = (int)MathF.Floor(pos.Z / _cellSizeUm);
//        return (ix, iy, iz);
//    }

//// ----------------------------------------------------------
////  ШАГ 3: Построить пространственный индекс по синапсам.
////  Ячейка = cube со стороной _cellSize.
////  После заполнения всех аксонов _cellSize берём
////  как среднее расстояние между синапсами * 4.
//// ----------------------------------------------------------
//_cellSizeUm = 15.0f; // мкм, исходя из плотности синапсов

//// ----------------------------------------------------------
////  ПРОСТРАНСТВЕННЫЙ ИНДЕКС ДЛЯ БЫСТРОГО ПОИСКА СИНАПСОВ
////
////  Реализован как словарь: ключ = целочисленная 3D-ячейка
////  (ix, iy, iz) пространственной решётки, значение = список
////  (индекс_аксона, индекс_синапса).
////
////  Это позволяет при поиске зон радиуса R перебирать только
////  синапсы в ближайших ячейках, а не все 200 × 10 000 = 2M.
//// ----------------------------------------------------------
//private readonly Dictionary<(int, int, int), FastList<(int axonIdx, int synIdx)>> _spatialIndex;

///// <summary>Размер ячейки пространственного индекса (мкм).</summary>
//private float _cellSizeUm;


//// ============================================================
////  ПОСТРОЕНИЕ ПРОСТРАНСТВЕННОГО ИНДЕКСА
//// ============================================================
///// <summary>
///// Строит пространственный хэш-индекс для всех синапсов.
///// Ключ = целочисленная ячейка (ix, iy, iz).
///// Ячейка имеет размер _cellSize мкм.
/////
///// Это позволяет выполнять поиск за O(k) вместо O(2M),
///// где k — число синапсов в нескольких соседних ячейках.
///// </summary>
//private void BuildSpatialIndex()
//{
//    for (int a = 0; a < AxonCount; a += 1)
//    {
//        var synapses = Axons[a].Synapses;
//        for (int s_index = 0; s_index < synapses.Length; s_index += 1)
//        {
//            var cell = GetCell(synapses[s_index].Position);
//            if (!_spatialIndex.TryGetValue(cell, out var list))
//            {
//                list = new FastList<(int, int)>(capacity: 8);
//                _spatialIndex[cell] = list;
//            }
//            list.Add((a, s_index));
//        }
//    }
//}