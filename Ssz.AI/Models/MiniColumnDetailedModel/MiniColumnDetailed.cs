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
    #region construction and destruction

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

        PyramidalAxons = new PyramidalAxon[PyramidalAxonsCount];

        // ----------------------------------------------------------
        //  ШАГ 1: Разместить 200 сом в цилиндре миниколонки.
        //  Сомы распределены по всем слоям (Z = 0..2000 мкм)
        //  и в радиусе ColumnRadiusUm от центральной оси.
        // ----------------------------------------------------------
        var pyramidalSomaPositions = GeneratePyramidalSomaPositions();

        // ----------------------------------------------------------
        //  ШАГ 2: Для каждого нейрона вырастить аксон.
        // ----------------------------------------------------------
        for (int i = 0; i < PyramidalAxonsCount; i += 1)
        {
            AxonPoint root = GrowPyramidalAxon(pyramidalSomaPositions[i], i);
            Synapse[] synapses = PlaceSynapses(root);
            PyramidalAxons[i] = new PyramidalAxon(i, root, synapses);
        }

        ThalamocorticalInput = new ThalamocorticalInput(_random, ColumnRadiusUm, ColumnHeightUm);
    }

    public void Dispose()
    {
        _gridTensorBuffer?.Dispose();
    }

    #endregion

    #region public functions

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
    public const int PyramidalAxonsCount = 200;

    /// <summary>Число исходящих синапсов на каждый аксон.</summary>
    public const int SynapsesPerAxon = 10_000; // Orig^10_000;

    /// <summary>Активные зоны пирамидальных аксонов (Зоны 1).</summary>
    public FastList<ActiveZone>? Temp_PyramidalZones;

    /// <summary>Активные зоны входящих таламокортикальных аксонов (Зоны 2).</summary>
    public FastList<ActiveZone>? Temp_ThalamocorticalZones;

    /// <summary>Зоны совместной активации: рядом есть и Зоны 1, и Зоны 2 (Зоны 3).</summary>
    public FastList<ActiveZone>? Temp_ConvergenceZones;

    /// <summary>
    /// Вычисляет зоны активных синапсов раздельно для пирамидальных аксонов
    /// и таламокортикальных афферентов, а затем находит их зоны конвергенции.
    /// </summary>
    /// <param name="activityBits">Активность локальных аксонов (длина AxonCount).</param>
    /// <param name="radiusUm">Радиус поиска для отдельных зон (мкм).</param>
    /// <param name="minActiveAxons">Минимум уникальных активных аксонов в одной зоне.</param>
    /// <param name="convergenceRadiusUm">
    ///   Максимальное расстояние между Зоной 1 и Зоной 2, при котором
    ///   они считаются зоной конвергенции (Зона 3). По умолчанию равно radiusUm.
    /// </param>
    public void FindActiveZones(
        float[] activityBits,
        float radiusUm,
        int minActiveAxons,
        float convergenceRadiusUm = -1f)
    {
        if (convergenceRadiusUm < 0f)
            convergenceRadiusUm = radiusUm;

        using var disposeScope = torch.NewDisposeScope();

        // ----------------------------------------------------------
        //  ШАГ 1: Индексы активных пирамидальных аксонов
        // ----------------------------------------------------------
        _activePyramidalAxons.Clear();
        for (int i = 0; i < PyramidalAxonsCount; i += 1)
        {
            bool isActive = PyramidalAxons[i].Temp_IsActive = (activityBits[i] > 0.5f);
            if (isActive)
                _activePyramidalAxons.Add(i);
        }

        // ----------------------------------------------------------
        //  ШАГ 2: Индексы активных ТК-аксонов
        // ----------------------------------------------------------
        _activeTcAxons.Clear();
        for (int i = 0; i < ThalamocorticalInput.TotalAxonCount; i += 1)
        {
            bool isActive = ThalamocorticalInput.Axons[i].Temp_IsActive;
            if (isActive)
                _activeTcAxons.Add(i);
        }

        // ----------------------------------------------------------
        //  ЗОНЫ 1: Пирамидальные аксоны
        // ----------------------------------------------------------
        Temp_PyramidalZones = _activePyramidalAxons.Count >= minActiveAxons
            ? ComputeActiveZones(
                GetSynapsesByPyramidalAxons(_activePyramidalAxons),
                _activePyramidalAxons.Count,
                radiusUm,
                minActiveAxons)
            : null;

        // ----------------------------------------------------------
        //  ЗОНЫ 2: Таламокортикальные аксоны
        // ----------------------------------------------------------
        Temp_ThalamocorticalZones = _activeTcAxons.Count >= minActiveAxons
            ? ComputeActiveZones(
                GetSynapsesByThalamocorticalAxons(_activeTcAxons),
                _activeTcAxons.Count,
                radiusUm,
                minActiveAxons)
            : null;

        // ----------------------------------------------------------
        //  ЗОНЫ 3: Конвергенция (Зоны 1 рядом с Зонами 2)
        // ----------------------------------------------------------
        Temp_ConvergenceZones = FindConvergenceZones(
            Temp_PyramidalZones,
            Temp_ThalamocorticalZones,
            convergenceRadiusUm);
    }

    #endregion

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
    public readonly PyramidalAxon[] PyramidalAxons;

    public readonly ThalamocorticalInput ThalamocorticalInput;

    // ----------------------------------------------------------
    //  ГЕНЕРАТОР СЛУЧАЙНЫХ ЧИСЕЛ
    //  Инициализируется с фиксированным seed для воспроизводимости.
    // ----------------------------------------------------------
    private readonly Random _random;

    private readonly Device _device;

    // ============================================================
    //  ГЕНЕРАЦИЯ ПОЗИЦИЙ СОМ
    // ============================================================
    /// <summary>
    /// Располагает 200 нейронных сом равномерно в цилиндре
    /// миниколонки. Диаметр ~40 мкм, высота ~2000 мкм.
    /// Сомы размещены в 5 слоях (II, III, IV, V, VI),
    /// что соответствует реальной корковой анатомии.
    /// </summary>
    private Vector3[] GeneratePyramidalSomaPositions()
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

        var positions = new Vector3[PyramidalAxonsCount];
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
    private AxonPoint GrowPyramidalAxon(Vector3 somaPos, int axonIdx)
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
            AxonPoint segEnd = GrowPyramidalAxonSegment(node, direction, stepLen, PointsPerSegment);

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
    private AxonPoint GrowPyramidalAxonSegment(AxonPoint start, Vector3 direction,
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
    // ============================================================
    //  ПОИСК АКТИВНЫХ ЗОН
    //
    //  Метод выполняет три независимых прохода через GPU-свёртку:
    //
    //    Зоны 1 (Temp_PyramidalZones):
    //      Пространственные кластеры, где синапсы >= minActiveAxons
    //      различных ПИРАМИДАЛЬНЫХ аксонов попадают в радиус radiusUm.
    //
    //    Зоны 2 (Temp_ThalamocorticalZones):
    //      Аналогичные кластеры, но только для ВХОДЯЩИХ ЛКТ-аксонов.
    //      Т.е. места, где таламический зрительный сигнал плотно
    //      сходится в пространстве миниколонки.
    //
    //    Зоны 3 (Temp_ConvergenceZones):
    //      Зоны схождения: позиции, где хотя бы одна Зона 1 и хотя бы
    //      одна Зона 2 находятся в пределах convergenceRadiusUm.
    //      Биологически это потенциальные зоны синаптической интеграции
    //      таламического входа и локальной кортикальной активности.
    // ============================================================

    

    // ============================================================
    //  ВСПОМОГАТЕЛЬНЫЙ: СБОР СИНАПСОВ ПО СПИСКУ АКТИВНЫХ АКСОНОВ
    // ============================================================
    private (Synapse[][] groups, int count) GetSynapsesByPyramidalAxons(FastList<int> indices)
    {
        int count = indices.Count;
        var groups = new Synapse[count][];
        for (int i = 0; i < count; i += 1)
        {
            groups[i] = PyramidalAxons[indices[i]].Synapses;
        }
        return (groups, count);
    }

    // ============================================================
    //  ВСПОМОГАТЕЛЬНЫЙ: СБОР СИНАПСОВ ПО СПИСКУ АКТИВНЫХ АКСОНОВ
    // ============================================================
    private (Synapse[][] groups, int count) GetSynapsesByThalamocorticalAxons(FastList<int> indices)
    {
        int count = indices.Count;
        var groups = new Synapse[count][];
        for (int i = 0; i < count; i += 1)
        {
            groups[i] = ThalamocorticalInput.Axons[indices[i]].Synapses;
        }
        return (groups, count);
    }

    // ============================================================
    //  ЯДРО ВЫЧИСЛЕНИЯ ЗОН ЧЕРЕЗ 3D-СВЁРТКУ (GPU/CPU)
    //
    //  Принимает набор групп синапсов (одна группа = один аксон),
    //  выполняет depthwise свёртку и возвращает список зон.
    // ============================================================
    private FastList<ActiveZone> ComputeActiveZones(
        (Synapse[][] groups, int count) input,
        int activeCount,
        float radiusUm,
        int minActiveAxons)
    {
        var groups = input.groups;
        float voxelSizeUm = 2.0f;

        // ---- Bounding box ----
        SceneBounds bounds = new();
        for (int a = 0; a < activeCount; a += 1)
        {
            var synapses = groups[a];
            for (int s = 0; s < synapses.Length; s += 1)
                bounds.Update(synapses[s].Position);
        }
        bounds.XMin -= radiusUm; bounds.YMin -= radiusUm; bounds.ZMin -= radiusUm;
        bounds.XMax += radiusUm; bounds.YMax += radiusUm; bounds.ZMax += radiusUm;

        int width = (int)MathF.Ceiling((bounds.XMax - bounds.XMin) / voxelSizeUm);
        int height = (int)MathF.Ceiling((bounds.YMax - bounds.YMin) / voxelSizeUm);
        int depth = (int)MathF.Ceiling((bounds.ZMax - bounds.ZMin) / voxelSizeUm);

        // ---- Grid tensor ----
        long totalVoxels = (long)activeCount * depth * height * width;

        if (_gridTensorBuffer is null)
            _gridTensorBuffer = new TensorBuffer(_device, totalVoxels);
        else
            _gridTensorBuffer.EnsureCapacity(totalVoxels);

        using var gridFlat_Cpu = _gridTensorBuffer.Tensor_Cpu_Buffer!.slice(0, 0, totalVoxels, 1);
        using var gridFlat_Device = _gridTensorBuffer.Tensor_device_Buffer.slice(0, 0, totalVoxels, 1);
        gridFlat_Cpu.zero_();

        long spatialSize = (long)depth * height * width;
        long areaSize = (long)height * width;
        var gridData = gridFlat_Cpu.data<float>();

        for (int a = 0; a < activeCount; a += 1)
        {
            var synapses = groups[a];
            long channelOffset = (long)a * spatialSize;
            for (int s = 0; s < synapses.Length; s += 1)
            {
                var p = synapses[s].Position;
                int ix = (int)((p.X - bounds.XMin) / voxelSizeUm);
                int iy = (int)((p.Y - bounds.YMin) / voxelSizeUm);
                int iz = (int)((p.Z - bounds.ZMin) / voxelSizeUm);

                if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 && iz < depth)
                    gridData[channelOffset + (iz * areaSize) + ((long)iy * width) + ix] = 1.0f;
            }
        }

        gridFlat_Device.copy_(gridFlat_Cpu);
        Tensor gridTensor = gridFlat_Device.view(1, activeCount, depth, height, width);

        // ---- Spherical kernel ----
        int kRad = (int)MathF.Ceiling(radiusUm / voxelSizeUm);
        int kSize = kRad * 2 + 1;
        long weightElements = (long)activeCount * kSize * kSize * kSize;

        if (_weightTensorBuffer is null)
            _weightTensorBuffer = new TensorBuffer(_device, weightElements);
        else
            _weightTensorBuffer.EnsureCapacity(weightElements);

        using var weightFlat_Cpu = _weightTensorBuffer.Tensor_Cpu_Buffer!.slice(0, 0, weightElements, 1);
        using var weightFlat_Device = _weightTensorBuffer.Tensor_device_Buffer.slice(0, 0, weightElements, 1);
        weightFlat_Cpu.zero_();
        var kernelData = weightFlat_Cpu.data<float>();

        long kSpatial = (long)kSize * kSize * kSize;
        long kArea = (long)kSize * kSize;

        for (int kz = -kRad; kz <= kRad; kz += 1)
        {
            for (int ky = -kRad; ky <= kRad; ky += 1)
            {
                for (int kx = -kRad; kx <= kRad; kx += 1)
                {
                    float dist = MathF.Sqrt(kx * kx + ky * ky + kz * kz) * voxelSizeUm;
                    if (dist <= radiusUm)
                    {
                        long offset = (long)(kz + kRad) * kArea + (ky + kRad) * kSize + (kx + kRad);
                        for (int c = 0; c < activeCount; c += 1)
                            kernelData[c * kSpatial + offset] = 1.0f;
                    }
                }
            }
        }

        weightFlat_Device.copy_(weightFlat_Cpu);
        using var weightTensor = weightFlat_Device.view(activeCount, 1, kSize, kSize, kSize);

        // ---- Convolution ----
        using var convResult = nn.functional.conv3d(
            gridTensor, weightTensor,
            padding: new long[] { kRad, kRad, kRad },
            groups: activeCount);

        using var presentMask = convResult.gt(0.0f).to_type(ScalarType.Float32);
        using var axonsPerVoxel = presentMask.sum(new long[] { 1 }, keepdim: false);
        using var validMask = axonsPerVoxel.ge(minActiveAxons);
        using var nonZeroIdx = validMask.nonzero();

        long numValid = nonZeroIdx.shape[0];
        long dims = nonZeroIdx.shape[1];
        long[] flatIdx = null!;
        if (numValid > 0)
            flatIdx = nonZeroIdx.data<long>().ToArray();

        // ---- Back-project ----
        var rawCenters = new List<Vector3>((int)numValid);
        for (long i = 0; i < numValid; i += 1)
        {
            long zIdx = flatIdx[i * dims + 1];
            long yIdx = flatIdx[i * dims + 2];
            long xIdx = flatIdx[i * dims + 3];
            rawCenters.Add(new Vector3(
                bounds.XMin + xIdx * voxelSizeUm + voxelSizeUm * 0.5f,
                bounds.YMin + yIdx * voxelSizeUm + voxelSizeUm * 0.5f,
                bounds.ZMin + zIdx * voxelSizeUm + voxelSizeUm * 0.5f));
        }

        // ---- Deduplicate / merge ----
        var finalZones = new FastList<ActiveZone>(512);
        float mergeRadSq = radiusUm * radiusUm;
        bool[] merged = new bool[rawCenters.Count];

        for (int i = 0; i < rawCenters.Count; i += 1)
        {
            if (merged[i]) continue;

            Vector3 sumPos = rawCenters[i];
            int cnt = 1;
            merged[i] = true;

            for (int j = i + 1; j < rawCenters.Count; j += 1)
            {
                if (!merged[j] && Vector3.DistanceSquared(rawCenters[i], rawCenters[j]) <= mergeRadSq)
                {
                    sumPos += rawCenters[j];
                    cnt += 1;
                    merged[j] = true;
                }
            }

            finalZones.Add(new ActiveZone { Center = sumPos / cnt });
        }

        return finalZones;
    }

    // ============================================================
    //  ПОИСК ЗОН КОНВЕРГЕНЦИИ (Зоны 3)
    //
    //  Для каждой Зоны 2 ищем ближайшую Зону 1 в радиусе
    //  convergenceRadiusUm. Если находим — создаём Зону 3
    //  в середине между ними. Дубликаты отбрасываются.
    // ============================================================
    private static FastList<ActiveZone>? FindConvergenceZones(
        FastList<ActiveZone>? pyramidalZones,
        FastList<ActiveZone>? tcZones,
        float convergenceRadiusUm)
    {
        if (pyramidalZones is null || pyramidalZones.Count == 0)
            return null;
        if (tcZones is null || tcZones.Count == 0)
            return null;

        float radSq = convergenceRadiusUm * convergenceRadiusUm;
        var result = new FastList<ActiveZone>(64);
        bool[] tcUsed = new bool[tcZones.Count];

        for (int p = 0; p < pyramidalZones.Count; p += 1)
        {
            Vector3 pCenter = pyramidalZones[p].Center;

            for (int t = 0; t < tcZones.Count; t += 1)
            {
                if (tcUsed[t]) continue;

                float distSq = Vector3.DistanceSquared(pCenter, tcZones[t].Center);
                if (distSq <= radSq)
                {
                    // Зона конвергенции — центр между двумя зонами
                    result.Add(new ActiveZone
                    {
                        Center = (pCenter + tcZones[t].Center) * 0.5f
                    });
                    tcUsed[t] = true; // каждая ТК-зона участвует не более одного раза
                }
            }
        }

        return result.Count > 0 ? result : null;
    }

    private readonly FastList<int> _activePyramidalAxons = new FastList<int>(capacity: PyramidalAxonsCount);
    private readonly FastList<int> _activeTcAxons = new FastList<int>(capacity: ThalamocorticalInput.TotalAxonCount);

    // Кэшированный тензор для переиспользования памяти
    private TensorBuffer? _gridTensorBuffer;

    private TensorBuffer? _weightTensorBuffer;
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