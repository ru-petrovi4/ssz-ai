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
//  Параметры, соответствующие анатомии человека (V1):
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
    public const int SynapsesPerAxon = 10_000;

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
    //  СЛОЙ КОРЫ: ENUM ДЛЯ СЛОЙ-СПЕЦИФИЧНОГО ПОВЕДЕНИЯ АКСОНОВ
    // ============================================================
    /// <summary>
    /// Корковые слои (II–VI) с учётом специфики V1.
    /// Используется для выбора стратегии роста аксона.
    ///
    /// Источники:
    ///   Mohan et al. 2015 (Cerebral Cortex) — морфология пирамидных нейронов по слоям;
    ///   Douglas &amp; Martin 2004 (Annu. Rev. Neurosci.) — функциональные схемы слоёв V1.
    /// </summary>
    private enum CorticalLayer
    {
        /// <summary>Малые пирамидные клетки: горизонтальные транскортикальные коллатерали.</summary>
        LayerII = 0,
        /// <summary>Средние пирамидные клетки: горизонтальные + транскаллозальные проекции.</summary>
        LayerIII = 1,
        /// <summary>Зернистый слой (4A/4B/4C в V1): таламические входы, короткие локальные аксоны.</summary>
        LayerIV = 2,
        /// <summary>Крупные пирамидные (клетки Беца): нисходящий проекционный аксон + восходящие коллатерали.</summary>
        LayerV = 3,
        /// <summary>Мультиформный слой: нисходящие кортикоталамические аксоны + локальные коллатерали.</summary>
        LayerVI = 4,
    }

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
        //  Возвращаем также слой каждого нейрона для GrowAxon.
        // ----------------------------------------------------------
        var (somaPositions, somaLayers) = GenerateSomaPositions();

        // ----------------------------------------------------------
        //  ШАГ 2: Для каждого нейрона вырастить аксон.
        //  Передаём слой нейрона для слой-специфичной морфологии.
        // ----------------------------------------------------------
        for (int i = 0; i < AxonCount; i += 1)
        {
            AxonPoint root = GrowAxon(somaPositions[i], somaLayers[i]);
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
    /// что соответствует реальной корковой анатомии V1.
    /// </summary>
    /// <returns>Пара: массив позиций и массив меток слоёв (CorticalLayer).</returns>
    private (Vector3[] positions, CorticalLayer[] layers) GenerateSomaPositions()
    {
        // Границы слоёв (Z в мкм): слой II-III верхний, VI нижний
        // Примерные относительные толщины слоёв коры:
        // L II: 0–200, L III: 200–600, L IV: 600–900,
        // L V: 900–1400, L VI: 1400–2000
        ReadOnlySpan<(float zMin, float zMax, int neuronCount, CorticalLayer layer)> layers =
        [
            (0f,    200f,   20, CorticalLayer.LayerII),   // слой II
            (200f,  600f,   50, CorticalLayer.LayerIII),  // слой III
            (600f,  900f,   30, CorticalLayer.LayerIV),   // слой IV
            (900f,  1400f,  60, CorticalLayer.LayerV),    // слой V
            (1400f, 2000f,  40, CorticalLayer.LayerVI),   // слой VI
        ];
        // Итого: 200 нейронов распределены по 5 слоям.

        var positions = new Vector3[AxonCount];
        var somaLayers = new CorticalLayer[AxonCount];
        int idx = 0;

        foreach (var (zMin, zMax, count, layer) in layers)
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
                somaLayers[idx] = layer;
                idx += 1;
            }
        }

        return (positions, somaLayers);
    }

    // ============================================================
    //  РОСТ АКСОНА (СЛОЙ-СПЕЦИФИЧНЫЙ)
    // ============================================================
    /// <summary>
    /// Строит дерево аксона, начиная с позиции сомы.
    /// Морфология определяется принадлежностью нейрона к слою коры.
    ///
    /// Стратегии по слоям (V1, Mohan et al. 2015):
    ///
    ///   L II/III — горизонтальные транскортикальные коллатерали:
    ///     AIS идёт вертикально вниз ~40–60 мкм, затем арбор
    ///     распространяется преимущественно горизонтально (горизонт. компонент ~0.95).
    ///     Именно эти нейроны формируют длинные горизонтальные связи
    ///     в пределах V1 (до 6–8 мм).
    ///
    ///   L IV — зернистый слой, таламические реле:
    ///     AIS короткий ~30–50 мкм, арбор строго локальный,
    ///     горизонтальный разброс минимален (горизонт. компонент ~0.80),
    ///     сегменты короче (~390 мкм среднее).
    ///     Источник: Yoshioka et al. 1994 — аксоны L4 V1 ограничены
    ///     пределами одной колонки.
    ///
    ///   L V — крупные пирамидные клетки (клетки Беца), нисходящие проекции:
    ///     AIS длинный ~60–100 мкм, основной аксон уходит вниз (проекционный).
    ///     Дополнительно добавляется восходящая рекуррентная коллатеральная ветвь
    ///     в слои II/III (восходящий компонент -Z в нашей СК).
    ///     Горизонтальный компонент начального арбора умеренный (~0.70),
    ///     значительный вертикальный дрейф вниз (+Z).
    ///
    ///   L VI — кортикоталамические нейроны:
    ///     Нисходящий аксон к белому веществу, локальные коллатерали короткие.
    ///     AIS ~50–80 мкм, горизонт. компонент ~0.60, вертикальный дрейф вниз.
    ///
    /// Биологические источники:
    ///   - Mohan et al. 2015, Cerebral Cortex 26:4839–4858
    ///   - Douglas &amp; Martin 2004, Annu. Rev. Neurosci. 27:419–451
    ///   - Yoshioka et al. 1994, J. Neurosci. 14(11):6652–6671
    ///   - Markram et al. 1997; Larkman &amp; Mason 1990 (коллатераль L5)
    /// </summary>
    private AxonPoint GrowAxon(Vector3 somaPos, CorticalLayer layer)
    {
        // Случайное число точек ветвления: Poisson-подобное ~4.5
        int totalBranchPoints = SamplePoissonBranchCount();

        // Создаём корневой узел в точке сомы
        var root = new AxonPoint(somaPos);

        // ----------------------------------------------------------
        //  СЛОЙ-СПЕЦИФИЧНЫЕ ПАРАМЕТРЫ AIS
        //  AIS всегда направлен вертикально вниз (к белому веществу).
        //  Длина варьируется по слою.
        // ----------------------------------------------------------
        float aisLength;
        switch (layer)
        {
            case CorticalLayer.LayerIV:
                // L4: короткий AIS — локальные интернейронные связи
                aisLength = 30.0f + _random.NextSingle() * 20.0f; // 30–50 мкм
                break;
            case CorticalLayer.LayerV:
            case CorticalLayer.LayerVI:
                // L5/L6: длинный AIS — проекционные нейроны
                aisLength = 60.0f + _random.NextSingle() * 40.0f; // 60–100 мкм
                break;
            default:
                // L2/L3: стандартный AIS
                aisLength = 40.0f + _random.NextSingle() * 20.0f; // 40–60 мкм
                break;
        }

        int aisPoints = 4;
        float aisStep = aisLength / aisPoints;

        AxonPoint current = root;
        for (int p = 1; p <= aisPoints; p += 1)
        {
            // Небольшое горизонтальное отклонение (реалистичная кривизна AIS)
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
        //  СЛОЙ-СПЕЦИФИЧНЫЕ ПАРАМЕТРЫ НАЧАЛЬНОГО НАПРАВЛЕНИЯ АРБОРА
        // ----------------------------------------------------------
        float angle = _random.NextSingle() * MathF.PI * 2.0f; // случайный азимут

        Vector3 initDir;
        switch (layer)
        {
            case CorticalLayer.LayerII:
            case CorticalLayer.LayerIII:
                // L2/L3: СТРОГО горизонтальное распространение.
                // Транскортикальные коллатерали V1 идут горизонтально
                // в плоскости слоя на расстояния до 6–8 мм.
                // Вертикальный компонент минимален (~0.05–0.10).
                initDir = new Vector3(
                    MathF.Cos(angle) * 0.95f,
                    MathF.Sin(angle) * 0.95f,
                    0.05f + _random.NextSingle() * 0.05f
                );
                break;

            case CorticalLayer.LayerIV:
                // L4: ЛОКАЛЬНОЕ горизонтальное распространение.
                // Аксоны L4 V1 ограничены пределами одной колонки/гиперколонки.
                // Горизонтальный компонент умеренный, сегменты короче.
                initDir = new Vector3(
                    MathF.Cos(angle) * 0.80f,
                    MathF.Sin(angle) * 0.80f,
                    0.15f + _random.NextSingle() * 0.10f
                );
                break;

            case CorticalLayer.LayerV:
                // L5: НИСХОДЯЩИЙ проекционный аксон.
                // Основная ветвь уходит вертикально вниз (к белому веществу).
                // Горизонтальный компонент меньше, вертикальный (+Z) значительный.
                initDir = new Vector3(
                    MathF.Cos(angle) * 0.55f,
                    MathF.Sin(angle) * 0.55f,
                    0.55f + _random.NextSingle() * 0.20f
                );
                break;

            case CorticalLayer.LayerVI:
                // L6: КОРТИКОТАЛАМИЧЕСКИЙ нисходящий аксон.
                // Аналогично L5, но вертикальный дрейф ещё сильнее —
                // аксон идёт к таламусу через белое вещество.
                initDir = new Vector3(
                    MathF.Cos(angle) * 0.45f,
                    MathF.Sin(angle) * 0.45f,
                    0.70f + _random.NextSingle() * 0.20f
                );
                break;

            default:
                initDir = new Vector3(
                    MathF.Cos(angle) * 0.9f,
                    MathF.Sin(angle) * 0.9f,
                    0.2f + _random.NextSingle() * 0.2f
                );
                break;
        }

        initDir = Vector3.Normalize(initDir);

        // ----------------------------------------------------------
        //  СЛОЙ-СПЕЦИФИЧНАЯ ДЛИНА СЕГМЕНТОВ
        //  L4 имеет короткие локальные сегменты, L2/L3 — длинные.
        // ----------------------------------------------------------
        float segLenFactor = layer switch
        {
            CorticalLayer.LayerIV => 0.65f,  // L4: короче (~390 мкм средний сегмент)
            CorticalLayer.LayerV => 0.90f,  // L5: умеренные
            CorticalLayer.LayerVI => 0.80f,  // L6: умеренно короче
            _ => 1.00f,  // L2/L3: стандарт
        };

        // ----------------------------------------------------------
        //  РЕКУРСИВНЫЙ РОСТ ВЕТВЕЙ
        //  Используем стек вместо рекурсии для производительности.
        // ----------------------------------------------------------
        var growthStack = new Stack<(AxonPoint node, int branchesLeft, Vector3 direction)>(32);
        growthStack.Push((current, totalBranchPoints, initDir));

        while (growthStack.Count > 0)
        {
            var (node, branchesLeft, direction) = growthStack.Pop();

            // Длина текущего сегмента (варьируется биологически, с поправкой по слою)
            float segLen = MeanSegmentLengthUm * segLenFactor * (0.6f + _random.NextSingle() * 0.8f);
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

        // ----------------------------------------------------------
        //  L5: ВОСХОДЯЩАЯ РЕКУРРЕНТНАЯ КОЛЛАТЕРАЛЬ
        //
        //  У нейронов слоя V характерна возвратная коллатераль,
        //  которая поднимается обратно в слои II/III (апикальная
        //  коллатераль). Она идёт ВВЕРХ (-Z в нашей СК).
        //
        //  Источник: Markram et al. 1997; Larkman & Mason 1990.
        // ----------------------------------------------------------
        if (layer == CorticalLayer.LayerV)
        {
            float collateralAngle = _random.NextSingle() * MathF.PI * 2.0f;
            var collateralDir = new Vector3(
                MathF.Cos(collateralAngle) * 0.5f,
                MathF.Sin(collateralAngle) * 0.5f,
                -0.70f  // ВВЕРХ (отрицательный Z = к поверхности коры)
            );
            collateralDir = Vector3.Normalize(collateralDir);

            // Коллатераль имеет меньше ветвлений, чем основной арбор
            int collateralBranches = Math.Max(1, totalBranchPoints / 3);

            var collateralStack = new Stack<(AxonPoint node, int branchesLeft, Vector3 direction)>(16);
            collateralStack.Push((current, collateralBranches, collateralDir));

            while (collateralStack.Count > 0)
            {
                var (node, branchesLeft, direction) = collateralStack.Pop();

                float segLen = MeanSegmentLengthUm * 0.70f * (0.6f + _random.NextSingle() * 0.8f);
                float stepLen = segLen / PointsPerSegment;

                AxonPoint segEnd = GrowSegment(node, direction, stepLen, PointsPerSegment);

                if (branchesLeft > 0)
                {
                    int branch1 = branchesLeft / 2;
                    int branch2 = branchesLeft - branch1 - 1;

                    float spreadAngle = 0.4f + _random.NextSingle() * 0.5f;
                    Vector3 dir1 = RotateVector(direction, spreadAngle, _random);
                    Vector3 dir2 = RotateVector(direction, -spreadAngle, _random);

                    collateralStack.Push((segEnd, branch1, Vector3.Normalize(dir1)));
                    collateralStack.Push((segEnd, branch2, Vector3.Normalize(dir2)));
                }
            }
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
        // 1. Собираем все отрезки (сегменты) дерева аксона.
        var segments = new List<(Vector3 Start, Vector3 End, float Length)>(capacity: 512);

        var traversalStack = new Stack<AxonPoint>(128);
        traversalStack.Push(root);

        float totalLength = 0f;

        while (traversalStack.Count > 0)
        {
            var pt = traversalStack.Pop();

            if (pt.Next != null)
            {
                for (int i = 0; i < pt.Next.Count; i += 1)
                {
                    var child = pt.Next[i];

                    float length = Vector3.Distance(pt.Position, child.Position);

                    if (length > 0f)
                    {
                        segments.Add((pt.Position, child.Position, length));
                        totalLength += length;
                    }

                    traversalStack.Push(child);
                }
            }
        }

        FastList<Synapse> synapses = new(SynapsesPerAxon);

        // 2. Краевой случай: вырожденный аксон
        if (totalLength <= 0f || segments.Count == 0)
        {
            for (int s = 0; s < SynapsesPerAxon; s += 1)
            {
                synapses.Add(new Synapse(root.Position));
            }
            return synapses.ToArray();
        }

        // 3. Шаг равномерного распределения синапсов
        float step = totalLength / SynapsesPerAxon;

        int currentSegmentIdx = 0;
        float currentDistInSegment = step * 0.5f;

        // 4. Расставляем синапсы равномерно
        for (int s = 0; s < SynapsesPerAxon; s += 1)
        {
            while (currentDistInSegment > segments[currentSegmentIdx].Length &&
                   currentSegmentIdx < segments.Count - 1)
            {
                currentDistInSegment -= segments[currentSegmentIdx].Length;
                currentSegmentIdx += 1;
            }

            var seg = segments[currentSegmentIdx];
            float t = currentDistInSegment / seg.Length;
            var basePos = Vector3.Lerp(seg.Start, seg.End, t);

            // 5. Небольшое случайное смещение (~1.5 мкм) — бутоны en passant
            float jitter = 1.5f;
            var synPos = new Vector3(
                basePos.X + (_random.NextSingle() - 0.5f) * jitter * 2f,
                basePos.Y + (_random.NextSingle() - 0.5f) * jitter * 2f,
                basePos.Z + (_random.NextSingle() - 0.5f) * jitter * 2f
            );

            // Фильтруем синапсы внутри двойного радиуса колонки
            float r = ColumnRadiusUm * 2.0f;
            if (synPos.X > -r && synPos.X < r &&
                synPos.Y > -r && synPos.Y < r)
            {
                synapses.Add(new Synapse(synPos));
            }

            currentDistInSegment += step;
        }

        return synapses.ToArray();
    }

    // ============================================================
    //  ПОИСК АКТИВНЫХ ЗОН (С ИСПОЛЬЗОВАНИЕМ TORCHSHARP)
    // ============================================================
    /// <summary>
    /// Ищет пространственные области (зоны) заданного радиуса,
    /// внутри которых присутствуют синапсы как минимум от N различных активных аксонов.
    /// Алгоритм использует вокселизацию пространства и 3D-свёртку на базе TorchSharp
    /// для точного математического поиска максимумов плотности в любой точке пространства
    /// (а не только с центрами в синапсах), после чего проводит дедупликацию.
    /// </summary>
    /// <param name="activityBits">Вектор, где 1.0f означает активность аксона.</param>
    /// <param name="radiusUm">Радиус поиска (мкм).</param>
    /// <param name="minActiveAxons">Минимальное количество уникальных активных аксонов в радиусе.</param>
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
        float voxelSizeUm = 2.0f; // Шаг сетки (2 мкм) — баланс между точностью и памятью GPU/CPU

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
        //  ШАГ 3: Создание входного тензора (Grid)
        // ----------------------------------------------------------
        long totalVoxels = (long)activeCount * depth * height * width;

        if (_gridTensorBuffer is null)
            _gridTensorBuffer = new TensorBuffer(_device, totalVoxels);
        else
            _gridTensorBuffer.EnsureCapacity(totalVoxels);

        using var gridTensor_Cpu_Flat = _gridTensorBuffer.Tensor_Cpu_Buffer!.slice(0, 0, totalVoxels, 1);
        using var gridTensor_device_Flat = _gridTensorBuffer.Tensor_device_Buffer.slice(0, 0, totalVoxels, 1);

        gridTensor_Cpu_Flat.zero_();

        long spatialSize = (long)depth * height * width;
        long areaSize = (long)height * width;

        var gridTensorData = gridTensor_Cpu_Flat.data<float>();

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
                    gridTensorData[index] = 1.0f;
                }
            }
        }

        gridTensor_device_Flat.copy_(gridTensor_Cpu_Flat);

        Tensor gridTensor_device = gridTensor_device_Flat.view(1, activeCount, depth, height, width);

        // ----------------------------------------------------------
        //  ШАГ 4: Создание сферического ядра для 3D-свёртки
        // ----------------------------------------------------------
        int kRad = (int)MathF.Ceiling(radiusUm / voxelSizeUm);
        int kSize = kRad * 2 + 1; // Нечётный размер ядра для наличия чёткого центра

        long weightElementsCount = activeCount * kSize * kSize * kSize;
        if (_weightTensorBuffer is null)
            _weightTensorBuffer = new TensorBuffer(_device, weightElementsCount);
        else
            _weightTensorBuffer.EnsureCapacity(weightElementsCount);

        using var weightTensor_Cpu_Flat = _weightTensorBuffer.Tensor_Cpu_Buffer!.slice(0, 0, weightElementsCount, 1);
        using var weightTensor_device_Flat = _weightTensorBuffer.Tensor_device_Buffer.slice(0, 0, weightElementsCount, 1);
        weightTensor_Cpu_Flat.zero_();
        var kernelData = weightTensor_Cpu_Flat.data<float>();

        long kSpatial = (long)kSize * kSize * kSize;
        long kArea = (long)kSize * kSize;

        // Формируем сферу. Если дистанция от центра <= радиус, ставим 1.0
        for (int kz = -kRad; kz <= kRad; kz += 1)
        {
            for (int ky = -kRad; ky <= kRad; ky += 1)
            {
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
            }
        }

        weightTensor_device_Flat.copy_(weightTensor_Cpu_Flat);

        // Тензор весов свёртки: [OutChannels=activeCount, InChannels=1 (из-за groups), kD, kH, kW]
        using var weightTensor_device = weightTensor_device_Flat.view(activeCount, 1, kSize, kSize, kSize);

        // ----------------------------------------------------------
        //  ШАГ 5: Выполнение вычислений через TorchSharp
        // ----------------------------------------------------------
        // Depthwise 3D свёртка (каждый аксон обрабатывается независимо)
        using var convResult_device = nn.functional.conv3d(
            gridTensor_device,
            weightTensor_device,
            padding: new long[] { kRad, kRad, kRad },
            groups: activeCount);

        // Бинаризуем результат: если > 0, значит синапсы этого аксона есть в радиусе R
        using var presentMask_device = convResult_device.gt(0.0f).to_type(ScalarType.Float32);

        // Суммируем по каналам (dim=1), чтобы получить число уникальных аксонов
        using var activeAxonsCount_device = presentMask_device.sum(new long[] { 1 }, keepdim: false);

        // Оставляем только воксели, где число аксонов >= N
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

    // Кэшированные тензоры для переиспользования памяти
    private TensorBuffer? _gridTensorBuffer;
    private TensorBuffer? _weightTensorBuffer;
}

// ============================================================
//  БУФЕР ТЕНЗОРА (переиспользуемая память GPU/CPU)
// ============================================================
public class TensorBuffer : IDisposable
{
    public TensorBuffer(Device device, long capacity)
    {
        Device = device;
        Tensor_Buffer_Capacity = capacity;

        Tensor_device_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: device).DetachFromDisposeScope();
        Tensor_Cpu_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: CPU).DetachFromDisposeScope();
    }

    public void EnsureCapacity(long capacity)
    {
        if (capacity <= Tensor_Buffer_Capacity)
            return;

        Tensor_device_Buffer.Dispose();
        Tensor_Cpu_Buffer.Dispose();

        Tensor_Buffer_Capacity = capacity;

        Tensor_device_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: Device).DetachFromDisposeScope();
        Tensor_Cpu_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: CPU).DetachFromDisposeScope();
    }

    public readonly Device Device;

    public Tensor Tensor_device_Buffer;
    public Tensor Tensor_Cpu_Buffer;

    /// <summary>Текущая вместимость (в элементах), чтобы понимать, когда нужно перевыделять память.</summary>
    public long Tensor_Buffer_Capacity = 0;

    public void Dispose()
    {
        Tensor_device_Buffer?.Dispose();
        Tensor_Cpu_Buffer?.Dispose();
    }
}
