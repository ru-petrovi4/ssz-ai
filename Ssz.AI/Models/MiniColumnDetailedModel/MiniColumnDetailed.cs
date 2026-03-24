using Avalonia.Media;
using Ssz.Utils;
using Ssz.Utils.Avalonia.Model3D;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

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
public sealed class MiniColumnDetailed
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
    public const int SynapsesPerAxon = 2_000; // Orig^10_000;

    public List<ActiveZone>? Temp_ActiveZones;

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
    //  ПРОСТРАНСТВЕННЫЙ ИНДЕКС ДЛЯ БЫСТРОГО ПОИСКА СИНАПСОВ
    //
    //  Реализован как словарь: ключ = целочисленная 3D-ячейка
    //  (ix, iy, iz) пространственной решётки, значение = список
    //  (индекс_аксона, индекс_синапса).
    //
    //  Это позволяет при поиске зон радиуса R перебирать только
    //  синапсы в ближайших ячейках, а не все 200 × 10 000 = 2M.
    // ----------------------------------------------------------
    private readonly Dictionary<(int, int, int), List<(int axonIdx, int synIdx)>> _spatialIndex;

    /// <summary>Размер ячейки пространственного индекса (мкм).</summary>
    private float _cellSizeUm;

    // ----------------------------------------------------------
    //  ГЕНЕРАТОР СЛУЧАЙНЫХ ЧИСЕЛ
    //  Инициализируется с фиксированным seed для воспроизводимости.
    // ----------------------------------------------------------
    private readonly Random _random;

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
        Axons = new Axon[AxonCount];
        _spatialIndex = new Dictionary<(int, int, int), List<(int, int)>>(
            capacity: AxonCount * SynapsesPerAxon / 10); // ~2M / 10 оценка

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

        // ----------------------------------------------------------
        //  ШАГ 3: Построить пространственный индекс по синапсам.
        //  Ячейка = cube со стороной _cellSize.
        //  После заполнения всех аксонов _cellSize берём
        //  как среднее расстояние между синапсами * 4.
        // ----------------------------------------------------------
        _cellSizeUm = 15.0f; // мкм, исходя из плотности синапсов
        BuildSpatialIndex();
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
                    x = (float)(_random.NextDouble() * 2.0 - 1.0) * ColumnRadiusUm;
                    y = (float)(_random.NextDouble() * 2.0 - 1.0) * ColumnRadiusUm;
                }
                while (x * x + y * y > ColumnRadiusUm * ColumnRadiusUm);

                float z = zMin + (float)_random.NextDouble() * (zMax - zMin);
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
        float aisLength = 60.0f + (float)_random.NextDouble() * 40.0f; // 60–100 мкм
        int aisPoints = 4;
        float aisStep = aisLength / aisPoints;

        AxonPoint current = root;
        for (int p = 1; p <= aisPoints; p += 1)
        {
            // Небольшое горизонтальное отклонение (реалистичная кривизна)
            float jitter = 1.5f;
            var pos = new Vector3(
                somaPos.X + (float)(_random.NextDouble() - 0.5) * jitter,
                somaPos.Y + (float)(_random.NextDouble() - 0.5) * jitter,
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
        float angle = (float)(_random.NextDouble() * Math.PI * 2.0); // случайный азимут
        var initDir = new Vector3(
            MathF.Cos(angle) * 0.9f,
            MathF.Sin(angle) * 0.9f,
            0.2f + (float)_random.NextDouble() * 0.2f  // небольшой вертикальный компонент
        );
        initDir = Vector3.Normalize(initDir);

        growthStack.Push((current, totalBranchPoints, initDir));

        while (growthStack.Count > 0)
        {
            var (node, branchesLeft, direction) = growthStack.Pop();

            // Длина текущего сегмента (варьируется биологически)
            float segLen = MeanSegmentLengthUm * (0.6f + (float)_random.NextDouble() * 0.8f);
            float stepLen = segLen / PointsPerSegment;

            // Рост сегмента от node до его конца
            AxonPoint segEnd = GrowSegment(node, direction, stepLen, PointsPerSegment);

            // Ветвление: если ещё есть ветвления — делим на 2 ветки
            if (branchesLeft > 0)
            {
                int branch1 = branchesLeft / 2;
                int branch2 = branchesLeft - branch1 - 1;

                // Угол расходящихся ветвей: ~30–60 градусов
                float spreadAngle = 0.4f + (float)(_random.NextDouble() * 0.5f); // ~23–52°
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
    /// Размещает 10 000 исходящих синапсов вдоль всего дерева аксона.
    ///
    /// Биологически: синапсы расположены на терминальных бутонах
    /// (en passant synapses) вдоль аксона, плотнее на дистальных
    /// ветвях. Координаты синапсов = координата точки аксона +
    /// небольшое случайное смещение (~0.5–2 мкм).
    /// </summary>
    private Synapse[] PlaceSynapses(AxonPoint root)
    {
        // Сначала собираем все точки дерева
        var allPoints = new List<AxonPoint>(capacity: 512);
        var traversalStack = new Stack<AxonPoint>(128);
        traversalStack.Push(root);

        while (traversalStack.Count > 0)
        {
            var pt = traversalStack.Pop();
            allPoints.Add(pt);
            foreach (var child in pt.Next)
                traversalStack.Push(child);
        }

        int totalPoints = allPoints.Count;
        var synapses = new Synapse[SynapsesPerAxon];

        // Распределяем синапсы по точкам аксона
        // Синапсы размещаются рядом с точками аксона + малый jitter
        for (int s = 0; s < SynapsesPerAxon; s += 1)
        {
            // Выбираем случайную точку аксона (равномерно)
            int ptIdx = _random.Next(totalPoints);
            var basePos = allPoints[ptIdx].Position;

            // Добавляем небольшое смещение (~0.5–2 мкм) для реализма
            // Синаптические бутоны расположены на небольшом расстоянии
            // от центральной оси аксона
            float jitter = 1.5f; // мкм
            var synPos = new Vector3(
                basePos.X + (float)(_random.NextDouble() - 0.5) * jitter * 2f,
                basePos.Y + (float)(_random.NextDouble() - 0.5) * jitter * 2f,
                basePos.Z + (float)(_random.NextDouble() - 0.5) * jitter * 2f
            );

            synapses[s] = new Synapse(synPos);
        }

        return synapses;
    }

    // ============================================================
    //  ПОСТРОЕНИЕ ПРОСТРАНСТВЕННОГО ИНДЕКСА
    // ============================================================
    /// <summary>
    /// Строит пространственный хэш-индекс для всех синапсов.
    /// Ключ = целочисленная ячейка (ix, iy, iz).
    /// Ячейка имеет размер _cellSize мкм.
    ///
    /// Это позволяет выполнять поиск за O(k) вместо O(2M),
    /// где k — число синапсов в нескольких соседних ячейках.
    /// </summary>
    private void BuildSpatialIndex()
    {
        for (int a = 0; a < AxonCount; a += 1)
        {
            var synapses = Axons[a].Synapses;
            for (int s = 0; s < SynapsesPerAxon; s += 1)
            {
                var cell = GetCell(synapses[s].Position);
                if (!_spatialIndex.TryGetValue(cell, out var list))
                {
                    list = new List<(int, int)>(capacity: 8);
                    _spatialIndex[cell] = list;
                }
                list.Add((a, s));
            }
        }
    }

    // ============================================================
    //  ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: ЯЧЕЙКА ПРОСТРАНСТВЕННОГО ИНДЕКСА
    // ============================================================
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private (int, int, int) GetCell(Vector3 pos)
    {
        int ix = (int)MathF.Floor(pos.X / _cellSizeUm);
        int iy = (int)MathF.Floor(pos.Y / _cellSizeUm);
        int iz = (int)MathF.Floor(pos.Z / _cellSizeUm);
        return (ix, iy, iz);
    }

    // ============================================================
    //  ОСНОВНОЙ МЕТОД: ПОИСК АКТИВНЫХ ЗОН
    // ============================================================
    /// <summary>
    /// Находит все области заданного радиуса, в которых активны
    /// N или более синапсов от РАЗНЫХ активных аксонов.
    ///
    /// Алгоритм:
    ///   1. Определить множество активных аксонов по вектору bits.
    ///   2. Собрать все синапсы активных аксонов.
    ///   3. Для каждого активного синапса проверить его окрестность
    ///      в пространственном индексе.
    ///   4. Подсчитать, сколько РАЗНЫХ активных аксонов имеют
    ///      синапс в радиусе R от данного синапса.
    ///   5. Если >= N — центр этого синапса является центром
    ///      активной зоны.
    ///   6. Дедупликация зон (merge overlapping).
    ///
    /// Сложность: O(A_active * S_per_axon * k_neighbors),
    /// где k_neighbors — среднее число синапсов в (2R/cellSize)^3 ячейках.
    /// </summary>
    /// <param name="activityBits">
    /// Битовый вектор длиной 200. Единица = активный аксон.
    /// Ожидается примерно 30 единиц.
    /// </param>
    /// <param name="radius">
    /// Радиус поиска в мкм. Например, 10–30 мкм.
    /// </param>
    /// <param name="minActiveAxons">
    /// Минимальное число N уникальных активных аксонов,
    /// синапсы которых должны попасть в зону.
    /// </param>
    /// <returns>
    /// Список найденных активных зон (дедуплицированных).
    /// </returns>
    public void FindActiveZones(
        float[] activityBits,
        float radius,
        int minActiveAxons)
    {
        if (activityBits.Length != AxonCount)
            throw new ArgumentException(
                $"Длина вектора активности должна быть {AxonCount}, получено {activityBits.Length}.");

        // ----------------------------------------------------------
        //  ШАГ 1: Множество индексов активных аксонов
        // ----------------------------------------------------------
        var activeAxonSet = new HashSet<int>(capacity: 64);
        for (int i = 0; i < AxonCount; i += 1)
        {
            if (activityBits[i] > 0.5f)
                activeAxonSet.Add(i);
        }

        int activeCount = activeAxonSet.Count;
        if (activeCount < minActiveAxons)
        {
            Temp_ActiveZones = null;
            return; // Невозможно выполнить условие
        }            

        // ----------------------------------------------------------
        //  ШАГ 2: Параметры поиска в пространственном индексе
        //  Нужно проверить все ячейки в кубе со стороной 2R
        // ----------------------------------------------------------
        float radiusSq = radius * radius;
        int cellSpan = (int)MathF.Ceiling(radius / _cellSizeUm);

        // ----------------------------------------------------------
        //  ШАГ 3: Проверяем каждый синапс каждого активного аксона
        //  как потенциальный центр зоны
        // ----------------------------------------------------------
        // Используем словарь центр->зона для объединения близких зон.
        // Для каждого синапса считаем окрестность.
        var rawZones = new List<ActiveZone>(capacity: 1024);

        // Для дедупликации: зоны с центрами, расстояние между
        // которыми < radius, считаем одной зоной (берём первый найденный центр).
        // Используем пространственный хэш для найденных центров.
        var foundCenters = new Dictionary<(int, int, int), ActiveZone>(capacity: 512);

        foreach (int axonIdx in activeAxonSet)
        {
            var synapses = Axons[axonIdx].Synapses;

            for (int s = 0; s < SynapsesPerAxon; s += 1)
            {
                var synPos = synapses[s].Position;

                // --------------------------------------------------
                //  Быстрая проверка: не смотрим в уже найденную зону
                //  (дедупликация на лету через пространственный хэш)
                // --------------------------------------------------
                var centerCell = GetCell(synPos);
                if (foundCenters.ContainsKey(centerCell))
                    continue;

                // --------------------------------------------------
                //  Подсчёт уникальных активных аксонов в радиусе R
                // --------------------------------------------------
                var uniqueActiveInRadius = new HashSet<int>(capacity: minActiveAxons + 4);

                // Обход ячеек в кубе [-cellSpan..+cellSpan]^3
                (int baseX, int baseY, int baseZ) = GetCell(synPos);

                for (int dz = -cellSpan; dz <= cellSpan; dz += 1)
                for (int dy = -cellSpan; dy <= cellSpan; dy += 1)
                for (int dx = -cellSpan; dx <= cellSpan; dx += 1)
                {
                    var neighborCell = (baseX + dx, baseY + dy, baseZ + dz);
                    if (!_spatialIndex.TryGetValue(neighborCell, out var neighbors))
                        continue;

                    foreach (var (nAxonIdx, nSynIdx) in neighbors)
                    {
                        // Только активные аксоны интересуют
                        if (!activeAxonSet.Contains(nAxonIdx))
                            continue;

                        // Проверяем точное расстояние (после быстрой ячейковой фильтрации)
                        var nPos = Axons[nAxonIdx].Synapses[nSynIdx].Position;
                        float distSq = Vector3.DistanceSquared(synPos, nPos);
                        if (distSq <= radiusSq)
                            uniqueActiveInRadius.Add(nAxonIdx);
                    }
                }

                // --------------------------------------------------
                //  Если нашли >= N уникальных активных аксонов — зона!
                // --------------------------------------------------
                if (uniqueActiveInRadius.Count >= minActiveAxons)
                {
                    var zone = new ActiveZone
                    {
                        Center = synPos,
                    };
                    foreach (int idx in uniqueActiveInRadius)
                        zone.ActiveAxonIndices.Add(idx);

                    rawZones.Add(zone);

                    // Отмечаем ячейку как найденную (дедупликация)
                    foundCenters[centerCell] = zone;
                }
            }
        }

        // ----------------------------------------------------------
        //  ШАГ 4: Постобработка — объединение перекрывающихся зон
        //  (greedy merge: идём по списку, объединяем зоны,
        //   центры которых ближе, чем radius/2)
        // ----------------------------------------------------------
        Temp_ActiveZones = MergeOverlappingZones(rawZones, radius * 0.5f);        
    }

    // ============================================================
    //  ОБЪЕДИНЕНИЕ ПЕРЕКРЫВАЮЩИХСЯ ЗОН
    // ============================================================
    /// <summary>
    /// Жадное объединение зон: если центры двух зон ближе mergeRadius,
    /// они считаются одной зоной (берётся центр первой найденной,
    /// индексы аксонов объединяются).
    ///
    /// Реализован за O(n^2) по числу зон. Для типичного случая
    /// (активных зон десятки–сотни) это приемлемо.
    /// Для больших R или больших N можно заменить на KD-tree merge.
    /// </summary>
    private static List<ActiveZone> MergeOverlappingZones(
        List<ActiveZone> zones, float mergeRadius)
    {
        float mergeRadSq = mergeRadius * mergeRadius;
        int n = zones.Count;
        var merged = new bool[n];
        var result = new List<ActiveZone>(n);

        for (int i = 0; i < n; i += 1)
        {
            if (merged[i]) continue;

            var zone = zones[i];
            for (int j = i + 1; j < n; j += 1)
            {
                if (merged[j]) continue;
                float distSq = Vector3.DistanceSquared(zone.Center, zones[j].Center);
                if (distSq <= mergeRadSq)
                {
                    // Объединяем аксоны
                    foreach (int idx in zones[j].ActiveAxonIndices)
                        zone.ActiveAxonIndices.Add(idx);
                    merged[j] = true;
                }
            }
            result.Add(zone);
        }

        return result;
    }    
}