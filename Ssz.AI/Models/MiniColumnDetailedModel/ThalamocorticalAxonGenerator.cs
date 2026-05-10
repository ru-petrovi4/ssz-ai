using System;
using System.Collections.Generic;
using System.Numerics;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

/// <summary>
/// Генератор таламокортикальных аксонов (M-путь, ЛКТ → слой 4Cα V1).
///
/// Анатомические параметры (данные по приматам, применимы к человеку):
///
/// Слой 4Cα первичной зрительной коры (V1):
///   - Верхняя граница:  Z = LAYER_TOP_Z    = -800 мкм (от поверхности коры)
///   - Нижняя граница:   Z = LAYER_BOTTOM_Z = -1000 мкм
///   - Толщина слоя:     ~200 мкм
///
/// Входящий ствол аксона:
///   - Начинается в белом веществе: Z = AXON_START_Z = -1350 мкм
///   - Идёт вертикально (вдоль оси Z) с небольшим случайным отклонением
///     в XY, имитируя лёгкую косинность волокон зрительной лучистости
///
/// Горизонтальный арбор (основное ветвление в слое 4Cα):
///   - Начинается при входе в слой (Z ≈ -1000 мкм)
///   - Глубина ветвления в слое: Z от -1000 до -800 мкм (±ARBOR_Z_SPREAD)
///   - Диаметр арбора:
///       мин:    ARBOR_RADIUS_MIN = 300 мкм (=600 мкм диаметр)
///       макс:   ARBOR_RADIUS_MAX = 600 мкм (=1200 мкм диаметр, Blasdel & Lund 1983)
///       среднее: ~400 мкм (=800 мкм диаметр)
///   - Число первичных ветвей: случайно 2–4 для каждого аксона.
///     Если первичных ветвей 3 или 4, они организованы через бинарные
///     узлы-развилки (см. BuildPrimaryFan), так что каждый AxonPoint
///     никогда не имеет более двух дочерних узлов (Next.Size == 2).
///   - Последующие уровни: строго бинарное ветвление (2 ветви).
///   - Уровней ветвления после первичного веера: BRANCH_LEVELS = 2
///   - Ветви почти горизонтальны (уклон по Z ±ARBOR_Z_SPREAD/BRANCH_LEVELS)
///   - Угловое распределение ветвей: равномерное по азимуту + случайный шум
///
/// Синапсы (бутончики, boutons en passant):
///   - Распределены вдоль всех ветвей с шагом SYNAPSE_STEP_MKM = 10 мкм
///     (диапазон: 8–15 мкм по данным по приматам)
///   - Дополнительные концевые кластеры TERMINAL_CLUSTER_COUNT = 3 синапса
///     на каждом конечном кончике ветви
///   - Типичное число синапсов на один M-аксон: ~500–1500
///     (оценка из плотности 0.46×10^8 синапсов/мм³ в 4Cα, Garcia-Marin et al. 2019)
///
/// Расположение аксонов в пространстве коры:
///   - Аксоны размещены на гексагональной решётке в плоскости XY
///   - Шаг решётки GRID_SPACING_MKM = 500 мкм
///     (соответствует шагу окулярно-доминантных колонок ~570 мкм, Hubel et al. 1978)
///   - Каждый аксон соответствует нейрону ЛКТ с центром рецептивного поля,
///     ретинотопически спроецированным в данную точку коры
///   - N ближайших к началу координат отбираются по расстоянию в плоскости XY
/// </summary>
public static class ThalamocorticalAxonGenerator
{
    // ─── Геометрия слоя 4Cα ────────────────────────────────────────────────
    /// <summary>Верхняя граница слоя 4Cα (ближе к поверхности). Мкм.</summary>
    private const float LAYER_TOP_Z = -800f;

    /// <summary>Нижняя граница слоя 4Cα (ближе к белому веществу). Мкм.</summary>
    private const float LAYER_BOTTOM_Z = -1000f;

    // ─── Параметры ствола аксона ────────────────────────────────────────────
    /// <summary>
    /// Z-координата начала аксона (в белом веществе под корой). Мкм.
    /// Зрительная лучистость подходит снизу; выбрано с запасом под слоем.
    /// </summary>
    private const float AXON_START_Z = -1350f;

    /// <summary>
    /// Максимальное случайное горизонтальное смещение ствола на пути
    /// от белого вещества до слоя (имитация лёгкой косинности волокон). Мкм.
    /// </summary>
    private const float TRUNK_XY_DRIFT = 30f;

    /// <summary>Шаг точек вдоль ствола аксона. Мкм.</summary>
    private const float TRUNK_STEP_MKM = 20f;

    // ─── Параметры горизонтального арбора ───────────────────────────────────
    /// <summary>
    /// Минимальный радиус горизонтального арбора (половина минимального диаметра 600 мкм).
    /// Источник: Blasdel & Lund 1983, Lund et al. 2003. Мкм.
    /// </summary>
    private const float ARBOR_RADIUS_MIN = 300f;

    /// <summary>
    /// Максимальный радиус горизонтального арбора (половина максимального диаметра 1200 мкм).
    /// Крупнейшие M-аксоны могут перекрывать до 3 окулярно-доминантных полос.
    /// Источник: Blasdel & Lund 1983. Мкм.
    /// </summary>
    private const float ARBOR_RADIUS_MAX = 600f;

    /// <summary>
    /// Разброс Z-координат внутри арбора (ветви не идеально горизонтальны).
    /// Ветви занимают всю толщину слоя 4Cα (~200 мкм), поэтому ±100 мкм. Мкм.
    /// </summary>
    private const float ARBOR_Z_SPREAD = 100f;

    /// <summary>
    /// Число уровней бинарного ветвления ПОСЛЕ первичного веера.    
    /// </summary>
    private const int BRANCH_LEVELS = 10;

    /// <summary>Шаг точек вдоль ветвей арбора. Мкм.</summary>
    private const float BRANCH_STEP_MKM = 15f;

    // ─── Параметры синапсов ─────────────────────────────────────────────────
    /// <summary>
    /// Средний шаг между синапсами (boutons en passant) вдоль ветви. Мкм.
    /// Диапазон 8–15 мкм по данным для M-аксонов приматов.
    /// Источник: Garcia-Marin et al. 2019; Peters & Palay 1996.
    /// </summary>
    private const float SYNAPSE_STEP_MKM = 10f;

    /// <summary>
    /// Случайный разброс шага синапсов (±половина). Мкм.
    /// Обеспечивает нерегулярное, биологически реалистичное распределение.
    /// </summary>
    private const float SYNAPSE_STEP_JITTER = 4f;

    /// <summary>
    /// Число дополнительных синапсов в концевом кластере каждой ветви.
    /// Концевые утолщения аксона (terminal boutons) дают несколько контактов.
    /// </summary>
    private const int TERMINAL_CLUSTER_COUNT = 3;

    /// <summary>Радиус концевого кластера (разброс терминалей вокруг кончика). Мкм.</summary>
    private const float TERMINAL_CLUSTER_RADIUS = 5f;

    // ─── Пространственное расположение аксонов ──────────────────────────────
    /// <summary>    
    /// Каждый «узел» решётки — центр арбора одного M-нейрона ЛКТ.
    /// нужно ~63–100
    /// </summary>
    private const float GRID_SPACING_MKM = 63f;

    /// <summary>
    /// Радиус поиска на решётке: насколько далеко от (0,0) берём узлы,
    /// чтобы точно найти N ближайших.
    /// </summary>
    private const int GRID_SEARCH_RADIUS = 10;

    // ───────────────────────────────────────────────────────────────────────
    /// <summary>
    /// Генерирует N ближайших к началу координат таламокортикальных аксонов
    /// (M-путь ЛКТ → слой 4Cα V1 человека).
    /// </summary>
    /// <param name="random">Генератор случайных чисел.</param>
    /// <param name="n">
    ///   Количество аксонов. Аксоны отбираются по расстоянию центра арбора
    ///   от точки (0, 0) в плоскости XY (ретинотопический центр поля).
    /// </param>
    /// <returns>Массив из N аксонов, отсортированных по расстоянию от (0,0).</returns>
    public static Axon[] Generate(Random random, int n)
    {
        var centers = GenerateHexGridCenters(random);

        centers.Sort((a, b) =>
            (a.X * a.X + a.Y * a.Y).CompareTo(b.X * b.X + b.Y * b.Y));

        if (centers.Count < n)
            throw new InvalidOperationException(
                $"На сетке только {centers.Count} узлов, запрошено {n}. " +
                $"Увеличьте GRID_SEARCH_RADIUS.");

        var axons = new Axon[n];
        for (int i = 0; i < n; i++)
            axons[i] = BuildAxon(centers[i], random);

        return axons;
    }

    // ── Построение одного аксона ────────────────────────────────────────────

    private static Axon BuildAxon(Vector2 arborCenter, Random random)
    {
        float arborRadius = RandomArborRadius(random);

        // Случайное число первичных ветвей: 2, 3 или 4
        int primaryBranches = random.Next(2, 5); // [2, 4] включительно

        float startX = arborCenter.X + (float)(random.NextDouble() - 0.5) * 2 * TRUNK_XY_DRIFT;
        float startY = arborCenter.Y + (float)(random.NextDouble() - 0.5) * 2 * TRUNK_XY_DRIFT;
        var rootPos = new Vector3(startX, startY, AXON_START_Z);

        AxonPoint root = new AxonPoint(rootPos);

        AxonPoint layerEntry = BuildTrunk(root, arborCenter, random);

        var synapsePositions = new List<Vector3>();
        BuildPrimaryFan(layerEntry, arborCenter, arborRadius, primaryBranches, random, synapsePositions);

        var synapses = new Synapse[synapsePositions.Count];
        for (int i = 0; i < synapsePositions.Count; i++)
            synapses[i] = new Synapse(synapsePositions[i]);

        return new Axon(root, synapses);
    }

    // ── Ствол аксона (белое вещество → вход в 4Cα) ─────────────────────────

    private static AxonPoint BuildTrunk(AxonPoint root, Vector2 arborCenter, Random random)
    {
        AxonPoint current = root;
        float z = AXON_START_Z + TRUNK_STEP_MKM;

        while (z < LAYER_BOTTOM_Z)
        {
            float t = (z - AXON_START_Z) / (LAYER_BOTTOM_Z - AXON_START_Z);
            float x = root.Position.X + t * (arborCenter.X - root.Position.X);
            float y = root.Position.Y + t * (arborCenter.Y - root.Position.Y);

            x += (float)(random.NextDouble() - 0.5) * 5f;
            y += (float)(random.NextDouble() - 0.5) * 5f;

            var next = new AxonPoint(new Vector3(x, y, z));
            current.AddNext(next);
            current = next;
            z += TRUNK_STEP_MKM;
        }

        var entry = new AxonPoint(new Vector3(arborCenter.X, arborCenter.Y, LAYER_BOTTOM_Z));
        current.AddNext(entry);
        return entry;
    }

    // ── Первичный веер (2–4 ветви через бинарные развилки) ──────────────────

    /// <summary>
    /// Строит первичный веер из <paramref name="primaryBranches"/> (2–4) ветвей,
    /// соблюдая ограничение NextAxonPoints.Size == 2.
    ///
    /// Стратегия:
    ///   - 2 ветви: обычная бинарная развилка прямо из <paramref name="start"/>.
    ///   - 3 ветви: одна ветвь выходит прямо из <paramref name="start"/>,
    ///     вторая точка-развилка (fork) присоединяется как второй дочерний узел
    ///     и даёт ещё 2 ветви.
    ///   - 4 ветви: два узла-развилки (fork0, fork1), оба дочерних у <paramref name="start"/>,
    ///     каждый разветвляется на 2 ветви.
    ///
    /// Таким образом каждый AxonPoint имеет не более двух дочерних узлов.
    /// </summary>
    private static void BuildPrimaryFan(
        AxonPoint start,
        Vector2 arborCenter,
        float arborRadius,
        int primaryBranches,
        Random random,
        List<Vector3> synapses)
    {
        // Равномерно распределённые базовые углы + шум ±15°
        double baseAngle = random.NextDouble() * 2 * Math.PI;
        var angles = new double[primaryBranches];
        for (int i = 0; i < primaryBranches; i++)
            angles[i] = baseAngle + i * (2 * Math.PI / primaryBranches)
                        + (random.NextDouble() - 0.5) * 0.52; // ±~15°

        switch (primaryBranches)
        {
            case 2:
                // Прямая бинарная развилка из start
                BuildBranch(start, angles[0], arborRadius, 0, random, synapses);
                BuildBranch(start, angles[1], arborRadius, 0, random, synapses);
                break;

            case 3:
                // Первая ветвь — прямо из start
                BuildBranch(start, angles[0], arborRadius, 0, random, synapses);
                // Узел-развилка на небольшом расстоянии от start по среднему направлению
                {
                    AxonPoint fork = MakeForkNode(start, angles[1], angles[2], random);
                    start.AddNext(fork);
                    BuildBranch(fork, angles[1], arborRadius, 0, random, synapses);
                    BuildBranch(fork, angles[2], arborRadius, 0, random, synapses);
                }
                break;

            case 4:
                // Два узла-развилки, оба дочерних у start
                {
                    AxonPoint fork0 = MakeForkNode(start, angles[0], angles[1], random);
                    AxonPoint fork1 = MakeForkNode(start, angles[2], angles[3], random);
                    start.AddNext(fork0);
                    start.AddNext(fork1);
                    BuildBranch(fork0, angles[0], arborRadius, 0, random, synapses);
                    BuildBranch(fork0, angles[1], arborRadius, 0, random, synapses);
                    BuildBranch(fork1, angles[2], arborRadius, 0, random, synapses);
                    BuildBranch(fork1, angles[3], arborRadius, 0, random, synapses);
                }
                break;
        }
    }

    /// <summary>
    /// Создаёт промежуточный узел-развилку между точкой <paramref name="from"/>
    /// и направлением, усреднённым между <paramref name="angle0"/> и <paramref name="angle1"/>.
    /// Узел смещён на небольшое расстояние от from (BRANCH_STEP_MKM * 2),
    /// чтобы развилка выглядела анатомически правдоподобно.
    /// Синапсы на промежуточных узлах-развилках не создаются.
    /// </summary>
    private static AxonPoint MakeForkNode(AxonPoint from, double angle0, double angle1, Random random)
    {
        double midAngle = (angle0 + angle1) / 2.0;
        float dist = BRANCH_STEP_MKM * 2f;
        float x = from.Position.X + (float)Math.Cos(midAngle) * dist;
        float y = from.Position.Y + (float)Math.Sin(midAngle) * dist;
        float z = from.Position.Z + (float)(random.NextDouble() - 0.5) * 6f;
        z = Math.Max(z, LAYER_BOTTOM_Z);
        z = Math.Min(z, LAYER_TOP_Z);
        return new AxonPoint(new Vector3(x, y, z));
    }

    // ── Рекурсивное бинарное ветвление ──────────────────────────────────────

    /// <summary>
    /// Строит одну ветвь арбора начиная от <paramref name="start"/>
    /// в направлении <paramref name="angle"/> и рекурсивно добавляет
    /// бинарные подветви.
    ///
    /// Каждый AxonPoint цепочки имеет ровно 1 дочерний узел (следующая точка
    /// на прямом отрезке); только конечная точка ветви при необходимости
    /// даёт 2 дочерних узла (бинарная развилка следующего уровня).
    /// Таким образом инвариант NextAxonPoints.Size == 2 соблюдается строго.
    /// </summary>
    private static void BuildBranch(
        AxonPoint start,
        double angle,
        float arborRadius,
        int level,
        Random random,
        List<Vector3> synapses)
    {
        float segmentLength = arborRadius / (float)Math.Pow(2, level);
        if (level == 0) segmentLength *= 0.6f;

        float dx = (float)Math.Cos(angle);
        float dy = (float)Math.Sin(angle);

        float branchZ = LAYER_BOTTOM_Z
            + (float)(random.NextDouble()) * ARBOR_Z_SPREAD
            + (level * ARBOR_Z_SPREAD / (BRANCH_LEVELS + 1));
        branchZ = Math.Min(branchZ, LAYER_TOP_Z);

        AxonPoint current = start;
        float distCovered = 0f;
        float x = start.Position.X;
        float y = start.Position.Y;
        float z = start.Position.Z;
        float zStep = (branchZ - z) / (segmentLength / BRANCH_STEP_MKM + 1);

        float nextSynapseAt = SYNAPSE_STEP_MKM
            + (float)(random.NextDouble() - 0.5) * 2 * SYNAPSE_STEP_JITTER;

        while (distCovered < segmentLength)
        {
            float step = Math.Min(BRANCH_STEP_MKM, segmentLength - distCovered);
            x += dx * step;
            y += dy * step;
            z += zStep * (step / BRANCH_STEP_MKM);
            x += (float)(random.NextDouble() - 0.5) * 3f;
            y += (float)(random.NextDouble() - 0.5) * 3f;

            var pt = new AxonPoint(new Vector3(x, y, z));
            current.AddNext(pt);
            current = pt;

            distCovered += step;

            if (distCovered >= nextSynapseAt)
            {
                synapses.Add(new Vector3(
                    x + (float)(random.NextDouble() - 0.5) * 1.5f,
                    y + (float)(random.NextDouble() - 0.5) * 1.5f,
                    z + (float)(random.NextDouble() - 0.5) * 1.5f));
                nextSynapseAt = distCovered + SYNAPSE_STEP_MKM
                    + (float)(random.NextDouble() - 0.5) * 2 * SYNAPSE_STEP_JITTER;
            }
        }

        // Концевой кластер синапсов (terminal boutons)
        for (int t = 0; t < TERMINAL_CLUSTER_COUNT; t++)
        {
            synapses.Add(new Vector3(
                x + (float)(random.NextDouble() - 0.5) * 2 * TERMINAL_CLUSTER_RADIUS,
                y + (float)(random.NextDouble() - 0.5) * 2 * TERMINAL_CLUSTER_RADIUS,
                z + (float)(random.NextDouble() - 0.5) * 2 * TERMINAL_CLUSTER_RADIUS));
        }

        // Рекурсивное бинарное ветвление (строго 2 дочерних у current)
        if (level < BRANCH_LEVELS - 1)
        {
            double spread = Math.PI / 5.0; // угол расхождения дочерних ветвей ~36°
            BuildBranch(current, angle - spread, arborRadius, level + 1, random, synapses);
            BuildBranch(current, angle + spread, arborRadius, level + 1, random, synapses);
        }
    }

    // ── Вспомогательные методы ───────────────────────────────────────────────

    private static List<Vector2> GenerateHexGridCenters(Random random)
    {
        var centers = new List<Vector2>();
        float h = GRID_SPACING_MKM;
        float hx = h;
        float hy = h * (float)Math.Sqrt(3) / 2f;

        for (int row = -GRID_SEARCH_RADIUS; row <= GRID_SEARCH_RADIUS; row++)
        {
            for (int col = -GRID_SEARCH_RADIUS; col <= GRID_SEARCH_RADIUS; col++)
            {
                float x = col * hx + (row % 2 != 0 ? hx / 2f : 0f);
                float y = row * hy;
                x += (float)(random.NextDouble() - 0.5) * 2 * h * 0.15f;
                y += (float)(random.NextDouble() - 0.5) * 2 * h * 0.15f;
                centers.Add(new Vector2(x, y));
            }
        }
        return centers;
    }

    private static float RandomArborRadius(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        float radius = 400f + (float)(normal * 70f);
        return Math.Clamp(radius, ARBOR_RADIUS_MIN, ARBOR_RADIUS_MAX);
    }
}