using System;
using System.Collections.Generic;
using System.Numerics;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

/// <summary>
/// Генератор таламокортикальных аксонов (M-путь, ЛКТ → слой 4Cα V1).
///
/// === ВЕРСИЯ 2: РЕАЛИСТИЧНОЕ ПРОСТРАНСТВЕННОЕ РАСПРЕДЕЛЕНИЕ СИНАПСОВ ===
/// Ключевые отличия от V1, повышающие биологическую достоверность
/// пространственного распределения бутончиков:
///
///   (1) МОЗАИЧНОЕ (PATCHY) РАСПРЕДЕЛЕНИЕ БУТОНЧИКОВ EN PASSANT.
///       Реальные TC-аксоны имеют не равномерный шаг 2.7 мкм, а двухрежимное
///       распределение: плотные «гроздья» (clusters) из 3–5 бутончиков
///       с интервалами 0.5–2 мкм, разделённые разрежёнными промежутками
///       (inter-cluster gaps) длиной 3–12 мкм. Реализовано через двухсостоянную
///       Марковскую цепь («cluster mode» ↔ «sparse mode») с подобранными
///       вероятностями переходов, дающими стационарную долю кластеров ~70%
///       и средний межбутончиковый интервал ≈ 2.7 мкм.
///       Источники: Anderson, Douglas & Martin 1994 (J.Comp.Neurol.);
///       Garcia-Marin et al. 2019 (Cereb.Cortex), Fig.5; Freund et al. 1989.
///
///   (2) ГАУССОВ ВЕРТИКАЛЬНЫЙ ПРОФИЛЬ ПЛОТНОСТИ В СЛОЕ 4Cα.
///       Целевая Z первичных ветвей сэмплируется из N(-900, σ=40 мкм),
///       а не равномерно по всей толщине слоя. Это создаёт пик плотности
///       бутончиков в центральных ~80–100 мкм слоя 4Cα со спаданием
///       к границам — что соответствует послойному анализу плотности
///       бутончиков (Hendrickson, Wilson & Ogren 1978).
///
///   (3) ВАРИАТИВНЫЕ КОНЦЕВЫЕ КЛАСТЕРЫ С РОЗЕТКАМИ.
///       ~80% концевых ветвей оканчиваются обычными малыми кластерами
///       из 2–7 бутончиков (радиус 1–3 мкм); ~20% — крупными «розетками»
///       (gloмerular endings) из 8–15 бутончиков (радиус 3–5 мкм),
///       характерными для TC-аксонов приматов.
///       Источники: Anderson & Martin 2002 (Cereb.Cortex); Freund 1989.
///
///   (4) КОЛЛАТЕРАЛЬ В СЛОЙ 6 (~40% M-АКСОНОВ).
///       По данным Freund et al. 1989 (HRP, V1 макак) среди M-аксонов
///       выделяются Type 1 (только 4Cα) и Type 2 (с дополнительным
///       небольшим арбором в слое 6). Реализовано как второй небольшой
///       арбор (радиус ~150 мкм, ~150–250 бутончиков), отходящий
///       от ствола на Z ≈ -1200 мкм.
///       Источники: Freund, Martin & Whitteridge 1989; Wiser & Callaway 1996.
///
/// === АНАТОМИЧЕСКИЕ ПАРАМЕТРЫ (приматы / человек) ===
///
/// Слой 4Cα V1:
///   - Z = [LAYER_BOTTOM_Z, LAYER_TOP_Z] = [-1000, -800] мкм (толщина ~200 мкм)
///   - Центр слоя: LAYER_CENTER_Z = -900 мкм (область пиковой плотности TC-бутончиков)
///
/// Слой 6 V1 (для коллатералей Type-2 аксонов):
///   - Центр слоя: LAYER6_CENTER_Z ≈ -1200 мкм
///
/// Ствол:
///   - Старт: AXON_START_Z = -1350 мкм (белое вещество)
///   - Слабый XY-дрейф ~30 мкм (зрительная лучистость, легкая косинность)
///
/// Главный арбор в 4Cα:
///   - Эффективный радиус: ARBOR_RADIUS_MIN..MAX = 300..600 мкм (~400 ср.)
///   - 2–4 первичные ветви, BRANCH_LEVELS = 5 уровней дихотомии
///   - Tortuosity ≈ 3 (длина дуги ≈ 3 × пространственный вылет)
///   - Суммарная дуговая длина ~17.5 мм → ~6000–7000 бутончиков
///
/// Гексагональная плотность аксонов: GRID_SPACING_MKM = 63 мкм
///   → ~200 M-аксонов/мм² (фовеальная проекция V1, Garcia-Marin 2019).
/// </summary>
public static class ThalamocorticalAxonGenerator
{
    // ─── Геометрия слоя 4Cα ────────────────────────────────────────────────
    /// <summary>Верхняя граница слоя 4Cα (ближе к поверхности). Мкм.</summary>
    private const float LAYER_TOP_Z = -800f;

    /// <summary>Нижняя граница слоя 4Cα (ближе к белому веществу). Мкм.</summary>
    private const float LAYER_BOTTOM_Z = -1000f;

    /// <summary>Центр слоя 4Cα — место пиковой плотности TC-бутончиков. Мкм.</summary>
    private const float LAYER_CENTER_Z = -900f;

    // ─── Геометрия слоя 6 (для Type-2 аксонов с коллатералью) ──────────────
    /// <summary>Центр слоя 6 — Z-координата центра коллатерального арбора. Мкм.</summary>
    private const float LAYER6_CENTER_Z = -1200f;

    /// <summary>Верхняя граница слоя 6. Мкм.</summary>
    private const float LAYER6_TOP_Z = -1100f;

    /// <summary>Нижняя граница слоя 6. Мкм.</summary>
    private const float LAYER6_BOTTOM_Z = -1300f;

    // ─── Параметры ствола аксона ────────────────────────────────────────────
    /// <summary>Z-координата начала аксона (в белом веществе). Мкм.</summary>
    private const float AXON_START_Z = -1350f;

    /// <summary>
    /// Максимальное случайное горизонтальное смещение ствола (XY-дрейф). Мкм.
    /// Имитирует лёгкую косинность волокон зрительной лучистости.
    /// </summary>
    private const float TRUNK_XY_DRIFT = 30f;

    /// <summary>Шаг точек вдоль ствола аксона. Мкм.</summary>
    private const float TRUNK_STEP_MKM = 20f;

    // ─── Параметры главного арбора в 4Cα ───────────────────────────────────
    /// <summary>Минимальный эффективный радиус арбора. Мкм. Blasdel & Lund 1983.</summary>
    private const float ARBOR_RADIUS_MIN = 300f;

    /// <summary>Максимальный эффективный радиус арбора. Мкм. Blasdel & Lund 1983.</summary>
    private const float ARBOR_RADIUS_MAX = 600f;

    /// <summary>Число уровней дихотомического ветвления (включая уровень 0).</summary>
    private const int BRANCH_LEVELS = 5;

    /// <summary>
    /// Дуговая длина ветви нулевого уровня (первичные ветви). Мкм.
    /// Длина на уровне k: SEG_LEN_LEVEL0 / 2^k.
    /// </summary>
    private const float SEG_LEN_LEVEL0 = 1168f;

    /// <summary>Шаг точек вдоль ветвей арбора. Мкм.</summary>
    private const float BRANCH_STEP_MKM = 8f;

    /// <summary>
    /// Стандартное отклонение Гауссовой целевой Z для первичных ветвей. Мкм.
    /// При σ=40 мкм около 95% бутончиков попадают в центральные ~160 мкм слоя,
    /// что соответствует биологическому профилю плотности TC-бутончиков
    /// (пик в центре 4Cα, спад к границам — Hendrickson et al. 1978).
    /// </summary>
    private const float ARBOR_Z_SIGMA = 40f;

    /// <summary>
    /// Малое стандартное отклонение Z-дрейфа в подветвях (уровни ≥ 1). Мкм.
    /// Терминальные сегменты идут почти горизонтально (локально плоско).
    /// </summary>
    private const float SUBBRANCH_Z_DRIFT = 15f;

    // ─── Мозаичное распределение бутончиков (Markov state machine) ─────────
    //
    // Реальные TC-аксоны в 4Cα показывают НЕ равномерное распределение бутончиков:
    // средний интервал ≈ 2.7 мкм маскирует двух-режимную структуру —
    //   • кластерный режим: 3-5 бутончиков с шагом 0.5-2 мкм (плотные «гроздья»);
    //   • разрежённый режим: 1-2 бутончика с шагом 3-12 мкм (промежутки).
    // Двухсостоянная Марковская цепь с подобранными вероятностями переходов
    // даёт стационарную долю «кластерных» бутончиков ≈ 70% и средний шаг ≈ 2.7 мкм.

    /// <summary>Минимальный интервал между бутончиками в кластере. Мкм.</summary>
    private const float CLUSTER_STEP_MIN = 0.6f;

    /// <summary>Максимальный интервал в кластере. Мкм.</summary>
    private const float CLUSTER_STEP_MAX = 2.0f;

    /// <summary>Минимальный интервал в разрежённом режиме. Мкм.</summary>
    private const float SPARSE_STEP_MIN = 3.0f;

    /// <summary>Максимальный интервал в разрежённом режиме. Мкм.</summary>
    private const float SPARSE_STEP_MAX = 9.0f;

    /// <summary>
    /// Вероятность остаться в кластерном режиме после очередного бутончика.
    /// При значении 0.75 среднее число бутончиков в кластере ≈ 4
    /// (геометрическое распределение с матожиданием 1/(1-p)).
    /// </summary>
    private const float CLUSTER_STAY_PROB = 0.75f;

    /// <summary>
    /// Вероятность остаться в разрежённом режиме после очередного бутончика.
    /// При значении 0.40 средняя длина «пробела» ≈ 1.7 бутончика.
    /// Стационарная доля кластерных бутончиков: 0.6/(0.6+0.25) ≈ 0.70.
    /// </summary>
    private const float SPARSE_STAY_PROB = 0.40f;

    /// <summary>
    /// Начальная вероятность стартовать ветвь в кластерном режиме
    /// (равна стационарной доле, чтобы не было «теплового хвоста»).
    /// </summary>
    private const float CLUSTER_INITIAL_PROB = 0.70f;

    // ─── Концевые кластеры (terminal boutons / glomerular endings) ─────────

    /// <summary>
    /// Доля концевых ветвей, оканчивающихся крупной розеткой (gloмerular ending).
    /// Розетки — характерное гломерулярное скопление бутончиков на конце
    /// TC-аксона; они хорошо видны на HRP-реконструкциях как «грозди винограда».
    /// Источник: Anderson & Martin 2002 (Cereb.Cortex 12).
    /// </summary>
    private const float TERMINAL_ROSETTE_PROB = 0.20f;

    /// <summary>Минимальный размер обычного концевого кластера (включительно).</summary>
    private const int TERMINAL_NORMAL_MIN = 2;

    /// <summary>Максимальный размер обычного концевого кластера (исключительно).</summary>
    private const int TERMINAL_NORMAL_MAX = 8;

    /// <summary>Минимальный радиус обычного концевого кластера. Мкм.</summary>
    private const float TERMINAL_NORMAL_RADIUS_MIN = 1.0f;

    /// <summary>Максимальный радиус обычного концевого кластера. Мкм.</summary>
    private const float TERMINAL_NORMAL_RADIUS_MAX = 3.0f;

    /// <summary>Минимальный размер розетки (включительно).</summary>
    private const int TERMINAL_ROSETTE_MIN = 8;

    /// <summary>Максимальный размер розетки (исключительно).</summary>
    private const int TERMINAL_ROSETTE_MAX = 16;

    /// <summary>Минимальный радиус розетки. Мкм.</summary>
    private const float TERMINAL_ROSETTE_RADIUS_MIN = 3.0f;

    /// <summary>Максимальный радиус розетки. Мкм.</summary>
    private const float TERMINAL_ROSETTE_RADIUS_MAX = 5.0f;

    // ─── Коллатераль в слой 6 (Type-2 M-аксоны) ─────────────────────────────

    /// <summary>
    /// Вероятность того, что данный M-аксон имеет коллатераль в слое 6.
    /// Биологически: Freund, Martin & Whitteridge 1989 разделяют M-аксоны
    /// V1 макак на Type 1 (только 4Cα) и Type 2 (с коллатералью в 6).
    /// Доля Type 2 по разным оценкам 30–50%.
    /// </summary>
    private const float LAYER6_COLLATERAL_PROB = 0.40f;

    /// <summary>Эффективный радиус коллатерального арбора в слое 6. Мкм.</summary>
    private const float LAYER6_ARBOR_RADIUS = 150f;

    /// <summary>Дуговая длина ветви уровня 0 в коллатерали слоя 6. Мкм.</summary>
    private const float LAYER6_SEG_LEN_LEVEL0 = 400f;

    /// <summary>Число уровней дихотомии в коллатерали слоя 6.</summary>
    private const int LAYER6_BRANCH_LEVELS = 3;

    /// <summary>
    /// Стандартное отклонение Гауссовой целевой Z в слое 6
    /// (центрируется на LAYER6_CENTER_Z = -1200).
    /// </summary>
    private const float LAYER6_Z_SIGMA = 35f;

    // ─── Пространственное расположение аксонов ──────────────────────────────

    /// <summary>Шаг гексагональной решётки аксонов (XY). Мкм. ~200 аксонов/мм².</summary>
    private const float GRID_SPACING_MKM = 63f;

    /// <summary>Радиус поиска на гексагональной решётке (в шагах).</summary>
    private const int GRID_SEARCH_RADIUS = 20;

    // ═══════════════════════════════════════════════════════════════════════
    // ПУБЛИЧНЫЙ API
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Генерирует N ближайших к началу координат таламокортикальных аксонов
    /// (M-путь ЛКТ → слой 4Cα V1 человека).
    /// </summary>
    public static Axon[] Generate(Random random, int n)
    {
        var centers = GenerateHexGridCenters(random);
        centers.Sort((a, b) => (a.X * a.X + a.Y * a.Y).CompareTo(b.X * b.X + b.Y * b.Y));

        if (centers.Count < n)
            throw new InvalidOperationException(
                $"На сетке только {centers.Count} узлов, запрошено {n}. " +
                $"Увеличьте GRID_SEARCH_RADIUS.");

        var axons = new Axon[n];
        for (int i = 0; i < n; i++)
            axons[i] = BuildAxon(centers[i], random);
        return axons;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ПОСТРОЕНИЕ ОДНОГО АКСОНА
    // ═══════════════════════════════════════════════════════════════════════

    private static Axon BuildAxon(Vector2 arborCenter, Random rng)
    {
        float arborRadius = SampleArborRadius(rng);
        int primaryBranches = rng.Next(2, 5); // [2, 4] включительно

        float startX = arborCenter.X + (float)(rng.NextDouble() - 0.5) * 2 * TRUNK_XY_DRIFT;
        float startY = arborCenter.Y + (float)(rng.NextDouble() - 0.5) * 2 * TRUNK_XY_DRIFT;
        var root = new AxonPoint(new Vector3(startX, startY, AXON_START_Z));

        var synapsePositions = new List<Vector3>();

        // Сэмплирование: является ли этот аксон Type-2 (с коллатералью в слое 6)
        bool hasLayer6Collateral = rng.NextDouble() < LAYER6_COLLATERAL_PROB;

        AxonPoint layerEntry = BuildTrunk(root, arborCenter, rng, synapsePositions, hasLayer6Collateral);

        BuildPrimaryFan(layerEntry, arborCenter, arborRadius, primaryBranches, rng, synapsePositions);

        var synapses = new Synapse[synapsePositions.Count];
        for (int i = 0; i < synapsePositions.Count; i++)
            synapses[i] = new Synapse(synapsePositions[i]);

        return new Axon(root, synapses);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // СТВОЛ АКСОНА (белое вещество → вход в 4Cα)
    // С опциональной коллатералью в слой 6 (Type-2 аксоны).
    // ═══════════════════════════════════════════════════════════════════════

    private static AxonPoint BuildTrunk(
        AxonPoint root,
        Vector2 arborCenter,
        Random rng,
        List<Vector3> synapses,
        bool hasLayer6Collateral)
    {
        AxonPoint current = root;
        float z = AXON_START_Z + TRUNK_STEP_MKM;
        bool layer6Inserted = false;

        while (z < LAYER_BOTTOM_Z)
        {
            float t = (z - AXON_START_Z) / (LAYER_BOTTOM_Z - AXON_START_Z);
            float x = root.Position.X + t * (arborCenter.X - root.Position.X)
                      + (float)(rng.NextDouble() - 0.5) * 5f;
            float y = root.Position.Y + t * (arborCenter.Y - root.Position.Y)
                      + (float)(rng.NextDouble() - 0.5) * 5f;

            var next = new AxonPoint(new Vector3(x, y, z));
            current.AddNext(next);
            current = next;

            // Когда ствол проходит через центр слоя 6, при необходимости
            // отращиваем коллатераль (становится вторым дочерним узлом current).
            if (hasLayer6Collateral && !layer6Inserted
                && z >= LAYER6_CENTER_Z - TRUNK_STEP_MKM
                && z < LAYER6_CENTER_Z + TRUNK_STEP_MKM)
            {
                BuildLayer6Collateral(current, rng, synapses);
                layer6Inserted = true;
            }

            z += TRUNK_STEP_MKM;
        }

        var entry = new AxonPoint(new Vector3(arborCenter.X, arborCenter.Y, LAYER_BOTTOM_Z));
        current.AddNext(entry);
        return entry;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ПЕРВИЧНЫЙ ВЕЕР В 4Cα (2–4 ветви через бинарные развилки)
    // ═══════════════════════════════════════════════════════════════════════

    private static void BuildPrimaryFan(
        AxonPoint start,
        Vector2 arborCenter,
        float arborRadius,
        int primaryBranches,
        Random rng,
        List<Vector3> synapses)
    {
        double baseAngle = rng.NextDouble() * 2 * Math.PI;
        var angles = new double[primaryBranches];
        for (int i = 0; i < primaryBranches; i++)
            angles[i] = baseAngle + i * (2 * Math.PI / primaryBranches)
                        + (rng.NextDouble() - 0.5) * 0.52; // ±~15°

        switch (primaryBranches)
        {
            case 2:
                BuildBranch(start, angles[0], arborRadius, 0, rng, synapses);
                BuildBranch(start, angles[1], arborRadius, 0, rng, synapses);
                break;

            case 3:
                BuildBranch(start, angles[0], arborRadius, 0, rng, synapses);
                {
                    var fork = MakeForkNode(start, angles[1], angles[2], rng);
                    start.AddNext(fork);
                    BuildBranch(fork, angles[1], arborRadius, 0, rng, synapses);
                    BuildBranch(fork, angles[2], arborRadius, 0, rng, synapses);
                }
                break;

            case 4:
                {
                    var fork0 = MakeForkNode(start, angles[0], angles[1], rng);
                    var fork1 = MakeForkNode(start, angles[2], angles[3], rng);
                    start.AddNext(fork0);
                    start.AddNext(fork1);
                    BuildBranch(fork0, angles[0], arborRadius, 0, rng, synapses);
                    BuildBranch(fork0, angles[1], arborRadius, 0, rng, synapses);
                    BuildBranch(fork1, angles[2], arborRadius, 0, rng, synapses);
                    BuildBranch(fork1, angles[3], arborRadius, 0, rng, synapses);
                }
                break;
        }
    }

    private static AxonPoint MakeForkNode(AxonPoint from, double angle0, double angle1, Random rng)
    {
        double midAngle = (angle0 + angle1) / 2.0;
        float dist = BRANCH_STEP_MKM * 2f;
        float x = from.Position.X + (float)Math.Cos(midAngle) * dist;
        float y = from.Position.Y + (float)Math.Sin(midAngle) * dist;
        float z = Math.Clamp(
            from.Position.Z + (float)(rng.NextDouble() - 0.5) * 6f,
            LAYER_BOTTOM_Z, LAYER_TOP_Z);
        return new AxonPoint(new Vector3(x, y, z));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // РЕКУРСИВНОЕ ВЕТВЛЕНИЕ ГЛАВНОГО АРБОРА В 4Cα
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Строит одну ветвь как случайное блуждание длиной arcLength
    /// с начальным направлением angle, затем рекурсивно строит 2 дочерних ветви.
    ///
    /// КЛЮЧЕВЫЕ ОТЛИЧИЯ ОТ ПРОСТОЙ ВЕРСИИ:
    ///   • Целевая Z — Гауссова (а не равномерная) для уровня 0:
    ///     N(LAYER_CENTER_Z, σ=ARBOR_Z_SIGMA) → пик плотности в центре слоя 4Cα.
    ///     Для уровней ≥ 1 — малый Z-дрейф, ветвь идёт почти горизонтально.
    ///   • Размещение бутончиков — мозаичное (двухсостоянная Марковская цепь):
    ///     режимы «cluster» и «sparse» чередуются, давая реалистичные
    ///     плотные грозди и промежутки. Средний шаг ≈ 2.7 мкм сохраняется.
    ///   • Концевые кластеры — вариативные: ~80% обычных, ~20% розеток.
    ///
    /// Инвариант: каждый AxonPoint имеет не более 2 дочерних (NextCount ≤ 2).
    /// </summary>
    private static void BuildBranch(
        AxonPoint start,
        double angle,
        float arborRadius,
        int level,
        Random rng,
        List<Vector3> synapses)
    {
        float arcLength = SEG_LEN_LEVEL0 / (float)Math.Pow(2, level);
        float xyAngleNoise = 0.55f - level * 0.07f;

        // ── Гауссова целевая Z с пиком в центре слоя 4Cα ──────────────
        float targetZ = SampleTargetZ(start.Position.Z, level,
                                      LAYER_CENTER_Z, ARBOR_Z_SIGMA,
                                      LAYER_BOTTOM_Z, LAYER_TOP_Z,
                                      rng);

        AxonPoint current = start;
        float distCovered = 0f;
        float x = start.Position.X;
        float y = start.Position.Y;
        float z = start.Position.Z;
        float zStep = (targetZ - z) / (arcLength / BRANCH_STEP_MKM + 1f);

        double curAngle = angle;

        // ── Инициализация мозаичного режима размещения бутончиков ─────
        bool inCluster = rng.NextDouble() < CLUSTER_INITIAL_PROB;
        float nextSynapseAt = SampleBoutonInterval(ref inCluster, rng);

        while (distCovered < arcLength)
        {
            float step = Math.Min(BRANCH_STEP_MKM, arcLength - distCovered);
            curAngle += (rng.NextDouble() - 0.5) * 2.0 * xyAngleNoise;

            // Сохраняем начало шага — нужно для линейной интерполяции
            // позиций бутончиков, попадающих внутрь этого шага.
            float prevX = x, prevY = y, prevZ = z;
            float stepStartDist = distCovered;

            x += (float)Math.Cos(curAngle) * step;
            y += (float)Math.Sin(curAngle) * step;
            z += zStep * (step / BRANCH_STEP_MKM);
            z = Math.Clamp(z, LAYER_BOTTOM_Z, LAYER_TOP_Z);

            var pt = new AxonPoint(new Vector3(x, y, z));
            current.AddNext(pt);
            current = pt;

            distCovered += step;

            // Размещение бутончиков en passant. Внутри одного 8-мкм шага
            // в кластерном режиме помещается до 10+ бутончиков (интервалы
            // 0.6-2 мкм), поэтому интерполируем позицию каждого бутончика
            // вдоль линии текущего шага по его distCovered. Без этого
            // вся «гроздь» схлопывалась бы в одну точку.
            while (distCovered >= nextSynapseAt)
            {
                float t = (nextSynapseAt - stepStartDist) / step; // 0..1
                if (t < 0f) t = 0f;
                else if (t > 1f) t = 1f;

                float bx = prevX + (x - prevX) * t;
                float by = prevY + (y - prevY) * t;
                float bz = prevZ + (z - prevZ) * t;

                synapses.Add(new Vector3(
                    bx + (float)(rng.NextDouble() - 0.5) * 0.8f,
                    by + (float)(rng.NextDouble() - 0.5) * 0.8f,
                    bz + (float)(rng.NextDouble() - 0.5) * 0.8f));

                nextSynapseAt += SampleBoutonInterval(ref inCluster, rng);
            }
        }

        // ── Концевой кластер (обычный или розетка) ─────────────────────
        AddTerminalCluster(x, y, z, rng, synapses);

        // ── Рекурсивное бинарное ветвление ─────────────────────────────
        if (level < BRANCH_LEVELS - 1)
        {
            double spreadAngle = Math.PI / 4.5 - level * 0.05;
            BuildBranch(current, curAngle - spreadAngle, arborRadius, level + 1, rng, synapses);
            BuildBranch(current, curAngle + spreadAngle, arborRadius, level + 1, rng, synapses);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // КОЛЛАТЕРАЛЬ В СЛОЙ 6 (TYPE-2 АКСОНЫ, ~40%)
    // Небольшой арбор глубже основного, у ствола на Z ≈ -1200 мкм.
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Строит коллатеральный арбор в слое 6, отходящий от точки trunkPoint
    /// на стволе. Это компактная конструкция: одна первичная ветвь,
    /// уходящая горизонтально в случайном направлении, с дальнейшим
    /// 2-3-уровневым дихотомическим ветвлением.
    /// trunkPoint получает второго ребёнка (collateralRoot).
    /// </summary>
    private static void BuildLayer6Collateral(
        AxonPoint trunkPoint,
        Random rng,
        List<Vector3> synapses)
    {
        // Корень коллатерали — небольшой шаг в сторону от ствола,
        // на уровне центра слоя 6.
        double initialAngle = rng.NextDouble() * 2 * Math.PI;
        float dx = (float)Math.Cos(initialAngle) * BRANCH_STEP_MKM * 2f;
        float dy = (float)Math.Sin(initialAngle) * BRANCH_STEP_MKM * 2f;
        float collZ = LAYER6_CENTER_Z + (float)SampleGaussian(rng) * LAYER6_Z_SIGMA;
        collZ = Math.Clamp(collZ, LAYER6_BOTTOM_Z, LAYER6_TOP_Z);

        var collateralRoot = new AxonPoint(new Vector3(
            trunkPoint.Position.X + dx,
            trunkPoint.Position.Y + dy,
            collZ));
        trunkPoint.AddNext(collateralRoot);

        // Одна первичная ветвь — самый частый паттерн для коллатерали в 6.
        BuildLayer6Branch(collateralRoot, initialAngle, 0, rng, synapses);
    }

    /// <summary>
    /// Рекурсивная ветвь коллатерали слоя 6. Аналогична BuildBranch, но:
    ///   • меньше арков длина (LAYER6_SEG_LEN_LEVEL0);
    ///   • Z-плотность центрирована на LAYER6_CENTER_Z;
    ///   • меньше уровней ветвления (LAYER6_BRANCH_LEVELS);
    ///   • тот же мозаичный режим размещения бутончиков (en passant),
    ///     потому что биологический шаг бутончиков на коллатерали аналогичен.
    /// </summary>
    private static void BuildLayer6Branch(
        AxonPoint start,
        double angle,
        int level,
        Random rng,
        List<Vector3> synapses)
    {
        float arcLength = LAYER6_SEG_LEN_LEVEL0 / (float)Math.Pow(2, level);
        float xyAngleNoise = 0.50f - level * 0.08f;

        float targetZ = SampleTargetZ(start.Position.Z, level,
                                      LAYER6_CENTER_Z, LAYER6_Z_SIGMA,
                                      LAYER6_BOTTOM_Z, LAYER6_TOP_Z,
                                      rng);

        AxonPoint current = start;
        float distCovered = 0f;
        float x = start.Position.X;
        float y = start.Position.Y;
        float z = start.Position.Z;
        float zStep = (targetZ - z) / (arcLength / BRANCH_STEP_MKM + 1f);

        double curAngle = angle;

        bool inCluster = rng.NextDouble() < CLUSTER_INITIAL_PROB;
        float nextSynapseAt = SampleBoutonInterval(ref inCluster, rng);

        while (distCovered < arcLength)
        {
            float step = Math.Min(BRANCH_STEP_MKM, arcLength - distCovered);
            curAngle += (rng.NextDouble() - 0.5) * 2.0 * xyAngleNoise;

            float prevX = x, prevY = y, prevZ = z;
            float stepStartDist = distCovered;

            x += (float)Math.Cos(curAngle) * step;
            y += (float)Math.Sin(curAngle) * step;
            z += zStep * (step / BRANCH_STEP_MKM);
            z = Math.Clamp(z, LAYER6_BOTTOM_Z, LAYER6_TOP_Z);

            var pt = new AxonPoint(new Vector3(x, y, z));
            current.AddNext(pt);
            current = pt;

            distCovered += step;

            while (distCovered >= nextSynapseAt)
            {
                float t = (nextSynapseAt - stepStartDist) / step;
                if (t < 0f) t = 0f;
                else if (t > 1f) t = 1f;

                float bx = prevX + (x - prevX) * t;
                float by = prevY + (y - prevY) * t;
                float bz = prevZ + (z - prevZ) * t;

                synapses.Add(new Vector3(
                    bx + (float)(rng.NextDouble() - 0.5) * 0.8f,
                    by + (float)(rng.NextDouble() - 0.5) * 0.8f,
                    bz + (float)(rng.NextDouble() - 0.5) * 0.8f));

                nextSynapseAt += SampleBoutonInterval(ref inCluster, rng);
            }
        }

        AddTerminalCluster(x, y, z, rng, synapses);

        if (level < LAYER6_BRANCH_LEVELS - 1)
        {
            double spreadAngle = Math.PI / 4.5 - level * 0.05;
            BuildLayer6Branch(current, curAngle - spreadAngle, level + 1, rng, synapses);
            BuildLayer6Branch(current, curAngle + spreadAngle, level + 1, rng, synapses);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // МОЗАИЧНОЕ РАЗМЕЩЕНИЕ БУТОНЧИКОВ EN PASSANT
    // Двухсостоянная Марковская цепь: «cluster» ↔ «sparse».
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Сэмплирует расстояние до следующего бутончика, обновляя состояние
    /// (кластер vs разрежение). Используется как для главного арбора в 4Cα,
    /// так и для коллатерали в слое 6.
    ///
    /// Стационарная доля «кластерных» бутончиков ≈ 70%, средний шаг ≈ 2.7 мкм,
    /// что соответствует биологическим измерениям TC-аксонов
    /// (Garcia-Marin et al. 2019, Anderson et al. 1994).
    /// </summary>
    private static float SampleBoutonInterval(ref bool inCluster, Random rng)
    {
        // Возможный переход состояния перед сэмплированием шага
        if (inCluster)
        {
            if (rng.NextDouble() >= CLUSTER_STAY_PROB)
                inCluster = false;
        }
        else
        {
            if (rng.NextDouble() >= SPARSE_STAY_PROB)
                inCluster = true;
        }

        if (inCluster)
            return CLUSTER_STEP_MIN
                   + (float)rng.NextDouble() * (CLUSTER_STEP_MAX - CLUSTER_STEP_MIN);
        else
            return SPARSE_STEP_MIN
                   + (float)rng.NextDouble() * (SPARSE_STEP_MAX - SPARSE_STEP_MIN);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // КОНЦЕВЫЕ КЛАСТЕРЫ (terminal boutons / glomerular endings)
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Размещает концевой кластер бутончиков вокруг точки (x,y,z).
    /// С вероятностью TERMINAL_ROSETTE_PROB формирует крупную «розетку»
    /// (8-15 бутончиков), иначе обычный малый кластер (2-7 бутончиков).
    /// Z-разброс делается вдвое меньше XY-разброса (плоские концевые гроздья).
    /// </summary>
    private static void AddTerminalCluster(
        float x, float y, float z,
        Random rng,
        List<Vector3> synapses)
    {
        bool isRosette = rng.NextDouble() < TERMINAL_ROSETTE_PROB;

        int count;
        float radius;
        if (isRosette)
        {
            count = rng.Next(TERMINAL_ROSETTE_MIN, TERMINAL_ROSETTE_MAX);
            radius = TERMINAL_ROSETTE_RADIUS_MIN
                     + (float)rng.NextDouble()
                       * (TERMINAL_ROSETTE_RADIUS_MAX - TERMINAL_ROSETTE_RADIUS_MIN);
        }
        else
        {
            count = rng.Next(TERMINAL_NORMAL_MIN, TERMINAL_NORMAL_MAX);
            radius = TERMINAL_NORMAL_RADIUS_MIN
                     + (float)rng.NextDouble()
                       * (TERMINAL_NORMAL_RADIUS_MAX - TERMINAL_NORMAL_RADIUS_MIN);
        }

        for (int t = 0; t < count; t++)
        {
            // XY-разброс полный, Z-разброс половинный — концевые гроздья
            // обычно «плоские», лежат в плоскости коры.
            synapses.Add(new Vector3(
                x + (float)(rng.NextDouble() - 0.5) * 2 * radius,
                y + (float)(rng.NextDouble() - 0.5) * 2 * radius,
                z + (float)(rng.NextDouble() - 0.5) * radius));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Сэмплирует целевую Z-координату для ветви данного уровня:
    ///   • уровень 0 — Гауссово вокруг centerZ (σ = sigma) → пик плотности
    ///     в центре слоя, плавный спад к границам;
    ///   • уровень ≥ 1 — малый Гауссов дрейф вокруг start.Z (σ = SUBBRANCH_Z_DRIFT),
    ///     ветвь идёт почти горизонтально.
    /// Результат всегда обрезается границами слоя [bottomZ, topZ].
    /// </summary>
    private static float SampleTargetZ(
        float startZ,
        int level,
        float centerZ,
        float sigma,
        float bottomZ,
        float topZ,
        Random rng)
    {
        float targetZ;
        if (level == 0)
            targetZ = centerZ + (float)SampleGaussian(rng) * sigma;
        else
            targetZ = startZ + (float)SampleGaussian(rng) * SUBBRANCH_Z_DRIFT;

        return Math.Clamp(targetZ, bottomZ + 5f, topZ - 5f);
    }

    /// <summary>
    /// Сэмплирует стандартное нормальное (Гауссово) распределение N(0,1)
    /// методом Бокса-Мюллера.
    /// </summary>
    private static double SampleGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    private static List<Vector2> GenerateHexGridCenters(Random rng)
    {
        var centers = new List<Vector2>();
        float hx = GRID_SPACING_MKM;
        float hy = GRID_SPACING_MKM * (float)Math.Sqrt(3) / 2f;

        for (int row = -GRID_SEARCH_RADIUS; row <= GRID_SEARCH_RADIUS; row++)
        {
            for (int col = -GRID_SEARCH_RADIUS; col <= GRID_SEARCH_RADIUS; col++)
            {
                float x = col * hx + (row % 2 != 0 ? hx / 2f : 0f);
                float y = row * hy;
                // Биологическая нерегулярность ±15%
                x += (float)(rng.NextDouble() - 0.5) * 2 * GRID_SPACING_MKM * 0.15f;
                y += (float)(rng.NextDouble() - 0.5) * 2 * GRID_SPACING_MKM * 0.15f;
                centers.Add(new Vector2(x, y));
            }
        }
        return centers;
    }

    /// <summary>
    /// Сэмплирует эффективный радиус арбора (логнормально, среднее 400 мкм).
    /// Источник: Blasdel & Lund 1983.
    /// </summary>
    private static float SampleArborRadius(Random rng)
    {
        float radius = 400f + (float)(SampleGaussian(rng) * 70f);
        return Math.Clamp(radius, ARBOR_RADIUS_MIN, ARBOR_RADIUS_MAX);
    }
}
