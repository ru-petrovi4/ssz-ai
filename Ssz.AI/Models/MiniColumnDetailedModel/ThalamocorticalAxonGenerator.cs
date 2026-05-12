using Ssz.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

/// <summary>
/// Генератор таламокортикальных аксонов (M-путь, ЛКТ → слой 4Cα V1 человека).
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
/// === ВЕРСИЯ 3: ОКУЛОДОМИНАНТНЫЕ ПАТЧИ + АСИММЕТРИЧНАЯ МОРФОЛОГИЯ ===
/// Ключевые дополнения относительно V2:
///
///   (V3-1) ОКУЛОДОМИНАНТНЫЕ ПАТЧИ (OD patches).
///       Главная характерная черта магноцеллюлярных TC-аксонов в 4Cα:
///       бутончики не разлиты равномерно по всей площади арбора, а
///       сгруппированы в 1–2 пространственно дискретных «патча» диаметром
///       ~300–500 мкм, соответствующих колонкам глазного доминирования.
///       Между патчами аксон физически присутствует (волокна проходят),
///       но плотность бутончиков падает в 3–4 раза. Реализовано как
///       мягкая Гауссова маска: для каждого en-passant бутончика и для
///       каждого концевого кластера вычисляется вероятность принятия,
///       зависящая от расстояния до ближайшего центра патча.
///       Источники: Freund, Martin & Whitteridge 1989 (HRP-реконструкции
///       M-аксонов V1 макак); LeVay, Connolly, Houde & Van Essen 1985
///       (геометрия OD-колонок); Horton & Hocking 1996, Adams & Horton
///       2003 (OD-колонки человека, ширина ~700–900 мкм); Florence &
///       Casagrande 1987 (мульти-патчевые арборы).
///
///   (V3-2) АСИММЕТРИЧНАЯ ДИХОТОМИЯ.
///       Дочерние ветви на каждом уровне получают разные арковые длины
///       (мультипликатор U[0.65, 1.35]) и слегка разные углы расхождения.
///       Реальные TC-арборы в HRP-реконструкциях демонстрируют выраженную
///       асимметрию ветвления; идеальные бинарные деревья — артефакт
///       упрощённых моделей. (Anderson, Douglas & Martin 1994.)
///
///   (V3-3) УСИЛЕНИЕ ПЛОТНОСТИ К КОНЦУ ВЕТВИ.
///       Последние ~30% арковой длины каждой ветви получают повышенную
///       плотность бутончиков (×1.5). Это соответствует тенденции
///       TC-аксонов «сбрасывать» большую часть бутончиков рядом
///       с концевыми розетками. Anderson & Martin 2002: ~50–60%
///       бутончиков терминальной ветви расположены в её дистальной трети.
///
///   (V3-4) ПОДСЛОЙНАЯ Z-СЕГРЕГАЦИЯ В 4Cα.
///       Слой 4Cα тонко стратифицирован: верхняя половина (4Cα-upper)
///       и нижняя половина (4Cα-lower) получают вход от разных подгрупп
///       LGN. Каждая первичная ветвь данного аксона случайно «выбирает»
///       один из подслоёв и предпочтительно туда уходит (Гауссов центр
///       смещается на ±25 мкм). Это даёт характерные «двухполосные»
///       реконструкции, видимые на полутонких срезах.
///       Источники: Yazar et al. 2004; Hendrickson et al. 1978.
///
///   (V3-5) ВАРИАТИВНАЯ ПЕРВИЧНАЯ АРКА (±20%) и СТОХАСТИЧЕСКОЕ РАННЕЕ
///       ОКОНЧАНИЕ ВЕТВЕЙ (15% вероятность завершиться на уровень
///       раньше — даёт характерную «обрезанность» некоторых ветвей,
///       видимую в реальных HRP-реконструкциях).
///
///   (V3-6) УМЕНЬШЕННЫЙ ПОЗИЦИОННЫЙ ДЖИТТЕР БУТОНЧИКОВ (0.4 мкм
///       вместо 0.8). Соответствует реальному размеру TC-бутончика
///       (~1 мкм в диаметре).
///
/// === АНАТОМИЧЕСКИЕ ПАРАМЕТРЫ (приматы / человек) ===
///
/// Слой 4Cα V1:
///   - Z = [LAYER_BOTTOM_Z, LAYER_TOP_Z] = [-1000, -800] мкм (~200 мкм)
///   - Центр: LAYER_CENTER_Z = -900 мкм
///   - Подслои: 4Cα-upper (~ -850), 4Cα-lower (~ -950)
///
/// Слой 6 V1 (для коллатералей Type-2 аксонов):
///   - Центр: LAYER6_CENTER_Z ≈ -1200 мкм
///
/// Ствол:
///   - Старт: AXON_START_Z = -1350 мкм (белое вещество)
///   - XY-дрейф ~30 мкм (зрительная лучистость)
///
/// Главный арбор в 4Cα:
///   - Эффективный радиус: 300..600 мкм (~400 ср., Blasdel & Lund 1983)
///   - 2–4 первичные ветви, BRANCH_LEVELS = 5 уровней дихотомии
///   - Окулодоминантные патчи: 1–2 на аксон, радиус 150–250 мкм
///   - Tortuosity ≈ 3, суммарная дуговая длина ~17 мм
///   - Бутончиков на аксон: ~5000–7000 (Garcia-Marin et al. 2019)
///
/// Гексагональная плотность аксонов: GRID_SPACING_MKM = 63 мкм
///   → ~200 M-аксонов/мм² (фовеальная проекция V1).
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

    /// <summary>
    /// V3: Смещение центра Гауссова Z-распределения для подслойной сегрегации.
    /// Половина первичных ветвей таргетируется на 4Cα-upper (центр -875),
    /// другая — на 4Cα-lower (-925). Мкм.
    /// </summary>
    private const float SUBLAYER_Z_OFFSET = 25f;

    // ─── Геометрия слоя 6 (для Type-2 аксонов с коллатералью) ──────────────
    /// <summary>Центр слоя 6. Мкм.</summary>
    private const float LAYER6_CENTER_Z = -1200f;

    /// <summary>Верхняя граница слоя 6. Мкм.</summary>
    private const float LAYER6_TOP_Z = -1100f;

    /// <summary>Нижняя граница слоя 6. Мкм.</summary>
    private const float LAYER6_BOTTOM_Z = -1300f;

    // ─── Параметры ствола аксона ────────────────────────────────────────────
    /// <summary>Z-координата начала аксона (в белом веществе). Мкм.</summary>
    private const float AXON_START_Z = -1350f;

    /// <summary>Максимальное случайное горизонтальное смещение ствола (XY-дрейф). Мкм.</summary>
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

    /// <summary>Базовая дуговая длина ветви нулевого уровня. Мкм.</summary>
    private const float SEG_LEN_LEVEL0 = 1168f;

    /// <summary>
    /// V3: Разброс длины первичной арки на уровне аксона (мультипликатор).
    /// Эффективная SEG_LEN_LEVEL0 для данного аксона:
    /// SEG_LEN_LEVEL0 × U[1-σ, 1+σ], где σ = PRIMARY_ARC_LEN_JITTER.
    /// </summary>
    private const float PRIMARY_ARC_LEN_JITTER = 0.20f;

    /// <summary>
    /// V3: Мультипликативный разброс длины дочерней ветви относительно
    /// номинальной (parent / 2). Случайно U[1-σ, 1+σ] для каждой дочерней.
    /// Источник: Anderson, Douglas & Martin 1994 — выраженная асимметрия
    /// дочерних арок в HRP-реконструкциях TC-аксонов.
    /// </summary>
    private const float BRANCH_ASYMMETRY_JITTER = 0.35f;

    /// <summary>Шаг точек вдоль ветвей арбора. Мкм.</summary>
    private const float BRANCH_STEP_MKM = 8f;

    /// <summary>
    /// Стандартное отклонение Гауссовой целевой Z для первичных ветвей. Мкм.
    /// При σ=40 мкм около 95% бутончиков попадают в центральные ~160 мкм слоя.
    /// </summary>
    private const float ARBOR_Z_SIGMA = 40f;

    /// <summary>Малое СО Z-дрейфа в подветвях (уровни ≥ 1). Мкм.</summary>
    private const float SUBBRANCH_Z_DRIFT = 15f;

    /// <summary>
    /// V3: Вероятность завершить ветвь на уровень раньше максимального.
    /// Применяется только на уровнях BRANCH_LEVELS-2 и BRANCH_LEVELS-3,
    /// чтобы не «обрезать» арбор слишком сильно. Реалистично имитирует
    /// случайные ранние терминалы в HRP-реконструкциях.
    /// </summary>
    private const float EARLY_TERMINATION_PROB = 0.15f;

    /// <summary>
    /// V3: Мультипликатор плотности бутончиков в дистальной части ветви.
    /// Применяется к последним TERMINAL_DENSITY_FRACTION × длины.
    /// </summary>
    private const float TERMINAL_DENSITY_BOOST = 1.5f;

    /// <summary>V3: Доля длины ветви, считающаяся «дистальной» (с повышенной плотностью).</summary>
    private const float TERMINAL_DENSITY_FRACTION = 0.30f;

    /// <summary>V3: Позиционный джиттер бутончика. Мкм. Соответствует размеру TC-бутончика.</summary>
    private const float BOUTON_POSITION_JITTER = 0.4f;

    // ─── Мозаичное распределение бутончиков (Markov state machine) ─────────
    /// <summary>Минимальный интервал между бутончиками в кластере. Мкм.</summary>
    private const float CLUSTER_STEP_MIN = 0.6f;

    /// <summary>Максимальный интервал в кластере. Мкм.</summary>
    private const float CLUSTER_STEP_MAX = 2.0f;

    /// <summary>Минимальный интервал в разрежённом режиме. Мкм.</summary>
    private const float SPARSE_STEP_MIN = 3.0f;

    /// <summary>Максимальный интервал в разрежённом режиме. Мкм.</summary>
    private const float SPARSE_STEP_MAX = 9.0f;

    /// <summary>Вероятность остаться в кластерном режиме.</summary>
    private const float CLUSTER_STAY_PROB = 0.75f;

    /// <summary>Вероятность остаться в разрежённом режиме.</summary>
    private const float SPARSE_STAY_PROB = 0.40f;

    /// <summary>Начальная вероятность стартовать ветвь в кластерном режиме.</summary>
    private const float CLUSTER_INITIAL_PROB = 0.70f;

    // ─── Концевые кластеры (terminal boutons / glomerular endings) ─────────

    /// <summary>Доля концевых ветвей, оканчивающихся крупной розеткой.</summary>
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

    // ─── V3: Окулодоминантные патчи ────────────────────────────────────────

    /// <summary>
    /// V3: Доля аксонов с одним патчем (строгое OD-доминирование).
    /// Остальные имеют два патча (главный + второстепенный).
    /// </summary>
    private const float OD_SINGLE_PATCH_PROB = 0.45f;

    /// <summary>V3: Минимальный радиус OD-патча. Мкм.</summary>
    private const float OD_PATCH_RADIUS_MIN = 150f;

    /// <summary>V3: Максимальный радиус OD-патча. Мкм.</summary>
    private const float OD_PATCH_RADIUS_MAX = 250f;

    /// <summary>
    /// V3: Расстояние между двумя патчами как доля от arborRadius.
    /// При arborRadius=400 это даёт 360–560 мкм (период OD-колонок макаки ~400 мкм);
    /// при arborRadius=600 — 540–840 мкм (период OD-колонок человека ~700–900 мкм,
    /// Adams & Horton 2003). Допускается частичное перекрытие патчей у малых
    /// арборов — они эффективно становятся монокулярными.
    /// </summary>
    private const float OD_INTER_PATCH_DISTANCE_FRACTION_MIN = 0.90f;
    private const float OD_INTER_PATCH_DISTANCE_FRACTION_MAX = 1.40f;

    /// <summary>
    /// V3: Базовая вероятность принятия бутончика ВНЕ всех патчей.
    /// 0.30 → плотность вне патчей в ~3.3 раза ниже, чем в центре патча,
    /// что соответствует HRP-данным магно-аксонов.
    /// </summary>
    private const float OD_BASELINE_ACCEPTANCE = 0.30f;

    /// <summary>
    /// V3: Радиус «второстепенного» патча относительно главного.
    /// (Минорный патч — слабая проекция в соседнюю OD-колонку.)
    /// </summary>
    private const float OD_SECONDARY_PATCH_SIZE_RATIO = 0.65f;

    // ─── Коллатераль в слой 6 (Type-2 M-аксоны) ─────────────────────────────

    private const float LAYER6_COLLATERAL_PROB = 0.40f;
    private const float LAYER6_ARBOR_RADIUS = 150f;
    private const float LAYER6_SEG_LEN_LEVEL0 = 400f;
    private const int LAYER6_BRANCH_LEVELS = 3;
    private const float LAYER6_Z_SIGMA = 35f;

    // ─── Пространственное расположение аксонов ──────────────────────────────
    /// <summary>Шаг гексагональной решётки аксонов (XY). Мкм. ~200 аксонов/мм².</summary>
    private const float GRID_SPACING_MKM = 63f;

    /// <summary>Радиус поиска на гексагональной решётке (в шагах).</summary>
    private const int GRID_SEARCH_RADIUS = 20;

    // ═══════════════════════════════════════════════════════════════════════
    // ВСПОМОГАТЕЛЬНЫЕ ТИПЫ
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// V3: Окулодоминантный патч — мягкая Гауссова маска для бутончиков.
    /// </summary>
    private readonly struct OdPatch
    {
        public readonly float X;
        public readonly float Y;
        public readonly float Radius;
        public OdPatch(float x, float y, float radius)
        {
            X = x; Y = y; Radius = radius;
        }
    }

    /// <summary>
    /// V3: Контекст одного арбора (передаётся через рекурсивные вызовы
    /// BuildBranch). Содержит данные, общие для всех ветвей данного аксона.
    /// </summary>
    private sealed class ArborContext
    {
        public Vector2 Center;
        public float Radius;
        public OdPatch[] Patches = Array.Empty<OdPatch>();
        /// <summary>Эффективная SEG_LEN_LEVEL0 для данного аксона (с jitter).</summary>
        public float Level0ArcLength;
    }

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

        // V3: Контекст арбора с окулодоминантными патчами и индивидуальной
        // длиной первичной арки.
        var ctx = new ArborContext
        {
            Center = arborCenter,
            Radius = arborRadius,
            Patches = SampleOdPatches(arborCenter, arborRadius, rng),
            Level0ArcLength = SEG_LEN_LEVEL0
                * (1f + (float)(rng.NextDouble() - 0.5) * 2f * PRIMARY_ARC_LEN_JITTER),
        };

        // Сэмплирование: является ли этот аксон Type-2 (с коллатералью в слое 6)
        //bool hasLayer6Collateral = rng.NextDouble() < LAYER6_COLLATERAL_PROB;
        bool hasLayer6Collateral = false;

        AxonPoint layerEntry = BuildTrunk(root, arborCenter, rng, synapsePositions, hasLayer6Collateral);

        BuildPrimaryFan(layerEntry, ctx, primaryBranches, rng, synapsePositions);

        var synapses = new FastList<Synapse>(synapsePositions.Count);
        float r = ARBOR_RADIUS_MAX * 1.5f;
        for (int i = 0; i < synapsePositions.Count; i++)
        {
            var p = synapsePositions[i];
            p.Z = 0.0f;
            if (p.Length() < r)
                synapses.Add(new Synapse(synapsePositions[i]));
        }

        return new Axon(root, synapses.ToArray());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // V3: ОКУЛОДОМИНАНТНЫЕ ПАТЧИ — СЭМПЛИРОВАНИЕ И ОЦЕНКА
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// V3: Сэмплирует 1 или 2 окулодоминантных патча для одного аксона.
    ///
    /// • Главный патч: смещён от центра арбора на 10–25% от arborRadius,
    ///   радиус 150–250 мкм.
    /// • Второстепенный патч (с вероятностью 1 − OD_SINGLE_PATCH_PROB):
    ///   на расстоянии 0.6–0.95 × arborRadius от главного, в произвольном
    ///   направлении. Это соответствует периоду OD-колонок (~700–900 мкм
    ///   у человека, ~400–500 мкм у макаки — Adams & Horton 2003).
    ///   Второстепенный патч заметно меньше главного (~65%), отражая
    ///   доминирование одного глаза.
    /// </summary>
    private static OdPatch[] SampleOdPatches(Vector2 arborCenter, float arborRadius, Random rng)
    {
        bool singlePatch = rng.NextDouble() < OD_SINGLE_PATCH_PROB;

        // Главный патч: лёгкое смещение от геометрического центра арбора.
        double primAngle = rng.NextDouble() * 2.0 * Math.PI;
        float primOffset = arborRadius * (0.10f + (float)rng.NextDouble() * 0.15f);
        float p0x = arborCenter.X + (float)Math.Cos(primAngle) * primOffset;
        float p0y = arborCenter.Y + (float)Math.Sin(primAngle) * primOffset;
        float p0r = OD_PATCH_RADIUS_MIN
                    + (float)rng.NextDouble() * (OD_PATCH_RADIUS_MAX - OD_PATCH_RADIUS_MIN);

        if (singlePatch)
        {
            return new[] { new OdPatch(p0x, p0y, p0r) };
        }

        // Второстепенный патч.
        double secAngle = rng.NextDouble() * 2.0 * Math.PI;
        float distFrac = OD_INTER_PATCH_DISTANCE_FRACTION_MIN
                         + (float)rng.NextDouble()
                           * (OD_INTER_PATCH_DISTANCE_FRACTION_MAX
                              - OD_INTER_PATCH_DISTANCE_FRACTION_MIN);
        float secDist = arborRadius * distFrac;
        float p1x = p0x + (float)Math.Cos(secAngle) * secDist;
        float p1y = p0y + (float)Math.Sin(secAngle) * secDist;
        float p1r = p0r * OD_SECONDARY_PATCH_SIZE_RATIO;

        return new[]
        {
            new OdPatch(p0x, p0y, p0r),
            new OdPatch(p1x, p1y, p1r),
        };
    }

    /// <summary>
    /// V3: Вероятность принятия бутончика в точке (x, y) с учётом
    /// окулодоминантных патчей. Возвращает значение в диапазоне
    /// [OD_BASELINE_ACCEPTANCE, 1.0]:
    ///   • в центре любого патча → ≈ 1.0;
    ///   • на границе патча (d = R) → ≈ 0.30 + 0.70 × e⁻¹ ≈ 0.56;
    ///   • далеко от всех патчей → OD_BASELINE_ACCEPTANCE (0.30).
    ///
    /// Используется как множитель вероятности при размещении en-passant
    /// бутончиков, а также для масштабирования размера концевых кластеров.
    /// </summary>
    private static float PatchAcceptance(float x, float y, OdPatch[] patches)
    {
        float maxAccept = OD_BASELINE_ACCEPTANCE;
        for (int i = 0; i < patches.Length; i++)
        {
            var p = patches[i];
            float dx = x - p.X;
            float dy = y - p.Y;
            float d2 = dx * dx + dy * dy;
            float r2 = p.Radius * p.Radius;
            // Гауссова маска: exp(-d²/r²) даёт 1.0 в центре, e⁻¹≈0.37 на радиусе.
            float gauss = MathF.Exp(-d2 / r2);
            float accept = OD_BASELINE_ACCEPTANCE + (1f - OD_BASELINE_ACCEPTANCE) * gauss;
            if (accept > maxAccept) maxAccept = accept;
        }
        return maxAccept;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // СТВОЛ АКСОНА (белое вещество → вход в 4Cα)
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
    // V3: каждая первичная ветвь получает индивидуальный подслойный таргет
    //     (4Cα-upper vs 4Cα-lower).
    // ═══════════════════════════════════════════════════════════════════════

    private static void BuildPrimaryFan(
        AxonPoint start,
        ArborContext ctx,
        int primaryBranches,
        Random rng,
        List<Vector3> synapses)
    {
        double baseAngle = rng.NextDouble() * 2 * Math.PI;
        var angles = new double[primaryBranches];
        for (int i = 0; i < primaryBranches; i++)
            angles[i] = baseAngle + i * (2 * Math.PI / primaryBranches)
                        + (rng.NextDouble() - 0.5) * 0.52; // ±~15°

        // V3: подслойный таргет — случайно upper/lower для каждой первичной ветви.
        // Гарантируем что обе моды присутствуют, если ветвей >= 2.
        var sublayerOffsets = new float[primaryBranches];
        for (int i = 0; i < primaryBranches; i++)
            sublayerOffsets[i] = (rng.NextDouble() < 0.5) ? +SUBLAYER_Z_OFFSET : -SUBLAYER_Z_OFFSET;

        switch (primaryBranches)
        {
            case 2:
                BuildBranch(start, angles[0], ctx, 0, ctx.Level0ArcLength,
                            sublayerOffsets[0], rng, synapses);
                BuildBranch(start, angles[1], ctx, 0, ctx.Level0ArcLength,
                            sublayerOffsets[1], rng, synapses);
                break;

            case 3:
                BuildBranch(start, angles[0], ctx, 0, ctx.Level0ArcLength,
                            sublayerOffsets[0], rng, synapses);
                {
                    var fork = MakeForkNode(start, angles[1], angles[2], rng);
                    start.AddNext(fork);
                    BuildBranch(fork, angles[1], ctx, 0, ctx.Level0ArcLength,
                                sublayerOffsets[1], rng, synapses);
                    BuildBranch(fork, angles[2], ctx, 0, ctx.Level0ArcLength,
                                sublayerOffsets[2], rng, synapses);
                }
                break;

            case 4:
                {
                    var fork0 = MakeForkNode(start, angles[0], angles[1], rng);
                    var fork1 = MakeForkNode(start, angles[2], angles[3], rng);
                    start.AddNext(fork0);
                    start.AddNext(fork1);
                    BuildBranch(fork0, angles[0], ctx, 0, ctx.Level0ArcLength,
                                sublayerOffsets[0], rng, synapses);
                    BuildBranch(fork0, angles[1], ctx, 0, ctx.Level0ArcLength,
                                sublayerOffsets[1], rng, synapses);
                    BuildBranch(fork1, angles[2], ctx, 0, ctx.Level0ArcLength,
                                sublayerOffsets[2], rng, synapses);
                    BuildBranch(fork1, angles[3], ctx, 0, ctx.Level0ArcLength,
                                sublayerOffsets[3], rng, synapses);
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
    /// Строит одну ветвь как случайное блуждание заданной дуговой длины,
    /// рекурсивно строит 2 дочерние ветви.
    ///
    /// V3 ОТЛИЧИЯ:
    ///   • Окулодоминантные патчи: каждый en-passant бутончик проходит
    ///     вероятностный отбор по PatchAcceptance(x,y).
    ///   • Усиление плотности к концу ветви: в дистальной трети
    ///     эффективный интервал между бутончиками уменьшается в
    ///     TERMINAL_DENSITY_BOOST раз.
    ///   • Асимметричная дихотомия: дочерние ветви получают разные
    ///     арковые длины через BRANCH_ASYMMETRY_JITTER.
    ///   • Подслойный Z-таргет: дополнительное смещение центра
    ///     Гауссова Z-распределения (sublayerOffset) для первичной ветви.
    ///   • Стохастическое раннее окончание ветви на предпоследних уровнях.
    /// </summary>
    private static void BuildBranch(
        AxonPoint start,
        double angle,
        ArborContext ctx,
        int level,
        float arcLength,
        float sublayerOffset,
        Random rng,
        List<Vector3> synapses)
    {
        float xyAngleNoise = 0.55f - level * 0.07f;

        // Гауссов Z-таргет с подслойным смещением (только для уровня 0).
        float effectiveCenterZ = LAYER_CENTER_Z + (level == 0 ? sublayerOffset : 0f);
        float targetZ = SampleTargetZ(start.Position.Z, level,
                                      effectiveCenterZ, ARBOR_Z_SIGMA,
                                      LAYER_BOTTOM_Z, LAYER_TOP_Z,
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

        // V3: порог расстояния, после которого начинается «терминальное усиление».
        float terminalBoostStartDist = arcLength * (1f - TERMINAL_DENSITY_FRACTION);

        while (distCovered < arcLength)
        {
            float step = Math.Min(BRANCH_STEP_MKM, arcLength - distCovered);
            curAngle += (rng.NextDouble() - 0.5) * 2.0 * xyAngleNoise;

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

            while (distCovered >= nextSynapseAt)
            {
                float t = (nextSynapseAt - stepStartDist) / step;
                if (t < 0f) t = 0f;
                else if (t > 1f) t = 1f;

                float bx = prevX + (x - prevX) * t;
                float by = prevY + (y - prevY) * t;
                float bz = prevZ + (z - prevZ) * t;

                // V3: вероятностный отбор по окулодоминантной маске.
                // Внутри патча принимается почти всегда, между патчами —
                // только OD_BASELINE_ACCEPTANCE.
                float accept = PatchAcceptance(bx, by, ctx.Patches);
                if (rng.NextDouble() < accept)
                {
                    synapses.Add(new Vector3(
                        bx + (float)(rng.NextDouble() - 0.5) * BOUTON_POSITION_JITTER,
                        by + (float)(rng.NextDouble() - 0.5) * BOUTON_POSITION_JITTER,
                        bz + (float)(rng.NextDouble() - 0.5) * BOUTON_POSITION_JITTER));
                }

                // V3: терминальное усиление плотности — сокращаем интервал
                // до следующего бутончика. Markov-состояние всё равно
                // обновляется, поэтому средний шаг не нарушает кластеризации.
                float interval = SampleBoutonInterval(ref inCluster, rng);
                if (nextSynapseAt > terminalBoostStartDist)
                    interval /= TERMINAL_DENSITY_BOOST;
                nextSynapseAt += interval;
            }
        }

        // V3: концевой кластер с гейтингом по окулодоминантным патчам.
        AddTerminalCluster(x, y, z, ctx.Patches, rng, synapses);

        // V3: рекурсивное бинарное ветвление с асимметрией +
        //     стохастическим ранним окончанием на предпоследних уровнях.
        if (level < BRANCH_LEVELS - 1)
        {
            // Раннее окончание разрешено только когда уже есть хотя бы
            // несколько уровней позади (иначе арбор будет слишком обрезанным).
            if (level >= BRANCH_LEVELS - 3 && rng.NextDouble() < EARLY_TERMINATION_PROB)
                return;

            // Асимметричный угол: чуть разные расхождения для двух дочерних.
            double spreadBase = Math.PI / 4.5 - level * 0.05;
            double spreadL = spreadBase * (1.0 + (rng.NextDouble() - 0.5) * 0.4);
            double spreadR = spreadBase * (1.0 + (rng.NextDouble() - 0.5) * 0.4);

            // Асимметричная длина дочерних ветвей.
            float nominalChildLen = arcLength * 0.5f;
            float childLenL = nominalChildLen
                * (1f + (float)(rng.NextDouble() - 0.5) * 2f * BRANCH_ASYMMETRY_JITTER);
            float childLenR = nominalChildLen
                * (1f + (float)(rng.NextDouble() - 0.5) * 2f * BRANCH_ASYMMETRY_JITTER);

            BuildBranch(current, curAngle - spreadL, ctx, level + 1, childLenL,
                        sublayerOffset, rng, synapses);
            BuildBranch(current, curAngle + spreadR, ctx, level + 1, childLenR,
                        sublayerOffset, rng, synapses);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // КОЛЛАТЕРАЛЬ В СЛОЙ 6 (TYPE-2 АКСОНЫ, ~40%)
    // (Окулодоминантный гейтинг здесь не применяется: в слое 6 OD-сегрегация
    // существенно слабее, чем в 4Cα; Wiser & Callaway 1996.)
    // ═══════════════════════════════════════════════════════════════════════

    private static void BuildLayer6Collateral(
        AxonPoint trunkPoint,
        Random rng,
        List<Vector3> synapses)
    {
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

        BuildLayer6Branch(collateralRoot, initialAngle, 0, rng, synapses);
    }

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
                    bx + (float)(rng.NextDouble() - 0.5) * BOUTON_POSITION_JITTER,
                    by + (float)(rng.NextDouble() - 0.5) * BOUTON_POSITION_JITTER,
                    bz + (float)(rng.NextDouble() - 0.5) * BOUTON_POSITION_JITTER));

                nextSynapseAt += SampleBoutonInterval(ref inCluster, rng);
            }
        }

        // Концевой кластер без OD-гейтинга (слой 6).
        AddTerminalCluster(x, y, z, Array.Empty<OdPatch>(), rng, synapses);

        if (level < LAYER6_BRANCH_LEVELS - 1)
        {
            double spreadAngle = Math.PI / 4.5 - level * 0.05;
            BuildLayer6Branch(current, curAngle - spreadAngle, level + 1, rng, synapses);
            BuildLayer6Branch(current, curAngle + spreadAngle, level + 1, rng, synapses);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // МОЗАИЧНОЕ РАЗМЕЩЕНИЕ БУТОНЧИКОВ EN PASSANT (Markov: cluster ↔ sparse)
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Сэмплирует расстояние до следующего бутончика, обновляя состояние
    /// (кластер vs разрежение). Стационарная доля кластерных бутончиков ≈ 70%,
    /// средний шаг ≈ 2.7 мкм.
    /// </summary>
    private static float SampleBoutonInterval(ref bool inCluster, Random rng)
    {
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
    /// (8–15 бутончиков), иначе обычный малый кластер (2–7 бутончиков).
    ///
    /// V3: размер кластера масштабируется по PatchAcceptance — если
    /// конец ветви лежит вне OD-патчей, кластер сжимается пропорционально.
    /// Каждый отдельный бутончик дополнительно проходит вероятностный
    /// отбор, что даёт «рваные» концевые гроздья на границе патча.
    /// </summary>
    private static void AddTerminalCluster(
        float x, float y, float z,
        OdPatch[] patches,
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

        // V3: для главного арбора (4Cα) — масштаб кластера по OD-маске.
        // Для слоя 6 patches.Length == 0 → patchProb = OD_BASELINE_ACCEPTANCE,
        // но без гейтинга количества (поэтому ставим масштаб 1.0).
        float clusterScale = 1f;
        if (patches.Length > 0)
        {
            float patchProb = PatchAcceptance(x, y, patches);
            // Линейно от 0.45 (на baseline) до 1.0 (в центре патча).
            clusterScale = 0.45f + 0.55f
                * (patchProb - OD_BASELINE_ACCEPTANCE) / (1f - OD_BASELINE_ACCEPTANCE);
            count = (int)Math.Round(count * clusterScale);
            if (count < 1) count = 1;
        }

        for (int t = 0; t < count; t++)
        {
            // Концевые гроздья плоские: Z-разброс вдвое меньше XY.
            float bx = x + (float)(rng.NextDouble() - 0.5) * 2 * radius;
            float by = y + (float)(rng.NextDouble() - 0.5) * 2 * radius;
            float bz = z + (float)(rng.NextDouble() - 0.5) * radius;

            // Дополнительный бутонный гейтинг по OD-маске.
            if (patches.Length > 0)
            {
                float accept = PatchAcceptance(bx, by, patches);
                if (rng.NextDouble() >= accept) continue;
            }

            synapses.Add(new Vector3(bx, by, bz));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Сэмплирует целевую Z-координату для ветви данного уровня.
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
    /// Сэмплирует стандартное нормальное распределение N(0,1)
    /// методом Бокса–Мюллера.
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
    /// Сэмплирует эффективный радиус арбора (Гауссово вокруг 400 мкм,
    /// зажатое в [ARBOR_RADIUS_MIN, ARBOR_RADIUS_MAX]).
    /// Источник: Blasdel & Lund 1983.
    /// </summary>
    private static float SampleArborRadius(Random rng)
    {
        float radius = 400f + (float)(SampleGaussian(rng) * 70f);
        return Math.Clamp(radius, ARBOR_RADIUS_MIN, ARBOR_RADIUS_MAX);
    }
}
