using Ssz.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

/// <summary>
/// Генератор таламокортикальных аксонов (M-путь, ЛКТ → слой 4Cα V1 человека).
///
/// === АНАТОМИЧЕСКИЕ ПАРАМЕТРЫ (приматы / человек) ===
///
/// Слой 4Cα V1:
/// - Z = [LAYER_BOTTOM_Z, LAYER_TOP_Z] = [-1000, -800] мкм (~200 мкм)
/// - Центр: LAYER_CENTER_Z = -900 мкм
/// - Подслои: 4Cα-upper (~ -850), 4Cα-lower (~ -950)
///
/// Слой 6 V1 (для коллатералей Type-2 аксонов):
/// - Центр: LAYER6_CENTER_Z ≈ -1200 мкм
///
/// Ствол:
/// - Старт: AXON_START_Z = -1350 мкм (белое вещество)
/// - XY-дрейф ~30 мкм (зрительная лучистость)
///
/// Главный арбор в 4Cα:
/// - Эффективный радиус: 300..600 мкм (~400 ср., Blasdel & Lund 1983)
/// - 2–4 первичные ветви, BRANCH_LEVELS = 5 уровней дихотомии
/// - Окулодоминантные патчи: 1–2 на аксон, радиус 175–325 мкм
/// - OD-межпатчевое расстояние: 700–950 мкм (ширина OD-полосы человека)
/// - Tortuosity ≈ 3, суммарная дуговая длина ~17 мм
/// - Бутончиков на аксон: ~5000–7000 (Garcia-Marin et al. 2019)
///
/// Гексагональная плотность аксонов: GRID_SPACING_MKM = 63 мкм
/// → ~200 M-аксонов/мм² (фовеальная проекция V1).
/// </summary>
public static class ThalamocorticalAxonGenerator
{
    // ─── Геометрия слоя 4Cα ────────────────────────────────────────────────
    /// Верхняя граница слоя 4Cα (ближе к поверхности). Мкм.
    private const float LAYER_TOP_Z = -800f;

    /// Нижняя граница слоя 4Cα (ближе к белому веществу). Мкм.
    private const float LAYER_BOTTOM_Z = -1000f;

    /// Центр слоя 4Cα — место пиковой плотности TC-бутончиков. Мкм.
    private const float LAYER_CENTER_Z = -900f;

    /// <summary>
    /// Смещение центра Гауссова Z-распределения для подслойной сегрегации.
    /// Половина первичных ветвей таргетируется на 4Cα-upper (центр -875),
    /// другая — на 4Cα-lower (-925). Мкм.
    /// </summary>
    private const float SUBLAYER_Z_OFFSET = 25f;

    // ─── Геометрия слоя 6 (для Type-2 аксонов с коллатералью) ──────────────
    /// Центр слоя 6. Мкм.
    private const float LAYER6_CENTER_Z = -1200f;

    /// Верхняя граница слоя 6. Мкм.
    private const float LAYER6_TOP_Z = -1100f;

    /// Нижняя граница слоя 6. Мкм.
    private const float LAYER6_BOTTOM_Z = -1300f;

    // ─── Параметры ствола аксона ────────────────────────────────────────────
    /// Z-координата начала аксона (в белом веществе). Мкм.
    private const float AXON_START_Z = -1350f;

    /// Максимальное случайное горизонтальное смещение ствола (XY-дрейф). Мкм.
    private const float TRUNK_XY_DRIFT = 30f;

    /// Шаг точек вдоль ствола аксона. Мкм.
    private const float TRUNK_STEP_MKM = 20f;

    // ─── Параметры главного арбора в 4Cα ───────────────────────────────────
    /// Минимальный эффективный радиус арбора. Мкм. Blasdel & Lund 1983.
    private const float ARBOR_RADIUS_MIN = 300f;

    /// Максимальный эффективный радиус арбора. Мкм. Blasdel & Lund 1983.
    private const float ARBOR_RADIUS_MAX = 600f;

    /// Число уровней дихотомического ветвления (включая уровень 0).
    private const int BRANCH_LEVELS = 5;

    /// Базовая дуговая длина ветви нулевого уровня. Мкм.
    /// При 3 первичных ветвях суммарная дуговая длина:
    /// 3 × SEG_LEN_LEVEL0 × (2^BRANCH_LEVELS − 1) / 2^(BRANCH_LEVELS−1)
    /// ≈ 3 × 1168 × 5 / 16 ≈ не так; реально суммируется по уровням:
    /// 3 × (1168 + 2×584 + 4×292 + 8×146 + 16×73) = 3 × 5840 ≈ 17.5 мм.
    /// Tortuosity ≈ 3 → реальная протяжённость ~5.8 мм. ОК.
    private const float SEG_LEN_LEVEL0 = 1168f;

    /// <summary>
    /// Разброс длины первичной арки на уровне аксона (мультипликатор).
    /// Эффективная SEG_LEN_LEVEL0 для данного аксона:
    /// SEG_LEN_LEVEL0 × U[1-σ, 1+σ], где σ = PRIMARY_ARC_LEN_JITTER.
    /// </summary>
    private const float PRIMARY_ARC_LEN_JITTER = 0.20f;

    /// <summary>
    /// Мультипликативный разброс длины дочерней ветви относительно
    /// номинальной (parent / 2). Случайно U[1-σ, 1+σ] для каждой дочерней.
    /// Источник: Anderson, Douglas & Martin 1994 — выраженная асимметрия
    /// дочерних арок в HRP-реконструкциях TC-аксонов.
    /// </summary>
    private const float BRANCH_ASYMMETRY_JITTER = 0.35f;

    /// Шаг точек вдоль ветвей арбора. Мкм.
    private const float BRANCH_STEP_MKM = 8f;

    /// <summary>
    /// Стандартное отклонение Гауссовой целевой Z для первичных ветвей. Мкм.
    /// При σ=40 мкм около 95% бутончиков попадают в центральные ~160 мкм слоя.
    /// </summary>
    private const float ARBOR_Z_SIGMA = 40f;

    /// Малое СО Z-дрейфа в подветвях (уровни ≥ 1). Мкм.
    private const float SUBBRANCH_Z_DRIFT = 15f;

    /// <summary>
    /// Вероятность завершить ветвь на уровень раньше максимального.
    /// Применяется только на уровнях BRANCH_LEVELS-2 и BRANCH_LEVELS-3,
    /// чтобы не «обрезать» арбор слишком сильно. Реалистично имитирует
    /// случайные ранние терминалы в HRP-реконструкциях.
    /// </summary>
    private const float EARLY_TERMINATION_PROB = 0.15f;

    /// <summary>
    /// Мультипликатор плотности бутончиков в дистальной части ветви.
    /// Применяется к последним TERMINAL_DENSITY_FRACTION × длины.
    /// </summary>
    private const float TERMINAL_DENSITY_BOOST = 1.5f;

    /// Доля длины ветви, считающаяся «дистальной» (с повышенной плотностью).
    private const float TERMINAL_DENSITY_FRACTION = 0.30f;

    /// Позиционный джиттер бутончика. Мкм. Соответствует размеру TC-бутончика.
    private const float BOUTON_POSITION_JITTER = 0.4f;

    // ─── Мозаичное распределение бутончиков (Markov state machine) ─────────
    /// Минимальный интервал между бутончиками в кластере. Мкм.
    private const float CLUSTER_STEP_MIN = 0.6f;

    /// Максимальный интервал в кластере. Мкм.
    private const float CLUSTER_STEP_MAX = 2.0f;

    /// Минимальный интервал в разрежённом режиме. Мкм.
    private const float SPARSE_STEP_MIN = 3.0f;

    /// Максимальный интервал в разрежённом режиме. Мкм.
    private const float SPARSE_STEP_MAX = 9.0f;

    /// <summary>
    /// Вероятность остаться в кластерном режиме.
    /// Стационарная доля кластеров: π_c = (1 − p_ss) / (2 − p_cc − p_ss)
    ///   = (1 − 0.40) / (2 − 0.75 − 0.40) = 0.60 / 0.85 ≈ 0.706 ≈ 70%. ОК.
    /// Средний шаг: 0.706 × (0.6+2.0)/2 + 0.294 × (3.0+9.0)/2
    ///   = 0.918 + 1.764 = 2.68 мкм ≈ 2.7 мкм. ОК.
    /// </summary>
    private const float CLUSTER_STAY_PROB = 0.75f;

    /// Вероятность остаться в разрежённом режиме.
    private const float SPARSE_STAY_PROB = 0.40f;

    /// Начальная вероятность стартовать ветвь в кластерном режиме.
    private const float CLUSTER_INITIAL_PROB = 0.70f;

    // ─── Концевые кластеры (terminal boutons / glomerular endings) ─────────

    /// Доля концевых ветвей, оканчивающихся крупной розеткой.
    private const float TERMINAL_ROSETTE_PROB = 0.20f;

    /// Минимальный размер обычного концевого кластера (включительно).
    private const int TERMINAL_NORMAL_MIN = 2;

    /// Максимальный размер обычного концевого кластера (исключительно).
    private const int TERMINAL_NORMAL_MAX = 8;

    /// Минимальный радиус обычного концевого кластера. Мкм.
    private const float TERMINAL_NORMAL_RADIUS_MIN = 1.0f;

    /// Максимальный радиус обычного концевого кластера. Мкм.
    private const float TERMINAL_NORMAL_RADIUS_MAX = 3.0f;

    /// Минимальный размер розетки (включительно).
    private const int TERMINAL_ROSETTE_MIN = 8;

    /// Максимальный размер розетки (исключительно).
    private const int TERMINAL_ROSETTE_MAX = 16;

    /// Минимальный радиус розетки. Мкм.
    private const float TERMINAL_ROSETTE_RADIUS_MIN = 3.0f;

    /// Максимальный радиус розетки. Мкм.
    private const float TERMINAL_ROSETTE_RADIUS_MAX = 5.0f;

    // Окулодоминантные патчи ──────────────

    /// <summary>
    /// Доля аксонов с одним патчем (строгое OD-доминирование).
    /// Остальные имеют два патча (главный + второстепенный).
    /// </summary>
    private const float OD_SINGLE_PATCH_PROB = 0.45f;

    /// <summary>
    /// Минимальный радиус OD-патча. Мкм.
    /// Изменено с 150 → 175 мкм. Клумпы бутончиков M-аксонов в HRP-реконструкциях
    /// (Blasdel & Lund 1983): 300–500 × 600–1200 мкм → эффективный радиус ~150–300 мкм.
    /// У человека OD-полосы шире (~863 мкм, Horton & Hocking 2007), поэтому
    /// нижняя граница радиуса патча увеличена.
    /// </summary>
    private const float OD_PATCH_RADIUS_MIN = 175f;

    /// <summary>
    /// Максимальный радиус OD-патча. Мкм.
    /// Изменено с 250 → 325 мкм. Соответствует верхней оценке клумп у человека.
    /// </summary>
    private const float OD_PATCH_RADIUS_MAX = 325f;

    /// <summary>
    /// Минимальное абсолютное расстояние между центрами двух патчей. Мкм.
    /// ЗАМЕНЯЕТ OD_INTER_PATCH_DISTANCE_FRACTION_MIN (относительную меру).
    /// Обоснование: ширина одной OD-полосы у человека 700–900 мкм (Horton &
    /// Hocking 2007, J.Neurosci. 27:10145). M-аксон охватывает 1–2 OD-полосы
    /// (Blasdel & Lund 1983); два его патча соответствуют двум смежным полосам
    /// одного глаза (главная) и противоположного (второстепенная), разделённым
    /// ровно одной полосой шириной ~863 мкм. Минимум взят с 10%-запасом.
    /// </summary>
    private const float OD_INTER_PATCH_DIST_MIN = 700f;

    /// <summary>
    /// Максимальное абсолютное расстояние между центрами двух патчей. Мкм.
    /// Верхняя граница: следующая OD-полоса того же глаза располагается через
    /// ~1700–1900 мкм. Для двухпатчевых арборов берём консервативный максимум
    /// 1000 мкм (расстояние до ближайшей колонки противоположного глаза).
    /// </summary>
    private const float OD_INTER_PATCH_DIST_MAX = 1000f;

    /// <summary>
    /// Базовая вероятность принятия бутончика концевого кластера ВНЕ всех патчей.
    /// Применяется ТОЛЬКО к концевым кластерам (AddTerminalCluster),
    /// но НЕ к en-passant бутончикам в BuildBranch.
    /// 0.30 → плотность концевых кластеров вне патчей в ~3.3 раза ниже,
    /// что соответствует HRP-данным магно-аксонов (Blasdel & Lund 1983,
    /// Fig. 4: клумпы бутончиков строго ограничены OD-полосами).
    /// </summary>
    private const float OD_BASELINE_ACCEPTANCE = 0.30f;

    /// <summary>
    /// Радиус «второстепенного» патча относительно главного.
    /// (Минорный патч — слабая проекция в соседнюю OD-колонку.)
    /// </summary>
    private const float OD_SECONDARY_PATCH_SIZE_RATIO = 0.65f;

    /// <summary>
    /// Сила притяжения ветви к ближайшему OD-патчу.
    /// На каждом шаге BRANCH_STEP_MKM угол ветви корректируется на величину
    /// Δθ = OD_BRANCH_BIAS_STRENGTH × sin(θ_к_патчу − θ_текущий).
    /// При 0.12 rad/шаг ветвь мягко «притягивается» к патчу, не будучи
    /// жёстко ограниченной — воспроизводит наблюдаемое в HRP-реконструкциях
    /// стягивание ветвей к OD-полосам.
    /// </summary>
    private const float OD_BRANCH_BIAS_STRENGTH = 0.12f;

    /// <summary>
    /// Мягкая радиальная граница арбора как доля от arborRadius.
    /// При расстоянии от центра > arborRadius × OD_BRANCH_RADIUS_SOFT_LIMIT
    /// начинается возвращающая сила (bias обратно к центру арбора).
    /// При расстоянии > 1.5 × arborRadius угол принудительно разворачивается
    /// к центру (жёсткое ограничение).
    /// </summary>
    private const float ARBOR_RADIUS_SOFT_LIMIT_FACTOR = 1.2f;

    // ─── Коллатераль в слой 6 (Type-2 M-аксоны) ─────────────────────────────

    private const float LAYER6_COLLATERAL_PROB = 0.40f;
    private const float LAYER6_ARBOR_RADIUS = 150f;
    private const float LAYER6_SEG_LEN_LEVEL0 = 400f;
    private const int LAYER6_BRANCH_LEVELS = 3;
    private const float LAYER6_Z_SIGMA = 35f;

    // ─── Пространственное расположение аксонов ──────────────────────────────
    /// Шаг гексагональной решётки аксонов (XY). Мкм. ~200 аксонов/мм².
    private const float GRID_SPACING_MKM = 63f;

    /// Радиус поиска на гексагональной решётке (в шагах).
    private const int GRID_SEARCH_RADIUS = 20;

    // ═══════════════════════════════════════════════════════════════════════
    // ВСПОМОГАТЕЛЬНЫЕ ТИПЫ
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Окулодоминантный патч — мягкая Гауссова маска для бутончиков.
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
    /// Контекст одного арбора (передаётся через рекурсивные вызовы
    /// BuildBranch). Содержит данные, общие для всех ветвей данного аксона.
    /// </summary>
    private sealed class ArborContext
    {
        public Vector2 Center;
        public float Radius;
        public OdPatch[] Patches = Array.Empty<OdPatch>();
        /// Эффективная SEG_LEN_LEVEL0 для данного аксона (с jitter).
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

        var ctx = new ArborContext
        {
            Center = arborCenter,
            Radius = arborRadius,
            Patches = SampleOdPatches(arborCenter, arborRadius, rng),
            Level0ArcLength = SEG_LEN_LEVEL0
                * (1f + (float)(rng.NextDouble() - 0.5) * 2f * PRIMARY_ARC_LEN_JITTER),
        };

        // коллатераль отключена по умолчанию во избежание переполнения
        // InlineArray(Size=2) при одновременном ветвлении ствола на коллатераль
        // и продолжение ствола из одного и того же узла.
        // При включении коллатерали (hasLayer6Collateral = true) необходимо
        // убедиться что узел ствола в точке ответвления имеет не более 1
        // дополнительного child (сама коллатераль), а продолжение ствола
        // идёт как следующий элемент цепочки ДО точки ответвления.
        bool hasLayer6Collateral = false;
        // bool hasLayer6Collateral = rng.NextDouble() < LAYER6_COLLATERAL_PROB;

        AxonPoint layerEntry = BuildTrunk(root, arborCenter, rng, synapsePositions, hasLayer6Collateral);

        BuildPrimaryFan(layerEntry, ctx, primaryBranches, rng, synapsePositions);

        var synapses = new FastList<Synapse>(synapsePositions.Count);
        for (int i = 0; i < synapsePositions.Count; i++)
            synapses.Add(new Synapse(synapsePositions[i]));

        return new Axon(root, synapses.ToArray());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // V4: ОКУЛОДОМИНАНТНЫЕ ПАТЧИ — СЭМПЛИРОВАНИЕ И ОЦЕНКА
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Сэмплирует 1 или 2 окулодоминантных патча для одного аксона.
    ///
    /// • Главный патч: смещён от центра арбора на 10–25% от arborRadius,
    ///   радиус 175–325 мкм.
    /// • Второстепенный патч (с вероятностью 1 − OD_SINGLE_PATCH_PROB):
    ///   на фиксированном абсолютном расстоянии 700–1000 мкм от главного.
    ///   Это соответствует периоду OD-колонок человека ~700–900 мкм
    ///   (Horton & Hocking 2007, J.Neurosci. 27:10145–10162).
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
            return new[] { new OdPatch(p0x, p0y, p0r) };

        // Второстепенный патч на АБСОЛЮТНОМ расстоянии 700–1000 мкм.
        // Это расстояние не зависит от arborRadius и отражает реальный период
        // OD-колонок в первичной зрительной коре человека.
        double secAngle = rng.NextDouble() * 2.0 * Math.PI;
        float secDist = OD_INTER_PATCH_DIST_MIN
            + (float)rng.NextDouble() * (OD_INTER_PATCH_DIST_MAX - OD_INTER_PATCH_DIST_MIN);
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
    /// Вероятность принятия бутончика концевого кластера в точке (x, y)
    /// с учётом окулодоминантных патчей. Возвращает значение в диапазоне
    /// [OD_BASELINE_ACCEPTANCE, 1.0]:
    /// • в центре любого патча → ≈ 1.0;
    /// • на границе патча (d = R) → OD_BASELINE + (1-OD_BASELINE)×e⁻¹ ≈ 0.56;
    /// • далеко от всех патчей → OD_BASELINE_ACCEPTANCE (0.30).
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

    /// <summary>
    /// Угловой bias ветви к ближайшему OD-патчу.
    /// Возвращает добавку к углу (rad), притягивающую ветвь к патчу.
    /// При отсутствии патчей или нахождении внутри патча возвращает 0.
    ///
    /// Параметры:
    ///   x, y — текущая позиция кончика ветви (мкм);
    ///   curAngle — текущий угол движения ветви (rad);
    ///   patches — массив OD-патчей аксона;
    ///   rng — генератор случайных чисел.
    ///
    /// Возвращаемое значение: поправка к углу (rad), знак определяет
    /// направление поворота к патчу.
    /// </summary>
    private static double ComputeOdBiasAngle(
        float x, float y, double curAngle,
        OdPatch[] patches)
    {
        if (patches.Length == 0) return 0.0;

        // Ищем ближайший патч, к которому не уже находимся «внутри».
        float bestDist2 = float.MaxValue;
        float bestDx = 0f, bestDy = 0f;
        float bestR = 0f;
        for (int i = 0; i < patches.Length; i++)
        {
            float dx = patches[i].X - x;
            float dy = patches[i].Y - y;
            float d2 = dx * dx + dy * dy;
            if (d2 < bestDist2)
            {
                bestDist2 = d2;
                bestDx = dx;
                bestDy = dy;
                bestR = patches[i].Radius;
            }
        }

        // Если уже внутри патча (d < 0.5×R) — bias не нужен.
        if (bestDist2 < (bestR * 0.5f) * (bestR * 0.5f)) return 0.0;

        // Угол к патчу.
        double angleToPatch = Math.Atan2(bestDy, bestDx);

        // Разность углов: сколько надо повернуть (в диапазоне [-π, π]).
        double deltaAngle = angleToPatch - curAngle;
        // Нормализация в [-π, π].
        while (deltaAngle > Math.PI) deltaAngle -= 2 * Math.PI;
        while (deltaAngle < -Math.PI) deltaAngle += 2 * Math.PI;

        // Bias пропорционален отклонению, но ограничен OD_BRANCH_BIAS_STRENGTH.
        // Затухает при нахождении внутри патча (bestDist < bestR).
        float distFactor = Math.Min(1.0f, MathF.Sqrt(bestDist2) / bestR);
        return Math.Sign(deltaAngle) * Math.Min(Math.Abs(deltaAngle), OD_BRANCH_BIAS_STRENGTH)
               * distFactor;
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

            // При включённой коллатерали в слой 6 необходимо
            // убедиться, что узел ствола имеет не более 2 дочерних узлов
            // (один — продолжение ствола, один — коллатераль). Это
            // гарантируется тем, что коллатераль отходит от ТЕКУЩЕГО узла
            // (current), который затем продолжает ствол как ЕДИНСТВЕННЫЙ
            // следующий узел. При таком порядке AddNext вызывается максимум
            // 2 раза для одного узла: один для продолжения ствола (сделано
            // выше) и один для коллатерали. Но так как AddNext(next) уже
            // вызван выше (current = next), то далее коллатераль нужно
            // присоединять к ПРЕДЫДУЩЕМУ узлу, а не к текущему.
            // Для простоты: при hasLayer6Collateral коллатераль создаётся
            // от current (уже следующего узла), это корректно для Size=2
            // только если коллатераль — единственный child этого узла
            // кроме его собственного next в стволе. Текущая реализация
            // BuildLayer6Collateral вызывает trunkPoint.AddNext(collateralRoot),
            // что при current.AddNext(next) уже выполненном даёт Next[1]=collateralRoot.
            // Итого 2 child — в пределах InlineArray(Size=2). Корректно.
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
    // каждая первичная ветвь получает индивидуальный подслойный таргет
    // (4Cα-upper vs 4Cα-lower).
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

        // подслойный таргет — случайно upper/lower для каждой первичной ветви.
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

        // начальное состояние Markov-цепи.
        bool inCluster = rng.NextDouble() < CLUSTER_INITIAL_PROB;
        // Первый интервал семплируется в текущем состоянии.
        float nextSynapseAt = SampleBoutonIntervalCurrentState(inCluster, rng);
        // Затем делаем переход для следующего шага.
        UpdateMarkovState(ref inCluster, rng);

        float terminalBoostStartDist = arcLength * (1f - TERMINAL_DENSITY_FRACTION);

        // Предвычисляем мягкую и жёсткую границу арбора.
        float softLimit = ctx.Radius * ARBOR_RADIUS_SOFT_LIMIT_FACTOR;
        float hardLimit = ctx.Radius * 1.5f;

        while (distCovered < arcLength)
        {
            float step = Math.Min(BRANCH_STEP_MKM, arcLength - distCovered);

            // Случайное блуждание угла.
            curAngle += (rng.NextDouble() - 0.5) * 2.0 * xyAngleNoise;

            // OD-притягивающий bias ветви к ближайшему патчу.
            curAngle += ComputeOdBiasAngle(x, y, curAngle, ctx.Patches);

            // Радиальный bias при выходе за мягкую границу арбора.
            float rdx = x - ctx.Center.X;
            float rdy = y - ctx.Center.Y;
            float dist = MathF.Sqrt(rdx * rdx + rdy * rdy);
            if (dist > softLimit)
            {
                // Угол к центру арбора.
                double angleToCenter = Math.Atan2(-rdy, -rdx);
                double deltaBack = angleToCenter - curAngle;
                while (deltaBack > Math.PI) deltaBack -= 2 * Math.PI;
                while (deltaBack < -Math.PI) deltaBack += 2 * Math.PI;

                if (dist > hardLimit)
                {
                    // Жёсткое ограничение: принудительно разворачиваем к центру.
                    curAngle = angleToCenter + (rng.NextDouble() - 0.5) * 0.3;
                }
                else
                {
                    // Мягкое: bias нарастает линейно от softLimit до hardLimit.
                    float overshootFactor = (dist - softLimit) / (hardLimit - softLimit);
                    curAngle += deltaBack * overshootFactor * 0.4;
                }
            }

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

                // En-passant бутончики добавляются БЕЗ OD-гейтинга.
                // Пространственная концентрация обеспечивается направлением ветвей
                // к OD-патчам (ComputeOdBiasAngle выше).
                synapses.Add(new Vector3(
                    bx + (float)(rng.NextDouble() - 0.5) * BOUTON_POSITION_JITTER,
                    by + (float)(rng.NextDouble() - 0.5) * BOUTON_POSITION_JITTER,
                    bz + (float)(rng.NextDouble() - 0.5) * BOUTON_POSITION_JITTER));

                // Следующий интервал семплируется в текущем состоянии,
                // затем делаем Markov-переход.
                float interval = SampleBoutonIntervalCurrentState(inCluster, rng);
                if (nextSynapseAt > terminalBoostStartDist)
                    interval /= TERMINAL_DENSITY_BOOST;
                nextSynapseAt += interval;

                UpdateMarkovState(ref inCluster, rng);
            }
        }

        // концевой кластер с гейтингом по OD-патчам (сохранён).
        AddTerminalCluster(x, y, z, ctx.Patches, rng, synapses);

        if (level < BRANCH_LEVELS - 1)
        {
            if (level >= BRANCH_LEVELS - 3 && rng.NextDouble() < EARLY_TERMINATION_PROB)
                return;

            double spreadBase = Math.PI / 4.5 - level * 0.05;
            double spreadL = spreadBase * (1.0 + (rng.NextDouble() - 0.5) * 0.4);
            double spreadR = spreadBase * (1.0 + (rng.NextDouble() - 0.5) * 0.4);

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
        float nextSynapseAt = SampleBoutonIntervalCurrentState(inCluster, rng);
        UpdateMarkovState(ref inCluster, rng);

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

                float interval = SampleBoutonIntervalCurrentState(inCluster, rng);
                nextSynapseAt += interval;
                UpdateMarkovState(ref inCluster, rng);
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
    /// Семплирует интервал до следующего бутончика
    /// в ТЕКУЩЕМ состоянии (без изменения состояния).
    /// Переход делается отдельно через UpdateMarkovState.
    ///
    /// Параметры:
    ///   inCluster — текущее состояние Markov-цепи
    ///     (true = кластерный режим, false = разрежённый);
    ///   rng — генератор случайных чисел.
    ///
    /// Возвращает: расстояние до следующего бутончика (мкм).
    /// </summary>
    private static float SampleBoutonIntervalCurrentState(bool inCluster, Random rng)
    {
        if (inCluster)
            return CLUSTER_STEP_MIN + (float)rng.NextDouble() * (CLUSTER_STEP_MAX - CLUSTER_STEP_MIN);
        else
            return SPARSE_STEP_MIN + (float)rng.NextDouble() * (SPARSE_STEP_MAX - SPARSE_STEP_MIN);
    }

    /// <summary>
    /// Делает переход Markov-цепи cluster↔sparse.
    ///
    /// Параметр:
    ///   inCluster — текущее состояние (изменяется in-place).
    ///
    /// Стационарное распределение:
    ///   π_cluster = (1 − p_ss) / (2 − p_cc − p_ss)
    ///             = (1 − 0.40) / (2 − 0.75 − 0.40) = 0.706 ≈ 70%.
    /// </summary>
    private static void UpdateMarkovState(ref bool inCluster, Random rng)
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
    }

    // ═══════════════════════════════════════════════════════════════════════
    // КОНЦЕВЫЕ КЛАСТЕРЫ (terminal boutons / glomerular endings)
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Размещает концевой кластер бутончиков вокруг точки (x,y,z).
    /// С вероятностью TERMINAL_ROSETTE_PROB формирует крупную «розетку»
    /// (8–15 бутончиков), иначе обычный малый кластер (2–7 бутончиков).
    ///
    /// Каждый отдельный бутончик дополнительно проходит вероятностный
    /// отбор по OD-маске (сохранено из V3 — это корректно для концевых
    /// розеток, которые строго локализованы в OD-патчах).
    ///
    /// Параметры:
    ///   x, y, z — позиция конца ветви (мкм);
    ///   patches — OD-патчи данного аксона;
    ///   rng — генератор случайных чисел;
    ///   synapses — список синапсов для добавления.
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
        
        // Для слоя 6 (patches.Length == 0) масштаб = 1.0 (без ограничений).
        if (patches.Length > 0)
        {
            float patchAccept = PatchAcceptance(x, y, patches);
            // Нормализованная позиция: 0.0 = вне патча (baseline), 1.0 = центр патча.
            float normalizedAccept = (patchAccept - OD_BASELINE_ACCEPTANCE)
                / (1f - OD_BASELINE_ACCEPTANCE);
            // Масштаб кластера: от 0.25 (вне патча) до 1.0 (в центре).
            float clusterScale = 0.25f + 0.75f * normalizedAccept;
            count = (int)Math.Round(count * clusterScale);
            if (count < 1) count = 1;
        }

        for (int t = 0; t < count; t++)
        {
            // Концевые гроздья плоские: Z-разброс вдвое меньше XY.
            float bx = x + (float)(rng.NextDouble() - 0.5) * 2 * radius;
            float by = y + (float)(rng.NextDouble() - 0.5) * 2 * radius;
            float bz = z + (float)(rng.NextDouble() - 0.5) * radius;

            // Дополнительный бутонный гейтинг по OD-маске (сохранён из V3).
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
    ///
    /// Параметры:
    ///   startZ — Z-координата начала ветви (мкм);
    ///   level — текущий уровень ветвления (0 = первичная);
    ///   centerZ — центральная Z-координата слоя (мкм);
    ///   sigma — стандартное отклонение Гауссового таргета (мкм);
    ///   bottomZ — нижняя граница слоя (мкм);
    ///   topZ — верхняя граница слоя (мкм);
    ///   rng — генератор случайных чисел.
    ///
    /// Возвращает: целевую Z-координату, зажатую в [bottomZ+5, topZ-5] (мкм).
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
    ///
    /// Параметры:
    ///   rng — генератор случайных чисел (double равномерное [0,1)).
    ///
    /// Возвращает: значение из N(0,1).
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
    ///
    /// Параметры:
    ///   rng — генератор случайных чисел.
    ///
    /// Источник: Blasdel & Lund 1983.
    /// </summary>
    private static float SampleArborRadius(Random rng)
    {
        float radius = 400f + (float)(SampleGaussian(rng) * 70f);
        return Math.Clamp(radius, ARBOR_RADIUS_MIN, ARBOR_RADIUS_MAX);
    }
}
