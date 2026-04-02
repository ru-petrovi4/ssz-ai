using System;
using System.Collections.Generic;
using System.Numerics;
using Ssz.Utils;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
// ТАЛАМОКОРТИКАЛЬНЫЙ ВХОДНОЙ БЛОК МИНИКОЛОНКИ
// ============================================================
//
// Биологическое обоснование числа аксонов:
//
// Плотность ТК-аксонов в V1 человека:
//   ~1 417 аксонов/мм² (Mazade & Alonso 2017, Visual Neuroscience 34:E007)
//   Всего ~3.4 млн LGN-нейронов → ~2 399 мм² V1 => 3.4×10⁶ / 2399 = 1 417 /мм²
//
// Соотношение M:P:K нейронов LGN человека (Solomon 2021; Wikipedia LGN):
//   M (magnocellular) ≈ 10% → 142 аксонов/мм²
//   P (parvocellular) ≈ 80% → 1 134 аксонов/мм²
//   K (koniocellular) ≈ 10% → 142 аксонов/мм²  ← примерно, K ≈ 10% relay-клеток
//
// Размер миниколонки V1 человека (Garcia-Marin et al. 2013, Cereb Cortex):
//   Поперечный размер тяжа ≈ 24.7 мкм; spacing (центр–центр) ≈ 30–50 мкм.
//   Площадь «территории» ≈ 24.7² = 610 мкм² (минимум) до 50² = 2 500 мкм².
//   Миниколонок на мм²: ~400–1 640.
//
// Число аксонов, арборы которых ПЕРЕКРЫВАЮТ данную миниколонку:
//   Формула: N = плотность_типа × площадь_арбора
//
//   M (арбор r=300 мкм, площадь π×0.3²=0.283 мм²):
//     N_M = 142 × 0.283 ≈ 40 аксонов
//
//   P (арбор r=175 мкм, площадь π×0.175²=0.096 мм²):
//     N_P = 1 134 × 0.096 ≈ 109 аксонов
//
//   K (арбор r=125 мкм, площадь π×0.125²=0.049 мм²):
//     N_K = 142 × 0.049 ≈ 7 аксонов
//
//   ИТОГО: ~156 аксонов покрывают одну миниколонку.
//
// Источники:
//   - Mazade & Alonso 2017, Vis Neurosci 34:E007 (плотность 1 417/мм²)
//   - Solomon 2021, J Physiol 599:2893 (соотношение M:P:K)
//   - Andrews, Halpern, Purves 1997, J Neurosci 17:2859 (объём LGN, площадь V1)
//   - Garcia-Marin et al. 2013 (via Garcia-Marin et al. 2024, Cereb Cortex):
//     размер миниколонки V1 человека = 24.7 мкм
//   - Blasdel & Lund 1983, J Neurosci 3:1389 (диаметры арборов macaque)
//   - Casagrande et al. 2007 (K-арборы)
// ============================================================

public sealed class ThalamocorticalInput
{
    // ----------------------------------------------------------
    // КОЛИЧЕСТВО ТК-АКСОНОВ ПО ТИПАМ
    //
    // Каждый аксон в массиве представляет один LGN-нейрон, чей
    // горизонтальный арбор перекрывает данную миниколонку.
    // Точка входа аксона в WM случайно распределена внутри
    // радиуса арбора (см. ThalamocorticalAxon.Generate),
    // поэтому «собственные» и «соседские» аксоны моделируются
    // единообразно — разница лишь в координатах точки входа.
    //
    // Расчёт: N = плотность_типа_в_V1 × площадь_арбора_типа
    //
    // M: 142/мм² × π×(300 мкм)² = 142 × 0.2827 мм² ≈ 40
    // P: 1134/мм² × π×(175 мкм)² = 1134 × 0.0962 мм² ≈ 109
    // K:  142/мм² × π×(125 мкм)² = 142  × 0.0491 мм² ≈ 7
    //
    // Итого: 40 + 109 + 7 = 156 аксонов на миниколонку.
    // ----------------------------------------------------------

    /// <summary>
    /// Число магноцеллюлярных (M) аксонов, покрывающих данную миниколонку.
    /// Расчёт: плотность M в V1 (~142/мм²) × площадь арбора M (~0.283 мм²) ≈ 40.
    /// Источник: Mazade & Alonso 2017; Blasdel & Lund 1983.
    /// </summary>
    public const int MAxonCount = 40;

    /// <summary>
    /// Число парвоцеллюлярных (P) аксонов, покрывающих данную миниколонку.
    /// Расчёт: плотность P в V1 (~1134/мм²) × площадь арбора P (~0.096 мм²) ≈ 109.
    /// P-путь доминирует (80% клеток LGN), но небольшой арбор (~175 мкм радиус).
    /// Источник: Mazade & Alonso 2017; Blasdel & Lund 1983.
    /// </summary>
    public const int PAxonCount = 109;

    /// <summary>
    /// Число конийоцеллюлярных (K) аксонов, покрывающих данную миниколонку.
    /// Расчёт: плотность K в V1 (~142/мм²) × площадь арбора K (~0.049 мм²) ≈ 7.
    /// K-путь имеет наименьшее число аксонов на миниколонку — малый арбор и равная
    /// M-пути плотность нейронов (~10% LGN) дают существенно меньший охват.
    /// Источник: Solomon 2021; Casagrande et al. 2007.
    /// </summary>
    public const int KAxonCount = 7;

    // ----------------------------------------------------------
    // ИТОГОВЫЙ СЧЁТЧИК
    // ----------------------------------------------------------

    /// <summary>
    /// Суммарное число всех ТК-аксонов, покрывающих данную миниколонку.
    /// M(40) + P(109) + K(7) = 156.
    /// </summary>
    public const int TotalAxonCount = MAxonCount + PAxonCount + KAxonCount;

    // ----------------------------------------------------------
    // ДАННЫЕ
    // ----------------------------------------------------------

    /// <summary>
    /// Все ТК-аксоны (M + P + K), чьи горизонтальные арборы достигают
    /// данной миниколонки. Аксоны, чья точка входа в WM находится за
    /// пределами радиуса миниколонки, моделируются с entryMaxRadius =
    /// arborRadius + columnRadiusUm, поэтому корректно отражают «соседские»
    /// входы от нейронов ЛКТ, принадлежащих соседним топографическим точкам.
    /// </summary>
    public readonly ThalamocorticalAxon[] ThalamocorticalAxons;

    // ============================================================
    // КОНСТРУКТОР
    // ============================================================

    /// <summary>
    /// Генерирует все таламокортикальные входящие аксоны для миниколонки.
    /// Общее число: MAxonCount + PAxonCount + KAxonCount = 156.
    /// </summary>
    /// <param name="random">Генератор случайных чисел.</param>
    /// <param name="columnRadiusUm">Радиус миниколонки (мкм).</param>
    /// <param name="columnHeightUm">Высота миниколонки (мкм).</param>
    public ThalamocorticalInput(Random random, float columnRadiusUm, float columnHeightUm)
    {
        ThalamocorticalAxons = new ThalamocorticalAxon[TotalAxonCount];

        int globalIdx = 0; // сквозной индекс в массиве

        // --- M-аксоны (магноцеллюлярные) ---
        // 40 аксонов из слоёв 1–2 LGN → слой L4Cα (Z: -600..-750 мкм)
        // Арбор большой (r=300 мкм), поэтому охватывает несколько миниколонок.
        for (int i = 0; i < MAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.Magnocellular,
                random,
                columnRadiusUm,
                columnHeightUm);

            ThalamocorticalAxons[globalIdx] = axon;
            globalIdx += 1;
        }

        // --- P-аксоны (парвоцеллюлярные) ---
        // 109 аксонов из слоёв 3–6 LGN → слой L4Cβ (Z: -750..-900 мкм)
        // Доминируют численно: 80% нейронов LGN — парвоцеллюлярные.
        for (int i = 0; i < PAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.Parvocellular,
                random,
                columnRadiusUm,
                columnHeightUm);

            ThalamocorticalAxons[globalIdx] = axon;
            globalIdx += 1;
        }

        // --- K-аксоны (конийоцеллюлярные) ---
        // 7 аксонов из K-слоёв LGN → L1 и blob-зоны L2/3.
        // Наименьшее число на миниколонку из-за малого арбора (r=125 мкм).
        for (int i = 0; i < KAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.Koniocellular,
                random,
                columnRadiusUm,
                columnHeightUm);

            ThalamocorticalAxons[globalIdx] = axon;
            globalIdx += 1;
        }
    }

    // ============================================================
    // АКТИВАЦИЯ ВХОДЯЩЕГО ЗРИТЕЛЬНОГО СИГНАЛА
    // ============================================================

    /// <summary>
    /// Устанавливает активность ТК-аксонов на основе
    /// вектора входящего зрительного сигнала.
    /// Длина массива activity должна совпадать с TotalAxonCount (156).
    /// </summary>
    /// <param name="activity">Массив активностей [0..1] длиной TotalAxonCount.</param>
    public void SetActivity(float[] activity)
    {
        for (int i = 0; i < activity.Length; i += 1)
            ThalamocorticalAxons[i].Temp_IsActive = activity[i] > 0.5f;
    }

    // ============================================================
    // СБОР АКТИВНЫХ СИНАПТИЧЕСКИХ ПОЗИЦИЙ
    // ============================================================

    /// <summary>
    /// Собирает позиции всех активных синапсов.
    /// Используется в FindActiveZones для расчёта зон активации.
    /// </summary>
    /// <returns>Список позиций активных ТК-синапсов внутри колонки.</returns>
    public FastList<Vector3> GetActiveAfferentSynapsePositions()
    {
        var result = new FastList<Vector3>(TotalAxonCount * 1000);

        for (int a = 0; a < TotalAxonCount; a += 1)
        {
            if (!ThalamocorticalAxons[a].Temp_IsActive)
                continue;

            var synapses = ThalamocorticalAxons[a].Synapses;

            for (int s = 0; s < synapses.Length; s += 1)
                result.Add(synapses[s].Position);
        }

        return result;
    }
}
