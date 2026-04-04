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
//   ~1 417 аксонов/мм² (Mazade & Alonso 2017, Vis Neurosci 34:E007)
//   Всего ~3.4 млн LGN-нейронов → ~2 399 мм² V1
//   => 3.4×10⁶ / 2399 ≈ 1 417 /мм²
//
// Соотношение M:P:K нейронов LGN человека:
//   M (magnocellular)    ≈ 10% → 142 аксонов/мм²
//   P (parvocellular)    ≈ 80% → 1 134 аксонов/мм²
//   K (koniocellular)    ≈ 10% → 142 аксонов/мм²
//   Источник: Solomon 2021, J Physiol 599:2893
//
// Разделение K-аксонов на два подтипа (Casagrande et al. 2007,
//   Cereb. Cortex 17:2334):
//
//   K1/K2 (KoniocellularSuperficial):
//     Вентральные K-слои LGN; проекция в L1 + L3A.
//     В макаке K-слои 6 (K1–K6), из них вентральные — K1, K2.
//     Доля: ~2/6 = 33% K-нейронов → ~47 аксонов/мм² (K_sup).
//
//   K3–K6 (KoniocellularBlob):
//     Дорсальные K-слои LGN; проекция в CO-блобы L3Bα.
//     Доля: ~4/6 = 67% K-нейронов → ~95 аксонов/мм² (K_blob).
//
// Размер миниколонки V1 человека:
//   Поперечный размер ≈ 24.7 мкм; spacing ≈ 30–50 мкм.
//   Источник: Garcia-Marin et al. 2013, Cereb Cortex
//
// Число аксонов, арборы которых ПЕРЕКРЫВАЮТ данную миниколонку:
//   Формула: N = плотность_типа × площадь_арбора
//
//   M (арбор r=300 мкм, площадь π×0.3²=0.2827 мм²):
//     N_M = 142 × 0.2827 ≈ 40 аксонов
//
//   P (арбор r=175 мкм, площадь π×0.175²=0.0962 мм²):
//     N_P = 1 134 × 0.0962 ≈ 109 аксонов
//
//   K_sup (K1/K2, арбор r=137.5 мкм, площадь π×0.1375²=0.0594 мм²):
//     N_K_sup = 47 × 0.0594 ≈ 3 аксона
//
//   K_blob (K3–K6, арбор r=112.5 мкм, площадь π×0.1125²=0.0398 мм²):
//     N_K_blob = 95 × 0.0398 ≈ 4 аксона
//
// ИТОГО: 40 + 109 + 3 + 4 = 156 аксонов на миниколонку.
//
//
// Источники:
//   - Mazade & Alonso 2017, Vis Neurosci 34:E007
//   - Solomon 2021, J Physiol 599:2893 (соотношение M:P:K)
//   - Casagrande et al. 2007, Cereb. Cortex 17:2334–2345
//     (K1/K2 → L1+L3A; K3–K6 → CO-blobs L3Bα)
//   - Hendry & Reid 2000, Annu. Rev. Neurosci. 23:127–153
//   - Andrews, Halpern, Purves 1997, J Neurosci 17:2859
//   - Garcia-Marin et al. 2013/2024, Cereb Cortex
//   - Blasdel & Lund 1983, J Neurosci 3:1389 (диаметры арборов)
// ============================================================

public sealed class ThalamocorticalInput
{
    // ----------------------------------------------------------
    // КОЛИЧЕСТВО ТК-АКСОНОВ ПО ТИПАМ
    // ----------------------------------------------------------

    ///
    /// Число магноцеллюлярных (M) аксонов, покрывающих миниколонку.
    /// N_M = 142/мм² × π×(300 мкм)² ≈ 40.
    /// Источник: Mazade & Alonso 2017; Blasdel & Lund 1983.
    ///
    public const int MAxonCount = 40;

    ///
    /// Число парвоцеллюлярных (P) аксонов, покрывающих миниколонку.
    /// N_P = 1134/мм² × π×(175 мкм)² ≈ 109.
    /// Источник: Mazade & Alonso 2017; Blasdel & Lund 1983.
    ///
    public const int PAxonCount = 109;

    ///
    /// Число конийоцеллюлярных поверхностных (K_sup, K1/K2) аксонов.
    /// Источник K1/K2: ~33% всех K-нейронов → ~47/мм².
    /// N_K_sup = 47/мм² × π×(137.5 мкм)² ≈ 3.
    /// Проекция: L1 + L3A (диффузный, простой арбор).
    /// Источник: Casagrande et al. 2007; Hendry & Reid 2000.
    ///
    public const int KSupAxonCount = 3;

    ///
    /// Число конийоцеллюлярных блоб-аксонов (K_blob, K3–K6).
    /// Источник K3–K6: ~67% всех K-нейронов → ~95/мм².
    /// N_K_blob = 95/мм² × π×(112.5 мкм)² ≈ 4.
    /// Проекция: CO-блобы L3Bα (компактный, фокусированный арбор).
    /// Источник: Casagrande et al. 2007; 2020 Cereb. Cortex review.
    ///
    public const int KBlobAxonCount = 4;

    ///
    /// Суммарное число всех ТК-аксонов, покрывающих миниколонку.
    /// M(40) + P(109) + K_sup(3) + K_blob(4) = 156.
    ///
    public const int TotalAxonCount = MAxonCount + PAxonCount + KSupAxonCount + KBlobAxonCount;

    // ----------------------------------------------------------
    // ДАННЫЕ
    // ----------------------------------------------------------

    ///
    /// Все ТК-аксоны (M + P + K_sup + K_blob), чьи горизонтальные
    /// арборы достигают данной миниколонки.
    ///
    public readonly ThalamocorticalAxon[] ThalamocorticalAxons;

    // ============================================================
    // КОНСТРУКТОР
    // ============================================================

    ///
    /// Генерирует все таламокортикальные входящие аксоны для миниколонки.
    /// Общее число: M(40) + P(109) + K_sup(3) + K_blob(4) = 156.
    ///
    /// <param name="random">Генератор случайных чисел.</param>
    /// <param name="columnRadiusUm">Радиус миниколонки (мкм).</param>
    /// <param name="columnHeightUm">Высота миниколонки (мкм).</param>
    public ThalamocorticalInput(Random random, float columnRadiusUm, float columnHeightUm)
    {
        ThalamocorticalAxons = new ThalamocorticalAxon[TotalAxonCount];

        int globalIdx = 0;

        // --- M-аксоны (магноцеллюлярные) ---
        // 40 аксонов из слоёв 1–2 LGN → слой L4Cα (Z: −600..−750 мкм)
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
        // 109 аксонов из слоёв 3–6 LGN → слой L4Cβ (Z: −750..−900 мкм)
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

        // --- K_sup-аксоны (конийоцеллюлярные, K1/K2) ---
        // 3 аксона из вентральных K-слоёв LGN → L1 + L3A.
        // Морфология: простой диффузный арбор; мало бутонов (~120).
        // Источник: Casagrande et al. 2007; Hendry & Reid 2000.
        for (int i = 0; i < KSupAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.KoniocellularSuperficial,
                random,
                columnRadiusUm,
                columnHeightUm);

            ThalamocorticalAxons[globalIdx] = axon;
            globalIdx += 1;
        }

        // --- K_blob-аксоны (конийоцеллюлярные, K3–K6) ---
        // 4 аксона из дорсальных K-слоёв LGN → CO-блобы L3Bα.
        // Морфология: компактный фокусированный арбор; 217 бутонов.
        // 93% бутонов в L3Bα; редкие коллатерали в L1 и L4A (~2% каждая).
        // Источник: Casagrande et al. 2007; 2020 Cereb. Cortex review.
        for (int i = 0; i < KBlobAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.KoniocellularBlob,
                random,
                columnRadiusUm,
                columnHeightUm);

            ThalamocorticalAxons[globalIdx] = axon;
            globalIdx += 1;
        }
    }
}
