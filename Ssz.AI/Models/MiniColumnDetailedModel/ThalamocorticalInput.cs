using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Ssz.AI.Helpers;
using Ssz.Utils;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
// ТАЛАМОКОРТИКАЛЬНЫЙ ВХОДНОЙ БЛОК МИНИКОЛОНКИ
// ============================================================
//
// ────────────────────────────────────────────────────────────
// ПРИНЦИП ОТБОРА АКСОНОВ
// ────────────────────────────────────────────────────────────
// Критерий «влияет на данную миниколонку»:
//
//   Старый критерий (геометрия арбора → центр колонки):
//     аксон учитывается, если его горизонтальный арбор
//     ПЕРЕКРЫВАЕТ ЦЕНТР миниколонки.
//     Эффективный радиус = arborRadius + columnRadius
//
//   Новый критерий (арбор → дендрит любого нейрона колонки):
//     аксон учитывается, если синапт его арбора ДОСТИГАЕТ
//     ДЕНДРИТА любого нейрона, расположенного в колонке.
//     Дендриты простираются за пределы колонки на dendriticReach.
//     Эффективный радиус = arborRadius + columnRadius + dendriticReach
//
// Биологическое обоснование dendriticReach:
//   Горизонтальный охват базальных дендритов пирамидных клеток
//   слоя 3 V1 макак составляет 194 ± 15 мкм (Amatrudo et al.
//   2012, J. Neurosci. 32:1480–1491), откуда радиус ≈ 97 мкм.
//   Апикальный туфт: горизонтальный охват ≈ 139 ± 15 мкм,
//   радиус ≈ 70 мкм. Берётся более широкий из двух (базальный),
//   т.к. базальные дендриты принимают входы из всех слоёв 2/3–4.
//   Источник: Amatrudo et al. 2012, J. Neurosci. 32:1480–1491
//             (сравнение V1 и dlPFC макак; Table 1).
//
// ────────────────────────────────────────────────────────────
// ПЛОТНОСТЬ АКСОНОВ В V1 ЧЕЛОВЕКА
// ────────────────────────────────────────────────────────────
//   Всего  ~1 417 аксонов/мм² (Mazade & Alonso 2017, Vis Neurosci 34:E007)
//   M  (~10% LGN):       142 /мм²
//   P  (~80% LGN):     1 134 /мм²
//   K_sup  (K1/K2 ≈ 33% K):  47 /мм²
//   K_blob (K3–K6 ≈ 67% K):  95 /мм²
//   Источник: Solomon 2021, J Physiol 599:2893;
//             Casagrande et al. 2007, Cereb Cortex 17:2334
//
// ────────────────────────────────────────────────────────────
// ПАРАМЕТРЫ МИНИКОЛОНКИ
// ────────────────────────────────────────────────────────────
//   Радиус ≈ 12.35 мкм (диаметр 24.7 мкм;
//   Garcia-Marin et al. 2013, Cereb Cortex)
//
// ────────────────────────────────────────────────────────────
// РАСЧЁТ ЧИСЛА АКСОНОВ
// ────────────────────────────────────────────────────────────
//   Формула: N = плотность × π × (arborRadius + columnRadius + dendriticReach)²
//   dendriticReach = 97 мкм (базальные дендриты)
//
//   M:      142 × π × (300 + 12.35 + 97)² мкм² = 142 × π × 409.4² ≈  75
//   P:    1 134 × π × (175 + 12.35 + 97)² мкм² =1134 × π × 284.4² ≈ 288
//   K_sup:   47 × π × (137.5+12.35+97)² мкм²  =  47 × π × 246.8² ≈   9
//   K_blob:  95 × π × (112.5+12.35+97)² мкм²  =  95 × π × 221.8² ≈  15
//
//   ИТОГО: 75 + 288 + 9 + 15 = 387 аксонов на миниколонку.
//
// Источники:
//   - Mazade & Alonso 2017, Vis Neurosci 34:E007
//   - Solomon 2021, J Physiol 599:2893
//   - Casagrande et al. 2007, Cereb Cortex 17:2334–2345
//   - Garcia-Marin et al. 2013, Cereb Cortex
//   - Blasdel & Lund 1983, J Neurosci 3:1389
//   - Amatrudo et al. 2012, J Neurosci 32:1480–1491
//   - Hendry & Reid 2000, Annu Rev Neurosci 23:127–153
// ============================================================

public sealed class ThalamocorticalInput
{
    // ----------------------------------------------------------
    // ДЕНДРИТНЫЙ ОХВАТ ПИРАМИДНЫХ КЛЕТОК V1
    // ----------------------------------------------------------

    ///
    /// Горизонтальный радиус базальных дендритов пирамидных клеток
    /// слоя 3 в V1 (мкм). Определяет, насколько далеко за пределы
    /// миниколонки дендриты нейронов могут принимать синаптические входы.
    ///
    /// Значение 97 мкм — половина от 194 ± 15 мкм горизонтального
    /// охвата базальных дендритов пирамид L3 V1 макак.
    /// Источник: Amatrudo et al. 2012, J. Neurosci. 32:1480–1491 (Table 1).
    ///
    public const float DendriticReachUm = 97.0f;

    // ----------------------------------------------------------
    // КОЛИЧЕСТВО ТК-АКСОНОВ ПО ТИПАМ
    //
    // Расчёт: N = density × π × (arborRadius + columnRadius + DendriticReachUm)²
    //
    // Каждый аксон в массиве представляет один LGN-нейрон, чей
    // горизонтальный арбор способен достигать дендритов хотя бы
    // одного нейрона данной миниколонки.
    // ----------------------------------------------------------

    ///
    /// Число M-аксонов, чьи арборы досягают дендритов нейронов миниколонки.
    /// R_eff = 300 + 12.35 + 97 = 409.4 мкм
    /// N_M = 142 × π × 0.4094² ≈ 75.
    /// Источник: Mazade & Alonso 2017; Blasdel & Lund 1983;
    ///           Amatrudo et al. 2012.
    ///
    public const int MAxonCount = 75;

    ///
    /// Число P-аксонов, чьи арборы досягают дендритов нейронов миниколонки.
    /// R_eff = 175 + 12.35 + 97 = 284.4 мкм
    /// N_P = 1 134 × π × 0.2844² ≈ 288.
    /// Источник: Mazade & Alonso 2017; Blasdel & Lund 1983;
    ///           Amatrudo et al. 2012.
    ///
    public const int PAxonCount = 288;

    ///
    /// Число K_sup-аксонов (K1/K2), чьи арборы досягают дендритов нейронов.
    /// R_eff = 137.5 + 12.35 + 97 = 246.8 мкм
    /// N_K_sup = 47 × π × 0.2468² ≈ 9.
    /// Источник: Casagrande et al. 2007; Amatrudo et al. 2012.
    ///
    public const int KSupAxonCount = 9;

    ///
    /// Число K_blob-аксонов (K3–K6), чьи арборы досягают дендритов нейронов.
    /// R_eff = 112.5 + 12.35 + 97 = 221.8 мкм
    /// N_K_blob = 95 × π × 0.2218² ≈ 15.
    /// Источник: Casagrande et al. 2007; Amatrudo et al. 2012.
    ///
    public const int KBlobAxonCount = 15;

    ///
    /// Суммарное число ТК-аксонов.
    /// M(75) + P(288) + K_sup(9) + K_blob(15) = 387.
    ///
    public const int TotalAxonCount = MAxonCount + PAxonCount + KSupAxonCount + KBlobAxonCount;

    // ----------------------------------------------------------
    // ДАННЫЕ
    // ----------------------------------------------------------

    ///
    /// Все ТК-аксоны (M + P + K_sup + K_blob), чьи горизонтальные
    /// арборы способны достигать дендритов нейронов данной миниколонки.
    ///
    public readonly ThalamocorticalAxon[] ThalamocorticalAxons;

    public readonly ThalamocorticalAxon[] Top200_M_P_ThalamocorticalAxons;

    // ============================================================
    // КОНСТРУКТОР
    // ============================================================

    ///
    /// Генерирует все таламокортикальные входящие аксоны.
    /// Критерий включения — арбор достигает дендритов нейронов
    /// данной миниколонки (радиус захвата = arborRadius + columnRadius
    /// + DendriticReachUm).
    ///
    /// <param name="random">Генератор случайных чисел.</param>
    /// <param name="columnRadiusUm">Радиус миниколонки (мкм).</param>
    /// <param name="columnHeightUm">Высота миниколонки (мкм).</param>
    public ThalamocorticalInput(Random random, float columnRadiusUm, float columnHeightUm)
    {
        FastList<ThalamocorticalAxon> thalamocorticalAxons = new (TotalAxonCount);
        FastList<ThalamocorticalAxon> m_P_ThalamocorticalAxons = new(TotalAxonCount);

        // --- M-аксоны (75) ---
        // Арбор r=300 мкм; захват = 300 + 12.35 + 97 = 409.4 мкм.
        // M-путь: движение, контраст → L4Cα (−600..−750 мкм).
        for (int i = 0; i < MAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                ThalamocorticalType.Magnocellular,
                random,
                columnRadiusUm,
                columnHeightUm,
                DendriticReachUm);
            thalamocorticalAxons.Add(axon);
            m_P_ThalamocorticalAxons.Add(axon);
        }

        // --- P-аксоны (288) ---
        // Арбор r=175 мкм; захват = 175 + 12.35 + 97 = 284.4 мкм.
        // P-путь: форма, цвет → L4Cβ (−750..−900 мкм).
        // Доминируют: плотность 1134/мм² × увеличенный эффективный
        // радиус → 288 аксонов — основной источник сенсорного входа.
        for (int i = 0; i < PAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                ThalamocorticalType.Parvocellular,
                random,
                columnRadiusUm,
                columnHeightUm,
                DendriticReachUm);
            thalamocorticalAxons.Add(axon);
            m_P_ThalamocorticalAxons.Add(axon);
        }

        // --- K_sup-аксоны (9) ---
        // Арбор r=137.5 мкм; захват = 137.5 + 12.35 + 97 = 246.8 мкм.
        // K1/K2: диффузный вход в L1 + L3A.
        for (int i = 0; i < KSupAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                ThalamocorticalType.KoniocellularSuperficial,
                random,
                columnRadiusUm,
                columnHeightUm,
                DendriticReachUm);
            thalamocorticalAxons.Add(axon);
        }

        // --- K_blob-аксоны (15) ---
        // Арбор r=112.5 мкм; захват = 112.5 + 12.35 + 97 = 221.8 мкм.
        // K3–K6: S-конус хроматика → CO-блобы L3Bα.
        for (int i = 0; i < KBlobAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                ThalamocorticalType.KoniocellularBlob,
                random,
                columnRadiusUm,
                columnHeightUm,
                DendriticReachUm);
            thalamocorticalAxons.Add(axon);
        }

        ThalamocorticalAxons = thalamocorticalAxons.ToArray();
        Top200_M_P_ThalamocorticalAxons = m_P_ThalamocorticalAxons.OrderBy(a => MathHelper.GetLengthXY(a.Root.Position)).Take(200).ToArray();
    }
}
