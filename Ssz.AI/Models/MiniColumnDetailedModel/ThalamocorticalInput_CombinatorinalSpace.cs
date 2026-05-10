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
public sealed class ThalamocorticalInput_CombinatorinalSpace
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
    public readonly ThalamocorticalAxon_CombinatorinalSpace[] ThalamocorticalAxons;

    public readonly ThalamocorticalAxon_CombinatorinalSpace[] TopHashLength_M_P_ThalamocorticalAxons;

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
    public ThalamocorticalInput_CombinatorinalSpace(Random random, float columnRadiusUm, float columnHeightUm, IRetinaConstants constants)
    {
        FastList<ThalamocorticalAxon_CombinatorinalSpace> thalamocorticalAxons = new (TotalAxonCount);
        FastList<ThalamocorticalAxon_CombinatorinalSpace> m_P_ThalamocorticalAxons = new(TotalAxonCount);

        // --- M-аксоны (75) ---
        // Арбор r=300 мкм; захват = 300 + 12.35 + 97 = 409.4 мкм.
        // M-путь: движение, контраст → L4Cα (−600..−750 мкм).
        for (int i = 0; i < MAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon_CombinatorinalSpace.Generate(
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
            var axon = ThalamocorticalAxon_CombinatorinalSpace.Generate(
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
            var axon = ThalamocorticalAxon_CombinatorinalSpace.Generate(
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
            var axon = ThalamocorticalAxon_CombinatorinalSpace.Generate(
                ThalamocorticalType.KoniocellularBlob,
                random,
                columnRadiusUm,
                columnHeightUm,
                DendriticReachUm);
            thalamocorticalAxons.Add(axon);
        }

        ThalamocorticalAxons = thalamocorticalAxons.ToArray();
        TopHashLength_M_P_ThalamocorticalAxons = m_P_ThalamocorticalAxons.OrderBy(a => MathHelper.GetLengthXY(a.Root.Position)).Take(constants.HashLength).ToArray();
    }
}
