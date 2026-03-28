using System;
using System.Collections.Generic;
using System.Numerics;
using Ssz.Utils;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
//  ТАЛАМОКОРТИКАЛЬНЫЙ ВХОДНОЙ БЛОК МИНИКОЛОНКИ
//
//  Содержит все афферентные ЛКТ-аксоны, проецирующиеся
//  в данную миниколонку первичной зрительной коры V1.
//
//  Численные оценки количества ТК-аксонов на миниколонку:
//
//    Всего LGN-нейронов (один глаз) ≈ 1.0–1.5 млн (человек)
//    Площадь V1 ≈ 2500–3000 мм² (один полушарий)
//    Площадь одной миниколонки ≈ π × 0.02² мм² ≈ 0.00126 мм²
//    Миниколонок в V1 ≈ 2500 / 0.00126 ≈ ~2 млн
//
//    Один магноцеллюлярный аксон покрывает ~0.3–0.4 мм²
//    → ~300 миниколонок пересекает один аксон
//    → на одну миниколонку приходится 1.5 M / (300 M×ratio)
//
//    Практическая оценка (Blasdel & Lund 1983; García-Marín 2019):
//      M-аксонов на миниколонку:  ~3–5
//      P-аксонов на миниколонку:  ~8–15
//      K-аксонов на миниколонку:  ~2–4
//
//    Здесь используем консервативные значения для одного
//    глаза/поля зрения (монокулярный вход):
//      M: 4, P: 10, K: 3  (итого 17 ТК-аксонов)
//
//  Источники:
//    - Blasdel & Lund 1983, J. Neurosci. 3:1389–1413
//    - García-Marín et al. 2019, Cereb. Cortex 29:134–151
//    - Callaway 1998, Annu. Rev. Neurosci. 21:47–74
//    - Hendry & Reid 2000, Annu. Rev. Neurosci. 23:127–153
// ============================================================
public sealed class ThalamocorticalInput
{
    // ----------------------------------------------------------
    //  КОЛИЧЕСТВО ТК-АКСОНОВ ПО ТИПАМ (на одну миниколонку)
    // ----------------------------------------------------------

    /// <summary>Число магноцеллюлярных (M) аксонов на миниколонку.</summary>
    public const int MAxonCount = 4;

    /// <summary>Число парвоцеллюлярных (P) аксонов на миниколонку.</summary>
    public const int PAxonCount = 10;

    /// <summary>Число конийоцеллюлярных (K) аксонов на миниколонку.</summary>
    public const int KAxonCount = 3;

    /// <summary>Общее число таламокортикальных афферентных аксонов.</summary>
    public const int TotalAxonCount = MAxonCount + PAxonCount + KAxonCount;

    // ----------------------------------------------------------
    //  ДАННЫЕ
    // ----------------------------------------------------------

    /// <summary>Все таламокортикальные афферентные аксоны.</summary>
    public readonly ThalamocorticalAxon[] Axons;

    /// <summary>Только магноцеллюлярные аксоны (быстрый доступ).</summary>
    public readonly ThalamocorticalAxon[] MAxons;

    /// <summary>Только парвоцеллюлярные аксоны (быстрый доступ).</summary>
    public readonly ThalamocorticalAxon[] PAxons;

    /// <summary>Только конийоцеллюлярные аксоны (быстрый доступ).</summary>
    public readonly ThalamocorticalAxon[] KAxons;

    // ============================================================
    //  КОНСТРУКТОР
    // ============================================================
    /// <summary>
    /// Генерирует все таламокортикальные входящие аксоны для миниколонки.
    /// </summary>
    /// <param name="random">Генератор случайных чисел.</param>
    /// <param name="columnRadiusUm">Радиус миниколонки (мкм).</param>
    /// <param name="columnHeightUm">Высота миниколонки (мкм).</param>
    public ThalamocorticalInput(Random random, float columnRadiusUm, float columnHeightUm)
    {
        Axons  = new ThalamocorticalAxon[TotalAxonCount];
        MAxons = new ThalamocorticalAxon[MAxonCount];
        PAxons = new ThalamocorticalAxon[PAxonCount];
        KAxons = new ThalamocorticalAxon[KAxonCount];

        int globalIdx = 0;

        // Генерируем M-аксоны
        for (int i = 0; i < MAxonCount; i += 1)
        {
            var axon          = ThalamocorticalAxon.Generate(
                globalIdx, ThalamocorticalType.Magnocellular,
                random, columnRadiusUm, columnHeightUm);
            Axons[globalIdx]  = axon;
            MAxons[i]         = axon;
            globalIdx        += 1;
        }

        // Генерируем P-аксоны
        for (int i = 0; i < PAxonCount; i += 1)
        {
            var axon          = ThalamocorticalAxon.Generate(
                globalIdx, ThalamocorticalType.Parvocellular,
                random, columnRadiusUm, columnHeightUm);
            Axons[globalIdx]  = axon;
            PAxons[i]         = axon;
            globalIdx        += 1;
        }

        // Генерируем K-аксоны
        for (int i = 0; i < KAxonCount; i += 1)
        {
            var axon          = ThalamocorticalAxon.Generate(
                globalIdx, ThalamocorticalType.Koniocellular,
                random, columnRadiusUm, columnHeightUm);
            Axons[globalIdx]  = axon;
            KAxons[i]         = axon;
            globalIdx        += 1;
        }
    }

    // ============================================================
    //  АКТИВАЦИЯ ВХОДЯЩЕГО ЗРИТЕЛЬНОГО СИГНАЛА
    // ============================================================
    /// <summary>
    /// Устанавливает активность таламокортикальных аксонов
    /// на основе вектора входящего зрительного сигнала.
    /// </summary>
    /// <param name="mActivity">
    ///   Активность M-аксонов: массив длиной MAxonCount,
    ///   значения 0.0 (покой) или 1.0 (активен).
    /// </param>
    /// <param name="pActivity">
    ///   Активность P-аксонов: массив длиной PAxonCount.
    /// </param>
    /// <param name="kActivity">
    ///   Активность K-аксонов: массив длиной KAxonCount.
    /// </param>
    public void SetActivity(float[] mActivity, float[] pActivity, float[] kActivity)
    {
        for (int i = 0; i < MAxonCount; i += 1)
            MAxons[i].Temp_IsActive = mActivity[i] > 0.5f;

        for (int i = 0; i < PAxonCount; i += 1)
            PAxons[i].Temp_IsActive = pActivity[i] > 0.5f;

        for (int i = 0; i < KAxonCount; i += 1)
            KAxons[i].Temp_IsActive = kActivity[i] > 0.5f;
    }

    /// <summary>
    /// Собирает все активные синаптические бутоны входящих ТК-аксонов.
    /// Используется для определения зон активации в FindActiveZones.
    /// </summary>
    /// <returns>Список позиций активных ТК-синапсов внутри миниколонки.</returns>
    public FastList<Vector3> GetActiveAfferentSynapsePositions()
    {
        var result = new FastList<Vector3>(TotalAxonCount * 1000);

        for (int a = 0; a < TotalAxonCount; a += 1)
        {
            if (!Axons[a].Temp_IsActive)
                continue;

            var synapses = Axons[a].Synapses;
            for (int s = 0; s < synapses.Length; s += 1)
                result.Add(synapses[s].Position);
        }

        return result;
    }
}
