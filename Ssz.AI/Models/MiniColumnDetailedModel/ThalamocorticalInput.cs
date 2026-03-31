using System;
using System.Collections.Generic;
using System.Numerics;
using Ssz.Utils;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
//  ТАЛАМОКОРТИКАЛЬНЫЙ ВХОДНОЙ БЛОК МИНИКОЛОНКИ
//
//  Содержит ВСЕ афферентные ЛКТ-аксоны, синапсы которых
//  попадают в данную миниколонку первичной зрительной коры V1.
//
//  Включает два класса аксонов:
//
//  1. «Собственные» аксоны (OwnAxons) — аксоны, чей ствол
//     поднимается из WM непосредственно под данной колонкой.
//     Точка входа равномерно случайна внутри columnRadiusUm.
//
//  2. «Соседские» аксоны (NeighborAxons) — полностью смоделированные
//     ТК-аксоны, чей геометрический центр арбора принадлежит соседней
//     колонке, но сам арбор (из-за большого радиуса) перекрывается
//     с данной колонкой.
//
//     Биологическая основа:
//       - M-арбор: диаметр ~600 мкм → один M-аксон охватывает
//         ~300 миниколонок (диаметр 40 мкм каждая).
//         → аксон, чей ствол входит на расстоянии до 280 мкм от центра
//         данной колонки, может иметь синапсы внутри неё.
//       - P-арбор: диаметр ~350 мкм → радиус перекрытия до 155 мкм.
//       - K-арбор: диаметр ~250 мкм → радиус перекрытия до 105 мкм.
//
//     Число «соседских» аксонов каждого типа оценивается как
//     число аксонов в кольцевой зоне [columnRadius .. arborRadius].
//     Источник: Blasdel & Lund 1983; García-Marín et al. 2019
//
//  Численные оценки «собственных» аксонов на миниколонку:
//    M: 4, P: 10, K: 3  (итого 17)
//    Источник: Callaway 1998; García-Marín et al. 2019
//
//  Численные оценки «соседских» аксонов:
//    Площадь кольца [R_col .. R_arbor]:
//      M: π(300²-20²)/π(300²) ≈ 0.996 × (M на колонку × кол-во колонок
//         в кольце) — упрощённо берём 8 ближайших позиций × 4 = 32
//         M-аксона, но большинство имеют лишь несколько синапсов внутри.
//         Для модели достаточно взять ~8 соседских M-аксонов (из
//         непосредственно примыкающих колонок).
//    Аналогично: P-соседских ≈ 20, K-соседских ≈ 6.
//
//  Соседские аксоны моделируются ПОЛНОСТЬЮ (от WM снизу),
//  с точкой входа в кольце [columnRadiusUm .. arborRadius].
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
    //  КОЛИЧЕСТВО СОБСТВЕННЫХ ТК-АКСОНОВ ПО ТИПАМ
    // ----------------------------------------------------------

    /// <summary>Число собственных магноцеллюлярных (M) аксонов.</summary>
    public const int OwnMAxonCount = 4;

    /// <summary>Число собственных парвоцеллюлярных (P) аксонов.</summary>
    public const int OwnPAxonCount = 10;

    /// <summary>Число собственных конийоцеллюлярных (K) аксонов.</summary>
    public const int OwnKAxonCount = 3;

    // ----------------------------------------------------------
    //  КОЛИЧЕСТВО СОСЕДСКИХ ТК-АКСОНОВ ПО ТИПАМ
    //
    //  Соседские аксоны принадлежат соседним колонкам,
    //  но их арборы (из-за большого диаметра) перекрывают данную.
    //  Оценки основаны на плотности и радиусах арборов:
    //    M (радиус 300 мкм): ~8 аксонов из ближайших колонок
    //    P (радиус 175 мкм): ~20 аксонов
    //    K (радиус 125 мкм): ~6 аксонов
    // ----------------------------------------------------------

    /// <summary>Число соседских M-аксонов, перекрывающих данную колонку.</summary>
    public const int NeighborMAxonCount = 8;

    /// <summary>Число соседских P-аксонов, перекрывающих данную колонку.</summary>
    public const int NeighborPAxonCount = 20;

    /// <summary>Число соседских K-аксонов, перекрывающих данную колонку.</summary>
    public const int NeighborKAxonCount = 6;

    // ----------------------------------------------------------
    //  ИТОГОВЫЕ СЧЁТЧИКИ
    // ----------------------------------------------------------

    /// <summary>Общее число собственных ТК-аксонов.</summary>
    public const int OwnAxonCount = OwnMAxonCount + OwnPAxonCount + OwnKAxonCount;

    /// <summary>Общее число соседских ТК-аксонов.</summary>
    public const int NeighborAxonCount = NeighborMAxonCount + NeighborPAxonCount + NeighborKAxonCount;

    /// <summary>Суммарное число всех ТК-аксонов (собственных + соседских).</summary>
    public const int TotalAxonCount = OwnAxonCount + NeighborAxonCount;

    // ----------------------------------------------------------
    //  ДАННЫЕ
    // ----------------------------------------------------------

    /// <summary>Все ТК-аксоны (собственные + соседские), единый массив.</summary>
    public readonly ThalamocorticalAxon[] Axons;

    /// <summary>Только собственные M-аксоны.</summary>
    public readonly ThalamocorticalAxon[] OwnMAxons;

    /// <summary>Только собственные P-аксоны.</summary>
    public readonly ThalamocorticalAxon[] OwnPAxons;

    /// <summary>Только собственные K-аксоны.</summary>
    public readonly ThalamocorticalAxon[] OwnKAxons;

    /// <summary>
    /// Все соседские аксоны (M + P + K из соседних колонок).
    /// Моделируются полностью, от WM снизу, с точкой входа
    /// в кольце columnRadius..arborRadius.
    /// </summary>
    public readonly ThalamocorticalAxon[] NeighborAxons;

    // ============================================================
    //  КОНСТРУКТОР
    // ============================================================
    /// <summary>
    /// Генерирует все таламокортикальные входящие аксоны для миниколонки:
    /// собственные (17 штук) и соседские (34 штуки).
    /// Соседские аксоны моделируются полностью с самого низа (WM),
    /// их точка входа смещена в кольцо [columnRadius .. arborRadius].
    /// </summary>
    /// <param name="random">Генератор случайных чисел.</param>
    /// <param name="columnRadiusUm">Радиус миниколонки (мкм).</param>
    /// <param name="columnHeightUm">Высота миниколонки (мкм).</param>
    public ThalamocorticalInput(Random random, float columnRadiusUm, float columnHeightUm)
    {
        Axons        = new ThalamocorticalAxon[TotalAxonCount];
        OwnMAxons    = new ThalamocorticalAxon[OwnMAxonCount];
        OwnPAxons    = new ThalamocorticalAxon[OwnPAxonCount];
        OwnKAxons    = new ThalamocorticalAxon[OwnKAxonCount];
        NeighborAxons = new ThalamocorticalAxon[NeighborAxonCount];

        int globalIdx   = 0; // сквозной индекс в Axons[]
        int neighborIdx = 0; // индекс в NeighborAxons[]

        // --- Собственные M-аксоны ---
        // Точка входа внутри columnRadiusUm — аксон «под своей» колонкой
        for (int i = 0; i < OwnMAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.Magnocellular,
                random,
                columnRadiusUm,
                columnHeightUm,
                entryInNeighborRing: false);

            Axons[globalIdx] = axon;
            OwnMAxons[i]     = axon;
            globalIdx        += 1;
        }

        // --- Собственные P-аксоны ---
        for (int i = 0; i < OwnPAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.Parvocellular,
                random,
                columnRadiusUm,
                columnHeightUm,
                entryInNeighborRing: false);

            Axons[globalIdx] = axon;
            OwnPAxons[i]     = axon;
            globalIdx        += 1;
        }

        // --- Собственные K-аксоны ---
        for (int i = 0; i < OwnKAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.Koniocellular,
                random,
                columnRadiusUm,
                columnHeightUm,
                entryInNeighborRing: false);

            Axons[globalIdx] = axon;
            OwnKAxons[i]     = axon;
            globalIdx        += 1;
        }

        // --- Соседские M-аксоны ---
        // entryInNeighborRing: true — точка входа в WM смещена в кольцо
        // [columnRadiusUm .. MArborRadius], т.е. аксон принадлежит
        // соседней колонке, но его арбор перекрывается с данной.
        for (int i = 0; i < NeighborMAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.Magnocellular,
                random,
                columnRadiusUm,
                columnHeightUm,
                entryInNeighborRing: true);

            Axons[globalIdx]         = axon;
            NeighborAxons[neighborIdx] = axon;
            globalIdx                += 1;
            neighborIdx              += 1;
        }

        // --- Соседские P-аксоны ---
        for (int i = 0; i < NeighborPAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.Parvocellular,
                random,
                columnRadiusUm,
                columnHeightUm,
                entryInNeighborRing: true);

            Axons[globalIdx]           = axon;
            NeighborAxons[neighborIdx] = axon;
            globalIdx                  += 1;
            neighborIdx                += 1;
        }

        // --- Соседские K-аксоны ---
        for (int i = 0; i < NeighborKAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.Koniocellular,
                random,
                columnRadiusUm,
                columnHeightUm,
                entryInNeighborRing: true);

            Axons[globalIdx]           = axon;
            NeighborAxons[neighborIdx] = axon;
            globalIdx                  += 1;
            neighborIdx                += 1;
        }
    }

    // ============================================================
    //  АКТИВАЦИЯ ВХОДЯЩЕГО ЗРИТЕЛЬНОГО СИГНАЛА
    // ============================================================
    /// <summary>
    /// Устанавливает активность собственных ТК-аксонов на основе
    /// вектора входящего зрительного сигнала.
    /// Соседские аксоны управляются отдельно через SetNeighborActivity.
    /// </summary>
    /// <param name="mActivity">Активность M-аксонов (длина OwnMAxonCount).</param>
    /// <param name="pActivity">Активность P-аксонов (длина OwnPAxonCount).</param>
    /// <param name="kActivity">Активность K-аксонов (длина OwnKAxonCount).</param>
    public void SetActivity(float[] mActivity, float[] pActivity, float[] kActivity)
    {
        for (int i = 0; i < OwnMAxonCount; i += 1)
            OwnMAxons[i].Temp_IsActive = mActivity[i] > 0.5f;

        for (int i = 0; i < OwnPAxonCount; i += 1)
            OwnPAxons[i].Temp_IsActive = pActivity[i] > 0.5f;

        for (int i = 0; i < OwnKAxonCount; i += 1)
            OwnKAxons[i].Temp_IsActive = kActivity[i] > 0.5f;
    }

    /// <summary>
    /// Устанавливает активность соседских ТК-аксонов.
    /// Вызывается при симуляции активности соседних миниколонок.
    /// </summary>
    /// <param name="neighborActivity">
    /// Массив длиной NeighborAxonCount: 1.0 = активен, 0.0 = покой.
    /// </param>
    public void SetNeighborActivity(float[] neighborActivity)
    {
        for (int i = 0; i < NeighborAxonCount; i += 1)
            NeighborAxons[i].Temp_IsActive = neighborActivity[i] > 0.5f;
    }

    // ============================================================
    //  СБОР АКТИВНЫХ СИНАПТИЧЕСКИХ ПОЗИЦИЙ
    // ============================================================
    /// <summary>
    /// Собирает позиции всех активных синапсов (собственных и соседских).
    /// Используется в FindActiveZones для расчёта зон активации.
    /// </summary>
    /// <returns>Список позиций активных ТК-синапсов внутри колонки.</returns>
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
