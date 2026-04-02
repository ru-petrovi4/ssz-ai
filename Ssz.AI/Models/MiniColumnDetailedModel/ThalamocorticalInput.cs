using System;
using System.Collections.Generic;
using System.Numerics;
using Ssz.Utils;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
//  ТАЛАМОКОРТИКАЛЬНЫЙ ВХОДНОЙ БЛОК МИНИКОЛОНКИ
// ============================================================

public sealed class ThalamocorticalInput
{
    // ----------------------------------------------------------
    //  КОЛИЧЕСТВО СОБСТВЕННЫХ ТК-АКСОНОВ ПО ТИПАМ
    // ----------------------------------------------------------

    /// <summary>Число магноцеллюлярных (M) аксонов.</summary>
    public const int MAxonCount = 100;

    /// <summary>Число парвоцеллюлярных (P) аксонов.</summary>
    public const int PAxonCount = 78;

    /// <summary>Число конийоцеллюлярных (K) аксонов.</summary>
    public const int KAxonCount = 25;

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

    // ----------------------------------------------------------
    //  ИТОГОВЫЙ СЧЁТЧИК
    // ----------------------------------------------------------

    /// <summary>Суммарное число всех ТК-аксонов.</summary>
    public const int TotalAxonCount = MAxonCount + PAxonCount + KAxonCount;

    // ----------------------------------------------------------
    //  ДАННЫЕ
    // ----------------------------------------------------------

    /// <summary>
    /// Все ТК-аксоны (собственные + соседские), единый массив.
    /// Соседские аксоны (M + P + K из соседних колонок) моделируются полностью, от WM снизу, с точкой входа
    /// в кольце columnRadius..arborRadius.
    /// </summary>
    public readonly ThalamocorticalAxon[] ThalamocorticalAxons;    

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
        ThalamocorticalAxons        = new ThalamocorticalAxon[TotalAxonCount];        

        int globalIdx   = 0; // сквозной индекс в Axons[]        

        // --- M-аксоны ---
        // Точка входа внутри columnRadiusUm — аксон «под своей» колонкой
        for (int i = 0; i < MAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.Magnocellular,
                random,
                columnRadiusUm,
                columnHeightUm);

            ThalamocorticalAxons[globalIdx] = axon;            
            globalIdx        += 1;
        }

        // --- P-аксоны ---
        for (int i = 0; i < PAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.Parvocellular,
                random,
                columnRadiusUm,
                columnHeightUm);

            ThalamocorticalAxons[globalIdx] = axon;            
            globalIdx        += 1;
        }

        // --- K-аксоны ---
        for (int i = 0; i < KAxonCount; i += 1)
        {
            var axon = ThalamocorticalAxon.Generate(
                globalIdx,
                ThalamocorticalType.Koniocellular,
                random,
                columnRadiusUm,
                columnHeightUm);

            ThalamocorticalAxons[globalIdx] = axon;            
            globalIdx        += 1;
        }
    }

    // ============================================================
    //  АКТИВАЦИЯ ВХОДЯЩЕГО ЗРИТЕЛЬНОГО СИГНАЛА
    // ============================================================
    /// <summary>
    /// Устанавливает активность ТК-аксонов на основе
    /// вектора входящего зрительного сигнала.
    /// Соседские аксоны управляются отдельно через SetNeighborActivity.
    /// </summary>
    /// <param name="mActivity">Активность M-аксонов (длина OwnMAxonCount).</param>
    /// <param name="pActivity">Активность P-аксонов (длина OwnPAxonCount).</param>
    /// <param name="kActivity">Активность K-аксонов (длина OwnKAxonCount).</param>
    public void SetActivity(float[] activity)
    {
        for (int i = 0; i < activity.Length; i += 1)
            ThalamocorticalAxons[i].Temp_IsActive = activity[i] > 0.5f;
    }

    // ============================================================
    //  СБОР АКТИВНЫХ СИНАПТИЧЕСКИХ ПОЗИЦИЙ
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
