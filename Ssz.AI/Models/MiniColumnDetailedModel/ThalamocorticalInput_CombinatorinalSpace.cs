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
    ///
    /// ТК-аксоны
    ///
    public readonly Axon[] ThalamocorticalAxons;

    // ============================================================
    // КОНСТРУКТОР
    // ============================================================

    ///
    /// Генерирует все таламокортикальные входящие аксоны.   
    ///
    /// <param name="random">Генератор случайных чисел.</param>
    /// <param name="miniColumnRadiusUm">Радиус миниколонки (мкм).</param>
    /// <param name="miniColumnHeightUm">Высота миниколонки (мкм).</param>
    public ThalamocorticalInput_CombinatorinalSpace(Random random, float miniColumnRadiusUm, float miniColumnHeightUm, IRetinaConstants constants)
    {
        ThalamocorticalAxons = null!;
    }
}
