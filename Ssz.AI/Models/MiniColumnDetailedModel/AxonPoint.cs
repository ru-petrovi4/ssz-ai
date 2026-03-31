using Ssz.Utils;
using System.Collections.Generic;
using System.Numerics;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
//  ТОЧКА АКСОНА
//  Хранит 3D-координату (x, y, z) в микрометрах (мкм).
//  Использует System.Numerics.Vector3 для SIMD-ускорения
//  операций с координатами.
//  Следующие точки (дочерние узлы) — список, т.к. может быть
//  бинарное ветвление: обычно 0, 1 или 2 следующих узла.
// ============================================================
public sealed class AxonPoint
{
    /// <summary>3D-координата в микрометрах (мкм).</summary>
    public Vector3 Position;

    /// <summary>
    /// Список следующих точек (дочерних узлов).
    /// Пустой список — терминальная точка (конец ветки аксона).
    /// 1 элемент — прямое продолжение без ветвления.
    /// 2 элемента — бинарное ветвление.
    /// </summary>
    public readonly FastList<AxonPoint> Next = new(capacity: 2);

    public AxonPoint(Vector3 position)
    {
        Position = position;
    }
}
