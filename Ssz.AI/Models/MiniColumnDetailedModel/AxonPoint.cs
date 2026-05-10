using Ssz.Utils;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
//  ТОЧКА АКСОНА
//  Хранит 3D-координату (x, y, z) в микрометрах (мкм).
//  Использует System.Numerics.Vector3 для SIMD-ускорения
//  операций с координатами.
//  Следующие точки (дочерние узлы) — список, т.к. может быть
//  бинарное ветвление: 0, 1 или 2 следующих узла.
// ============================================================
public sealed class AxonPoint
{
    public AxonPoint(Vector3 position)
    {
        Position = position;
    }

    /// <summary>3D-координата в микрометрах (мкм).</summary>
    public Vector3 Position;

    public int NextCount;

    public NextAxonPoints Next;

    public void AddNext(AxonPoint nextAxonPoint)
    {
        Next[NextCount] = nextAxonPoint;
        NextCount += 1;
    }
}

[InlineArray(Size)]
public struct NextAxonPoints
{
    public const int Size = 2;

    private AxonPoint _element0; // остальное генерируется компилятором
}
