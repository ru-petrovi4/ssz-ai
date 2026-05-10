using System.Numerics;
using System.Runtime.CompilerServices;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
//  АКСОН
//  Биологический аксон нейрона коры мозга.
// ============================================================
public class Axon
{
    public Axon(AxonPoint root, Synapse[] synapses)
    {        
        Root = root;
        Synapses = synapses;
    }

    /// <summary>
    /// Корневой узел дерева аксона — точка начала (AIS, у сомы).
    /// Всё дерево обходится через Next-ссылки.
    /// </summary>
    public readonly AxonPoint Root;

    /// <summary>
    /// Все исходящие синапсы этого аксона.    
    /// </summary>
    public readonly Synapse[] Synapses;    

    public bool Temp_IsActive;
}

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

// ============================================================
//  СИНАПС
//  Хранит координату синаптического контакта (мкм).
//  System.Numerics.Vector3 — SIMD-совместимый тип.
// ============================================================
public sealed class Synapse
{
    /// <summary>3D-координата синапса в мкм.</summary>
    public Vector3 Position;

    public Synapse(Vector3 position)
    {
        Position = position;
    }
}